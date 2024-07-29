# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time

import hydra
import torch

import torchrl.modules
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor

from tensordict.utils import expand_as_right
from torch import nn

from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    CompositeSpec,
    TensorDictReplayBuffer,
    UnboundedContinuousTensorSpec,
)
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    InitTracker,
    RewardSum,
    TensorDictPrimer,
    TransformedEnv,
)
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from utils.logging import init_logging, log_evaluation, log_training
from utils.utils import DoneTransform


def rendering_callback(env, td):
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=None))


class MultiAgentGRU(torch.nn.Module):
    def __init__(self, training, input_size, hidden_size, n_agents, device, gru=None):
        super().__init__()
        self.training = training
        self.input_size = input_size
        self.n_agents = n_agents
        self.hidden_size = hidden_size
        self.device = device
        if gru is None:
            self.gru = torchrl.modules.GRUCell(
                input_size, hidden_size, device=self.device
            )
        else:
            self.gru = gru

        self.vmap_rnn = self.get_for_loop(self.gru)
        # self.vmap_rnn_compiled = torch.compile(
        #     self.vmap_rnn, mode="reduce-overhead", fullgraph=True
        # )

    def get_training(self):
        return MultiAgentGRU(
            training=True,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            n_agents=self.n_agents,
            gru=self.gru,
            device=self.device,
        )

    def forward(
        self,
        input,
        is_init,
        h_0=None,
    ):
        assert is_init is not None, "We need to pass is_init"
        if (
            not self.training
        ):  # In collection we emulate the sequence dimension and we have the hidden state
            input = input.unsqueeze(1)
            assert h_0 is not None
        else:  # In training we have the sequence dimension and we do not have an initial state
            assert h_0 is None

        # Check input
        batch = input.shape[0]
        seq = input.shape[1]
        assert input.shape == (batch, seq, self.n_agents, self.input_size)

        if h_0 is not None:  # Collection
            assert h_0.shape == (
                batch,
                self.n_agents,
                self.hidden_size,
            )
            if is_init is not None:  # Set hidden to 0 when is_init
                h_0 = torch.where(expand_as_right(is_init, h_0), 0, h_0)

        if not self.training:  # If in collection emulate the sequence dimension
            is_init = is_init.unsqueeze(1)
        assert is_init.shape == (batch, seq, 1)
        is_init = is_init.unsqueeze(-2).expand(batch, seq, self.n_agents, 1)

        if h_0 is None:
            h_0 = torch.zeros(
                batch,
                self.n_agents,
                self.hidden_size,
                device=self.device,
                dtype=torch.float,
            )
        output = self.vmap_rnn(input, is_init, h_0)
        h_n = output[..., -1, :, :]

        if not self.training:
            output = output.squeeze(1)
        return output, h_n

    # @torch.compile(mode="reduce-overhead", fullgraph=True)

    @staticmethod
    def get_for_loop(rnn):
        def for_loop(input, is_init, h, time_dim=-3):
            hs = []
            for in_t, init_t in zip(input.unbind(time_dim), is_init.unbind(time_dim)):
                h = torch.where(init_t, 0, h)
                h = rnn(in_t, h)
                hs.append(h)
            output = torch.stack(hs, time_dim)
            return output

        return torch.vmap(for_loop)


@hydra.main(version_base="1.1", config_path="", config_name="mappo_ippo")
def train(cfg: "DictConfig"):  # noqa: F821
    # Device
    cfg.train.device = "cpu" if not torch.cuda.device_count() else "cuda:0"
    cfg.env.device = cfg.train.device

    # Seeding
    torch.manual_seed(cfg.seed)

    # Sampling
    cfg.env.vmas_envs = cfg.collector.frames_per_batch // cfg.env.max_steps
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    collector_time_dim = -(-cfg.collector.frames_per_batch // cfg.env.vmas_envs)
    cfg.buffer.memory_size = cfg.env.vmas_envs

    # Create env and env_test
    env = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.env.vmas_envs,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        # Scenario kwargs
        **cfg.env.scenario,
    )

    def get_rnn_transforms(env):
        transforms = [
            InitTracker(init_key="is_init"),
            TensorDictPrimer(
                {
                    "agents": CompositeSpec(
                        {
                            "_h": UnboundedContinuousTensorSpec(
                                shape=(*env.shape, env.n_agents, 128)
                            )
                        },
                        shape=(*env.shape, env.n_agents),
                    )
                }
            ),
        ]
        return transforms

    env = TransformedEnv(
        env,
        Compose(
            *(
                (get_rnn_transforms(env) if cfg.model.use_rnn else [])
                + [
                    RewardSum(
                        in_keys=[env.reward_key],
                        out_keys=[("agents", "episode_reward")],
                    )
                ]
            )
        ),
    )

    env_test = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.eval.evaluation_episodes,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        # Scenario kwargs
        **cfg.env.scenario,
    )
    env_test = TransformedEnv(
        env_test, Compose(*(get_rnn_transforms(env_test) if cfg.model.use_rnn else []))
    )
    gru = MultiAgentGRU(
        training=False,
        input_size=env.observation_spec["agents", "observation"].shape[-1],
        n_agents=env.n_agents,
        hidden_size=128,
        device=cfg.train.device,
    )

    gru_module = TensorDictModule(
        gru,
        in_keys=[
            ("agents", "observation"),
            "is_init",
            ("agents", "_h"),
        ],
        out_keys=[("agents", "intermediate"), ("next", "agents", "_h")],
    )
    gru_module_training = TensorDictModule(
        gru.get_training(),
        in_keys=[("agents", "observation"), "is_init"],
        out_keys=[("agents", "intermediate"), "_"],
    )

    # Policy
    policy_module = TensorDictModule(
        nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=128
                if cfg.model.use_rnn
                else env.observation_spec["agents", "observation"].shape[-1],
                n_agent_outputs=2 * env.action_spec.shape[-1],
                n_agents=env.n_agents,
                centralised=False,
                share_params=cfg.model.shared_parameters,
                device=cfg.train.device,
                depth=2,
                num_cells=256,
                activation_class=nn.Tanh,
            ),
            NormalParamExtractor(),
        ),
        in_keys=[("agents", "intermediate")]
        if cfg.model.use_rnn
        else [("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )

    actor_module = (
        TensorDictSequential(gru_module, policy_module)
        if cfg.model.use_rnn
        else policy_module
    )
    actor_module_training = (
        TensorDictSequential(gru_module_training, policy_module)
        if cfg.model.use_rnn
        else policy_module
    )

    policy = ProbabilisticActor(
        module=actor_module,
        spec=env.unbatched_action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.unbatched_action_spec[("agents", "action")].space.low,
            "high": env.unbatched_action_spec[("agents", "action")].space.high,
        },
        return_log_prob=True,
    )
    policy_training = ProbabilisticActor(
        module=actor_module_training,
        spec=env.unbatched_action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.unbatched_action_spec[("agents", "action")].space.low,
            "high": env.unbatched_action_spec[("agents", "action")].space.high,
        },
        return_log_prob=True,
    )

    # Critic
    module = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,
        n_agents=env.n_agents,
        centralised=cfg.model.centralised_critic,
        share_params=cfg.model.shared_parameters,
        device=cfg.train.device,
        depth=2,
        num_cells=256,
        activation_class=nn.Tanh,
    )
    value_module = ValueOperator(
        module=module,
        in_keys=[("agents", "observation")],
    )

    collector = SyncDataCollector(
        env,
        policy,
        device=cfg.env.device,
        storing_device=cfg.train.device,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        postproc=DoneTransform(reward_key=env.reward_key, done_keys=env.done_keys),
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.buffer.memory_size, device=cfg.train.device),
        sampler=SamplerWithoutReplacement(),
        batch_size=-(-cfg.train.minibatch_size // collector_time_dim),
    )

    # Loss
    loss_module = ClipPPOLoss(
        actor_network=policy_training,
        critic_network=value_module,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_coef=cfg.loss.entropy_eps,
        normalize_advantage=False,
    )
    loss_module.set_keys(
        reward=env.reward_key,
        action=env.action_key,
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )
    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=cfg.loss.gamma, lmbda=cfg.loss.lmbda
    )
    optim = torch.optim.Adam(loss_module.parameters(), cfg.train.lr)

    # Logging
    if cfg.logger.backend:
        model_name = (
            ("Het" if not cfg.model.shared_parameters else "")
            + ("MA" if cfg.model.centralised_critic else "I")
            + "PPO"
        )
        logger = init_logging(cfg, model_name)

    total_time = 0
    total_frames = 0
    sampling_start = time.time()
    for i, tensordict_data in enumerate(collector):
        torchrl_logger.info(f"\nIteration {i}")

        sampling_time = time.time() - sampling_start

        with torch.no_grad():
            loss_module.value_estimator(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )
        current_frames = tensordict_data.numel()
        total_frames += current_frames
        replay_buffer.extend(tensordict_data)

        training_tds = []
        training_start = time.time()
        for _ in range(cfg.train.num_epochs):
            for _ in range(
                -(-cfg.collector.frames_per_batch // cfg.train.minibatch_size)
            ):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                training_tds.append(loss_vals.detach())

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.train.max_grad_norm
                )
                training_tds[-1].set("grad_norm", total_norm.mean())

                optim.step()
                optim.zero_grad()

        collector.update_policy_weights_()

        training_time = time.time() - training_start

        iteration_time = sampling_time + training_time
        total_time += iteration_time
        training_tds = torch.stack(training_tds)

        # More logs
        if cfg.logger.backend:
            log_training(
                logger,
                training_tds,
                tensordict_data,
                sampling_time,
                training_time,
                total_time,
                i,
                current_frames,
                total_frames,
                step=i,
            )

        if (
            cfg.eval.evaluation_episodes > 0
            and i % cfg.eval.evaluation_interval == 0
            and cfg.logger.backend
        ):
            evaluation_start = time.time()
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                env_test.frames = []
                rollouts = env_test.rollout(
                    max_steps=cfg.env.max_steps,
                    policy=policy,
                    callback=rendering_callback,
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                    # We are running vectorized evaluation we do not want it to stop when just one env is done
                )

                evaluation_time = time.time() - evaluation_start

                log_evaluation(logger, rollouts, env_test, evaluation_time, step=i)

        if cfg.logger.backend == "wandb":
            logger.experiment.log({}, commit=True)
        sampling_start = time.time()


if __name__ == "__main__":
    train()
