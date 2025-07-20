#  Copyright (c) 2025
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

warnings.warn("vec_env.py has moved to batch_envs.py.", category=DeprecationWarning)

from .batched_envs import *  # noqa: F403, F401
