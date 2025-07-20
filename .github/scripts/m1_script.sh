#!/bin/bash

#
# Copyright (c) 2025
#

export BUILD_VERSION=0.4.0

${CONDA_RUN} pip install git+https://github.com/pytorch/tensordict.git -U
