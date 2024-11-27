# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [Note] Getting the 'torchao' package:
# This script requires the 'torchao' package to function correctly.
# Please ensure you have this package installed from the appropriate repository.
# You can obtain it from https://github.com/pytorch/ao by following the
# installation instructions.

# Note: Performance
# Float8 experimental is intended to be ran under `torch.compile`` for competitive performance

from typing import List, Union

import torch
import torch.nn as nn
from logging_utils import SingletonLogger


logger = SingletonLogger.get_logger()


def _is_sm89_or_later():
    # Float8 is only supported on SM89 or later (H100+ GPUs)
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


class Float8Handler:
    def __init__(self, use_fp8: bool = True):
        self.enabled = False

        if not _is_sm89_or_later():
            logger.warning(
                "Failed to swap to Float8Linear because float8 is only supported on SM89 or later",
            )
            return
        try:
            from torchao.float8 import CastConfig, Float8LinearConfig, ScalingType
        except ImportError as e:
            raise ImportError(
                "torchao is not installed. Please install it to use float8 linear layers."
            ) from e

        # Mutates the model inplace replacing instances of torch.nn.Linear with Float8Linear

        scaling_type_input = ScalingType("dynamic")
        scaling_type_weight = ScalingType("dynamic")
        scaling_type_grad_output = ScalingType("dynamic")
        self.config = Float8LinearConfig(
            cast_config_input=CastConfig(scaling_type=scaling_type_input),
            cast_config_weight=CastConfig(scaling_type=scaling_type_weight),
            cast_config_grad_output=CastConfig(scaling_type=scaling_type_grad_output),
        )

        self.enabled = True

        # for precompute_float8_dynamic_scale_for_fsdp
        """self.precompute_scale = (
            enable_fsdp_float8_all_gather
            and float8_config.precompute_float8_dynamic_scale_for_fsdp
        )
        """

        # for sync_float8_amax_and_scale_history
        self.delayed_scaling = (
            scaling_type_input is ScalingType.DELAYED
            or scaling_type_weight is ScalingType.DELAYED
            or scaling_type_grad_output is ScalingType.DELAYED
        )
        self._sync_float8_amax_and_scale_history = None
        self.compile = True
        logger.info("Float8 training active")

    def convert_to_float8_training(self, model: nn.Module):
        """
        This function converts the linear layers of `model` to `Float8Linear`.
        Note that today, only dynamic tensor scaling (the default) is supported.
        This will mutate the model inplace.
        """
        if not self.enabled:
            return

        from torchao.float8 import convert_to_float8_training

        # Mutates the model inplace replacing instances of nn.Linear with Float8Linear
        convert_to_float8_training(
            model,
            config=self.config,
            module_filter_fn=lambda mod, fqn: fqn != "lm_head",
        )
        logger.info(
            "Swapped to Float8Linear layers"  #  with enable_fsdp_float8_all_gather="
            # f"{self.config.enable_fsdp_float8_all_gather}"
        )

    def precompute_float8_dynamic_scale_for_fsdp(
        self, model: Union[nn.Module, List[nn.Module]]
    ):
        if not self.enabled:
            return

        if not self.precompute_scale:
            return

        from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

        models = [model] if isinstance(model, nn.Module) else model
        for m in models:
            precompute_float8_dynamic_scale_for_fsdp(m)

    def sync_float8_amax_and_scale_history(
        self, model: Union[nn.Module, List[nn.Module]]
    ):
        if not self.enabled:
            return

        if not self.delayed_scaling:
            return

        from torchao.float8 import sync_float8_amax_and_scale_history

        # TODO(vkuzo): see if precalculating the modules to sync over is going to
        # meaningfully help performance

        if self._sync_float8_amax_and_scale_history is None:
            if self.compile:
                self._sync_float8_amax_and_scale_history = torch.compile(
                    sync_float8_amax_and_scale_history
                )
            else:
                self._sync_float8_amax_and_scale_history = (
                    sync_float8_amax_and_scale_history
                )

        models = [model] if isinstance(model, nn.Module) else model
        for m in models:
            self._sync_float8_amax_and_scale_history(m)
