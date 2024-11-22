# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D parallelisms (except pipeline parallelism) and various
# training techniques (e.g. activation checkpointing and compile) to the NanoGPT model.

from collections import defaultdict

import torch
import torch.nn as nn

from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import (
    # CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
)
from torch.distributed._composable.replicate import replicate
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)


def parallelize_nanogpt(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """

    # TODO - AC if job_config.activation_checkpoint.mode != "none":
    # apply_ac(model, job_config.activation_checkpoint)

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    """if job_config.training.compile:
        if job_config.model.norm_type == "fused_rmsnorm":
            raise NotImplementedError(
                "fused_rmsnorm is not compatible with torch.compile yet. "
                "Please use rmsnorm or layernorm."
            )
        apply_compile(model)
    """

    # apply FSDP or HSDP, potentially with Context Parallel
    try:
        dp_mesh = world_mesh["dp"]
    except IndexError:
        # note: this is a workaround of the above logic for old pytorch version
        # where https://github.com/pytorch/pytorch/pull/138945 is not included
        # throw a warning to encourage users to upgrade to a newer pytorch version
        dp_mesh_dim_names = ("dp",)

        # note that mesh can only be flattened from the finest-grained mesh dimensions
        dp_mesh = world_mesh[dp_mesh_dim_names]

    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )

    print("Applied FSDP to the model")


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
):
    """
    Apply data parallelism to the model. FSDP2 is used here.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    for layer_id, transformer_block in model.layers.items():

        # As an optimization, do not reshard after forward for the last
        # transformer block since FSDP would prefetch it immediately
        reshard_after_forward = int(layer_id) < len(model.layers) - 1
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(model, **fsdp_config, reshard_after_forward=True)
