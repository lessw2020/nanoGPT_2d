# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D parallelisms (except pipeline parallelism) and various
# training techniques (e.g. activation checkpointing and compile) to the Llama model.

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
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)


def parallelize_nanogpt(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if job_config.training.compile:
        if job_config.model.norm_type == "fused_rmsnorm":
            raise NotImplementedError(
                "fused_rmsnorm is not compatible with torch.compile yet. "
                "Please use rmsnorm or layernorm."
            )
        apply_compile(model)

    if (
        parallel_dims.dp_shard_enabled
    ):  # apply FSDP or HSDP, potentially with Context Parallel
        try:
            dp_mesh = (
                world_mesh["dp_cp"] if parallel_dims.cp_enabled else world_mesh["dp"]
            )
        except IndexError:
            # note: this is a workaround of the above logic for old pytorch version
            # where https://github.com/pytorch/pytorch/pull/138945 is not included
            # throw a warning to encourage users to upgrade to a newer pytorch version
            check_if_feature_in_pytorch(
                "DeviceMesh flattening over 3D+ meshes",
                "https://github.com/pytorch/pytorch/pull/138945",
                "2.6.0.dev20241030",
            )
            # TODO: remove this workaround once PyTorch 2.6 is released
            dp_mesh_dim_names = ("dp",)

            # note that mesh can only be flattened from the finest-grained mesh dimensions
            dp_mesh = world_mesh[dp_mesh_dim_names]

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            tp_enabled=parallel_dims.tp_enabled,
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the model")
    elif parallel_dims.dp_replicate_enabled:
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            world_mesh,
            enable_compile=job_config.training.compile,
            enable_compiled_autograd=job_config.experimental.enable_compiled_autograd,
        )
