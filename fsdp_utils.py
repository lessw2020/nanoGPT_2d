import contextlib
import gc
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import timedelta
from typing import Generator, Iterable, List, Optional, Set, Union

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch import distributed as dist

from torch._utils import _get_available_device_type, _get_device_module
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor


def get_device_info():
    device_type = _get_available_device_type()
    if device_type is None:
        device_type = "cuda"  # default device_type: cuda
    device_module = _get_device_module(device_type)  # default device_module:torch.cuda
    return device_type, device_module


device_type, device_module = get_device_info()


def get_train_context(enable_loss_parallel: bool, enable_compiled_autograd: bool):
    @contextlib.contextmanager
    def context(cp_context: Optional[Generator[None, None, None]] = None):
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())

            if enable_compiled_autograd:
                stack.enter_context(
                    torch._dynamo.utils.maybe_enable_compiled_autograd(True)
                )

            if cp_context is not None:
                from torch.nn.attention import sdpa_kernel, SDPBackend

                # currently we only support these two SDP backends.
                stack.enter_context(
                    sdpa_kernel(
                        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
                    )
                )
                stack.enter_context(cp_context)

            yield

    return context


def init_distributed():

    # to mitigate the memory issue that collectives using
    # async_op=True hold memory longer than they should
    # such as those in tensor parallelism
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

    backend = "nccl"
    torch.distributed.init_process_group(
        backend=backend,
        timeout=timedelta(seconds=180),
    )
