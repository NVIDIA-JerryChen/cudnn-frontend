# SPDX-License-Identifier: BSD-3-Clause
"""CUDA architecture helpers used by host-side dispatch."""

from __future__ import annotations

import os
import re

import torch

from .fake_tensor import is_fake_mode

SUPPORTED_ARCHES = (90, 100, 103)


def get_device_arch(device_index: int | None = None) -> int:
    """Return the CUDA compute capability encoded as ``major * 10 + minor``."""

    target = os.environ.get("FLEX_ATTENTION_ARCH") or os.environ.get("CUTE_DSL_ARCH")
    if is_fake_mode() and target is not None:
        match = re.fullmatch(r"(?:sm_?)?(\d+)[af]?", target, re.IGNORECASE)
        if match is None:
            raise ValueError(f"invalid offline target architecture: {target!r}")
        arch = int(match.group(1))
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("cudnn.flex_attention requires a CUDA device or an explicit target in fake mode")
        major, minor = torch.cuda.get_device_capability(device_index)
        arch = major * 10 + minor
    if arch not in SUPPORTED_ARCHES:
        raise NotImplementedError(f"cudnn.flex_attention supports SM90, SM100, and SM103; got SM{arch}")
    return arch


__all__ = ["SUPPORTED_ARCHES", "get_device_arch"]
