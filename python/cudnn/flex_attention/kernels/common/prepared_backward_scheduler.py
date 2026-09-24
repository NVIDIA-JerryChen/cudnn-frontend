# SPDX-License-Identifier: BSD-3-Clause
"""Prepared backward tasks claimed only by resident CTA clusters."""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute  # noqa: PLR0402 - preserve the CuTe DSL namespace
from cudnn.flex_attention._compat.cute_dsl_utils import ParamsBase
from cudnn.flex_attention.kernels.common.communication import load_acquire_system
from cudnn.flex_attention.kernels.common.tile_scheduler import SingleTileScheduler
from cutlass import Int32


class PreparedBackwardScheduler(SingleTileScheduler):
    @dataclass
    class Params(ParamsBase):
        work: cute.Tensor
        state: cute.Tensor
        cluster_size: cutlass.Constexpr[int]

    @staticmethod
    def to_underlying_arguments(work: cute.Tensor, state: cute.Tensor, cluster_size: int) -> Params:
        return PreparedBackwardScheduler.Params(work, state, cluster_size)

    @staticmethod
    def get_grid_shape(params: Params, *, loc=None, ip=None):
        return (params.work.shape[0] * params.cluster_size, 1, 1)

    @staticmethod
    @cute.jit
    def create(params: Params, *, loc=None, ip=None):
        block, _, _ = cute.arch.block_idx()
        cluster = block // params.cluster_size
        cta = block % params.cluster_size
        # A 2CTA cluster is co-resident before either CTA starts. Only the
        # first thread claims work; every role/CTA acquires the
        # same assignment. Earlier tickets thus always belong to resident or
        # completed work, regardless of CUDA's physical block launch order.
        if cute.arch.thread_idx()[0] == 0 and cta == 0:
            ticket = cute.arch.atomic_add(params.state.iterator, Int32(1), sem="relaxed", scope="gpu")
            cute.arch.atomic_add(params.state.iterator + Int32(1) + cluster, ticket + Int32(1), sem="release", scope="gpu")
        task = Int32(-1)
        while task < 0:
            task = load_acquire_system(params.state.iterator + Int32(1) + cluster)
        n_cluster = params.work[task, 0]
        head = params.work[task, 1]
        batch = params.work[task, 2]
        return SingleTileScheduler(params, (n_cluster * params.cluster_size + cta, head, batch), loc=loc, ip=ip)
