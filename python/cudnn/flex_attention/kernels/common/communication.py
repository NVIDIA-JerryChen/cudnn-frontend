# SPDX-License-Identifier: BSD-3-Clause
"""Device ordering for caller-owned attention communication buffers."""

import cutlass
import cutlass.cute as cute  # noqa: PLR0402 - preserve the CuTe DSL namespace
from cutlass import Int32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def load_acquire_system(ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    value = llvm.inline_asm(
        T.i32(),
        [ptr.toint(loc=loc, ip=ip).ir_value()],
        "ld.global.acquire.sys.b32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(value)


@cute.jit
def wait_kv_ready(
    ready: cute.Tensor,
    head: Int32,
    first_token: Int32,
    token_count: cutlass.Constexpr[int],
    valid_tokens: Int32,
    comm_block_size: cutlass.Constexpr[int],
) -> None:
    """Acquire every communication block touched by this load/CTA cluster."""
    first = first_token // comm_block_size
    end_token = cutlass.min(first_token + token_count, valid_tokens)
    end = cute.ceil_div(end_token, comm_block_size)
    for block in cutlass.range(first, end):
        ptr = ready.iterator + head * ready.stride[0] + block * ready.stride[1]
        value = Int32(0)
        while value == 0:
            value = load_acquire_system(ptr)
    cute.arch.sync_warp()


@dsl_user_op
def fence_proxy_async_global(*, loc=None, ip=None) -> Int32:
    """Order completed async global stores before generic signal publication."""
    value = llvm.inline_asm(
        T.i32(),
        [],
        "fence.proxy.async.global; mov.u32 $0, 0;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(value)


@cute.jit
def publish_dkv_done(
    done: cute.Tensor,
    expected: cute.Tensor | None,
    completed: cute.Tensor | None,
    completed_count: cute.Tensor | None,
    head: Int32,
    first_token: Int32,
    token_count: cutlass.Constexpr[int],
    valid_tokens: Int32,
    block_size: cutlass.Constexpr[int],
) -> None:
    """One publisher after all writers in a physical CTA have completed."""
    end = cute.ceil_div(cutlass.min(first_token + token_count, valid_tokens), block_size)
    for block in cutlass.range(first_token // block_size, end):
        previous = cute.arch.atomic_add(done.iterator + head * done.stride[0] + block, Int32(1), sem="acq_rel", scope="sys")
        if const_expr(completed is not None):  # noqa: SIM102 - keep optional ABI pruning outside runtime control flow
            if previous + Int32(1) == expected[head, block]:
                cute.arch.atomic_add(completed.iterator + head * completed.stride[0] + block, Int32(1), sem="release", scope="sys")
                cute.arch.atomic_add(completed_count.iterator, Int32(1), sem="release", scope="sys")
