# SPDX-License-Identifier: BSD-3-Clause
"""Host-side contracts for versioned arbitrary-attention plans.

The sparse topology can be shared across architectures, but mask payloads are
laid out for one concrete kernel consumer.  This module keeps that boundary
explicit without importing CuTe DSL objects into host-side plan metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from cudnn.flex_attention.plan.validation import PlanGeometry


def _tensor_version(tensor: torch.Tensor, *, name: str) -> int:
    """Read PyTorch's host-side mutation counter without synchronizing CUDA."""

    try:
        version = tensor._version
    except RuntimeError as exc:
        raise ValueError(f"{name} must track a tensor version so arbitrary plans can " "detect in-place prefix mutations") from exc
    if type(version) is not int:
        raise TypeError(f"{name} tensor version must be an int")
    return version


def canonical_blackwell_arch_family(arch: int) -> str:
    """Normalize supported Blackwell targets to the SM100 consumer family."""

    if type(arch) is not int:
        raise TypeError("arch must be an int")
    if arch in (100, 103):
        return "sm100"
    raise ValueError(f"arch {arch} is not an SM100/SM103 target")


@dataclass(frozen=True)
class ArbitraryPlanTopology:
    """Architecture-neutral logical tile topology for a plan consumer.

    ``tile_m`` is one consumer Q subtile. ``q_stage`` counts Q subtiles owned
    by one CTA and ``cta_group_size`` counts cooperative CTAs.
    ``cluster_axis`` records whether cooperation expands the logical Q/M axis
    (forward) or the logical KV/N axis (generic SM100 backward).  Keeping these
    factors separate prevents a planner from silently confusing a Q256 tile
    with a K256 cluster-union tile.
    """

    tile_m: int
    tile_n: int
    q_stage: int
    cta_group_size: int
    pack_gqa: bool
    qhead_per_kvhead: int
    cluster_axis: str = "m"

    def __post_init__(self) -> None:
        for name in ("tile_m", "tile_n", "q_stage", "cta_group_size"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an int")
        if self.tile_m <= 0 or self.tile_n <= 0:
            raise ValueError("tile_m and tile_n must be positive")
        if self.q_stage <= 0:
            raise ValueError("q_stage must be positive")
        if self.cta_group_size not in (1, 2):
            raise ValueError("cta_group_size must be 1 or 2")
        if type(self.cluster_axis) is not str or self.cluster_axis not in ("m", "n"):
            raise ValueError("cluster_axis must be 'm' or 'n'")
        if type(self.pack_gqa) is not bool:
            raise TypeError("pack_gqa must be a bool")
        if type(self.qhead_per_kvhead) is not int:
            raise TypeError("qhead_per_kvhead must be an int")
        if self.qhead_per_kvhead <= 0:
            raise ValueError("qhead_per_kvhead must be positive")

    @property
    def physical_subtiles(self) -> int:
        """Number of CTA-owned payload slices for one logical plan tile."""

        return self.q_stage * self.cta_group_size

    @property
    def block_size(self) -> tuple[int, int]:
        return (
            self.tile_m * self.q_stage * (self.cta_group_size if self.cluster_axis == "m" else 1),
            self.tile_n * (self.cta_group_size if self.cluster_axis == "n" else 1),
        )

    @property
    def compile_key(self) -> tuple:
        return (
            self.tile_m,
            self.tile_n,
            self.q_stage,
            self.cta_group_size,
            self.pack_gqa,
            self.qhead_per_kvhead,
            self.cluster_axis,
        )


@dataclass(frozen=True)
class ArbitraryPlanRuntimeBinding:
    """Runtime geometry and varlen-prefix provenance captured by a plan.

    Q/K/V tensor identity is deliberately absent: a plan is reusable with new
    values as long as their geometry is unchanged.  Varlen prefix identity is
    part of the contract because equal aggregate sizes do not imply an equal
    sample partition.  Tensor references are excluded from dataclass equality
    to avoid elementwise ``Tensor.__eq__``.
    """

    is_varlen: bool
    batch_size: int
    seqlen_q: int | None
    seqlen_k: int | None
    total_q: int
    total_k: int
    max_seqlen_q: int
    max_seqlen_k: int
    cu_seqlens_q: torch.Tensor | None = field(default=None, compare=False, repr=False)
    cu_seqlens_k: torch.Tensor | None = field(default=None, compare=False, repr=False)
    cu_seqlens_q_version: int | None = None
    cu_seqlens_k_version: int | None = None

    def __post_init__(self) -> None:
        if type(self.is_varlen) is not bool:
            raise TypeError("is_varlen must be a bool")
        for name in (
            "batch_size",
            "total_q",
            "total_k",
            "max_seqlen_q",
            "max_seqlen_k",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.is_varlen:
            if self.seqlen_q is not None or self.seqlen_k is not None:
                raise ValueError("varlen runtime bindings must not set fixed seqlens")
            for name in ("cu_seqlens_q", "cu_seqlens_k"):
                if getattr(self, name) is None:
                    raise ValueError(f"varlen runtime binding requires {name}")
            for name in (
                "cu_seqlens_q_version",
                "cu_seqlens_k_version",
            ):
                version = getattr(self, name)
                if type(version) is not int or version < 0:
                    raise ValueError(f"{name} must be a non-negative int")
        else:
            for name in ("seqlen_q", "seqlen_k"):
                value = getattr(self, name)
                if type(value) is not int:
                    raise TypeError(f"fixed runtime binding {name} must be an int")
                if value < 0:
                    raise ValueError(f"fixed runtime binding {name} must be non-negative")
            if self.total_q != self.batch_size * self.seqlen_q:
                raise ValueError("fixed runtime binding total_q must equal B * Sq")
            if self.total_k != self.batch_size * self.seqlen_k:
                raise ValueError("fixed runtime binding total_k must equal B * Sk")
            if self.max_seqlen_q != self.seqlen_q:
                raise ValueError("fixed runtime binding max_seqlen_q must equal Sq")
            if self.max_seqlen_k != self.seqlen_k:
                raise ValueError("fixed runtime binding max_seqlen_k must equal Sk")
            if any(
                value is not None
                for value in (
                    self.cu_seqlens_q,
                    self.cu_seqlens_k,
                    self.cu_seqlens_q_version,
                    self.cu_seqlens_k_version,
                )
            ):
                raise ValueError("fixed runtime bindings must not carry cu_seqlens provenance")

    @classmethod
    def capture(
        cls,
        *,
        is_varlen: bool,
        batch_size: int,
        seqlen_q: int | None,
        seqlen_k: int | None,
        total_q: int,
        total_k: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        cu_seqlens_q: torch.Tensor | None,
        cu_seqlens_k: torch.Tensor | None,
    ) -> ArbitraryPlanRuntimeBinding:
        """Capture a runtime binding using only host-visible tensor metadata."""

        return cls(
            is_varlen=is_varlen,
            batch_size=batch_size,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            total_q=total_q,
            total_k=total_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q_version=(_tensor_version(cu_seqlens_q, name="cu_seqlens_q") if cu_seqlens_q is not None else None),
            cu_seqlens_k_version=(_tensor_version(cu_seqlens_k, name="cu_seqlens_k") if cu_seqlens_k is not None else None),
        )


def validate_arbitrary_plan_runtime_binding(
    binding: object,
    *,
    is_varlen: bool,
    batch_size: int,
    seqlen_q: int | None,
    seqlen_k: int | None,
    total_q: int,
    total_k: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    context: str,
) -> ArbitraryPlanRuntimeBinding:
    """Reject stale plan reuse using only shape, identity, and version metadata."""

    if not isinstance(binding, ArbitraryPlanRuntimeBinding):
        raise TypeError(f"{context} requires an ArbitraryPlanRuntimeBinding; got {type(binding).__name__}")
    if type(is_varlen) is not bool:
        raise TypeError("is_varlen must be a bool")
    if binding.is_varlen != is_varlen:
        raise ValueError(
            f"{context} mode mismatch: plan is " f"{'varlen' if binding.is_varlen else 'fixed'}, runtime is " f"{'varlen' if is_varlen else 'fixed'}"
        )

    if is_varlen:
        for name, runtime_tensor, bound_tensor, bound_version in (
            (
                "cu_seqlens_q",
                cu_seqlens_q,
                binding.cu_seqlens_q,
                binding.cu_seqlens_q_version,
            ),
            (
                "cu_seqlens_k",
                cu_seqlens_k,
                binding.cu_seqlens_k,
                binding.cu_seqlens_k_version,
            ),
        ):
            if runtime_tensor is not bound_tensor:
                raise ValueError(f"{context} {name} provenance mismatch: reuse the exact prefix " "tensor object used to build the plan, or rebuild the plan")
            runtime_version = _tensor_version(runtime_tensor, name=name)
            if runtime_version != bound_version:
                raise ValueError(
                    f"{context} {name} was modified in-place after plan "
                    f"construction (expected tensor version {bound_version}, got "
                    f"{runtime_version}); rebuild the plan"
                )

    runtime_geometry = {
        "batch_size": batch_size,
        "seqlen_q": seqlen_q,
        "seqlen_k": seqlen_k,
        "total_q": total_q,
        "total_k": total_k,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
    }
    geometry_names = (
        "batch_size",
        "total_q",
        "total_k",
        "max_seqlen_q",
        "max_seqlen_k",
    ) + (() if is_varlen else ("seqlen_q", "seqlen_k"))
    mismatches = [(name, getattr(binding, name), runtime_geometry[name]) for name in geometry_names if getattr(binding, name) != runtime_geometry[name]]
    if mismatches:
        details = ", ".join(f"{name}: expected {expected!r}, got {actual!r}" for name, expected, actual in mismatches)
        raise ValueError(f"{context} runtime geometry mismatch ({details})")
    return binding


@dataclass(frozen=True)
class ArbitraryTopologyTensors:
    """Consumer-independent sparse topology, separate from mask payloads.

    Indices are sample-local.  Prefix tensors map compact varlen rows back to
    each sample; payload tensors such as ``mask_block_masks`` deliberately do
    not appear here because their layout is architecture/consumer specific.
    """

    direction: str
    partial_count: torch.Tensor = field(compare=False)
    partial_offset: torch.Tensor = field(compare=False)
    partial_index: torch.Tensor = field(compare=False)
    full_count: torch.Tensor | None = field(compare=False)
    full_offset: torch.Tensor | None = field(compare=False)
    full_index: torch.Tensor | None = field(compare=False)
    cu_total_q_plan_rows: torch.Tensor | None = field(compare=False)
    cu_total_k_plan_rows: torch.Tensor | None = field(compare=False)
    runtime_binding: ArbitraryPlanRuntimeBinding = field(compare=False)
    dq_write_order: torch.Tensor | None = field(default=None, compare=False)
    dq_write_order_full: torch.Tensor | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.direction not in ("q2k", "k2q"):
            raise ValueError("topology direction must be 'q2k' or 'k2q'")
        if any(tensor is None for tensor in (self.partial_count, self.partial_offset, self.partial_index)):
            raise ValueError("partial count/offset/index topology tensors are required")
        full_present = tuple(tensor is not None for tensor in (self.full_count, self.full_offset, self.full_index))
        if any(full_present) and not all(full_present):
            raise ValueError("full count/offset/index must be provided together")
        if not isinstance(self.runtime_binding, ArbitraryPlanRuntimeBinding):
            raise TypeError("topology runtime_binding must be an ArbitraryPlanRuntimeBinding")


@dataclass(frozen=True)
class ArbitraryPlanSignature:
    """Static consumer contract carried by an arbitrary plan.

    Runtime values such as ``Hmask``, ``nfunc`` and sparse nnz intentionally do
    not belong here: changing them must not cause a kernel recompile.  The
    fields below describe only the architecture, traversal, and payload layout
    selected by the planner.
    """

    arch_family: str
    direction: str
    kernel_family: str
    tile_m: int
    tile_n: int
    q_stage: int
    cta_group_size: int
    pack_gqa: bool
    qhead_per_kvhead: int
    mma_atom_layout_id: str
    swap_ab: bool
    payload_layout_id: str
    dq_order_format: str
    cluster_axis: str = "m"
    scheduler_layout_id: str = "none"

    def __post_init__(self) -> None:
        # Reuse the strict topology validation for every duplicated field.
        _ = self.topology
        for name in (
            "arch_family",
            "direction",
            "kernel_family",
            "mma_atom_layout_id",
            "payload_layout_id",
            "dq_order_format",
            "scheduler_layout_id",
        ):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if type(self.swap_ab) is not bool:
            raise TypeError("swap_ab must be a bool")

    @property
    def compile_key(self) -> tuple:
        """Return the layout/config specialization portion of a JIT key."""

        return tuple(getattr(self, field.name) for field in fields(self))

    @property
    def topology(self) -> ArbitraryPlanTopology:
        return ArbitraryPlanTopology(
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            q_stage=self.q_stage,
            cta_group_size=self.cta_group_size,
            pack_gqa=self.pack_gqa,
            qhead_per_kvhead=self.qhead_per_kvhead,
            cluster_axis=self.cluster_axis,
        )


def resolve_arbitrary_pack_gqa(
    *,
    requested_pack_gqa: bool | None,
    num_q_heads: int,
    num_kv_heads: int,
    hmask: int,
    tile_m: int,
) -> tuple[bool, int]:
    """Resolve the architecture-neutral PackGQA topology contract."""

    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if hmask not in (1, num_q_heads):
        raise ValueError(f"Hmask must be 1 or Hq ({num_q_heads}); got {hmask}")
    qhead_per_kvhead = num_q_heads // num_kv_heads
    qratio_is_power_of_two = (qhead_per_kvhead & (qhead_per_kvhead - 1)) == 0
    can_pack = hmask == 1 and qratio_is_power_of_two and tile_m % qhead_per_kvhead == 0
    if requested_pack_gqa is True:
        if hmask != 1:
            raise ValueError("pack_gqa=True requires Hmask=1 for arbitrary attention")
        if not qratio_is_power_of_two:
            raise ValueError(f"pack_gqa=True requires qratio=Hq/Hkv to be a power of two; got {qhead_per_kvhead}")
        if tile_m % qhead_per_kvhead != 0:
            raise ValueError(f"pack_gqa=True requires tile_m ({tile_m}) to be divisible by " f"qratio ({qhead_per_kvhead})")
        return True, qhead_per_kvhead
    if requested_pack_gqa is False:
        return False, qhead_per_kvhead
    return num_q_heads > num_kv_heads and can_pack, qhead_per_kvhead


def validate_arbitrary_plan_signature(
    signature: object,
    expected: ArbitraryPlanSignature,
    *,
    context: str,
) -> ArbitraryPlanSignature:
    """Reject a plan that targets a different resolved consumer."""

    if not isinstance(signature, ArbitraryPlanSignature):
        raise TypeError(f"{context} requires an ArbitraryPlanSignature; got {type(signature).__name__}")
    mismatches = [
        (field.name, getattr(expected, field.name), getattr(signature, field.name))
        for field in fields(ArbitraryPlanSignature)
        if getattr(signature, field.name) != getattr(expected, field.name)
    ]
    if mismatches:
        details = ", ".join(f"{name}: expected {wanted!r}, got {actual!r}" for name, wanted, actual in mismatches)
        raise ValueError(f"{context} signature mismatch ({details})")
    return signature


def validate_arbitrary_topology_binding(
    plan: object,
    signature: ArbitraryPlanSignature,
    topology: ArbitraryTopologyTensors,
) -> None:
    """Ensure the versioned topology is the plan's single source of truth."""

    runtime_binding = getattr(topology, "runtime_binding", None)
    if not isinstance(runtime_binding, ArbitraryPlanRuntimeBinding):
        raise TypeError("arbitrary topology requires an ArbitraryPlanRuntimeBinding; rebuild the plan")

    # Dedicated hd256 dQ traverses the same Q-major coarse topology as
    # forward, but it is still a backward execution contract.  Kernel family
    # therefore disambiguates it from the K-major dKdV/generic-bwd plans.
    expected_direction = (
        "q2k"
        if signature.kernel_family == "sm100_hd256_dq"
        else {
            "forward": "q2k",
            "backward": "k2q",
        }.get(signature.direction)
    )
    if expected_direction is None or topology.direction != expected_direction:
        raise ValueError("arbitrary topology direction does not match plan signature")
    q_prefix = topology.cu_total_q_plan_rows
    k_prefix = topology.cu_total_k_plan_rows
    if runtime_binding.is_varlen:
        if q_prefix is None or k_prefix is None:
            raise ValueError("arbitrary varlen topology requires both Q-row and K-row prefixes")
    elif q_prefix is not None or k_prefix is not None:
        raise ValueError("arbitrary fixed topology must not carry row prefixes")
    bindings = (
        ("mask_block_cnt", topology.partial_count),
        ("mask_block_offset", topology.partial_offset),
        ("mask_block_idx", topology.partial_index),
        ("full_block_cnt", topology.full_count),
        ("full_block_offset", topology.full_offset),
        ("full_block_idx", topology.full_index),
        ("dq_write_order", topology.dq_write_order),
        ("dq_write_order_full", topology.dq_write_order_full),
    )
    for name, expected in bindings:
        if getattr(plan, name, None) is not expected:
            raise ValueError(f"arbitrary topology binding mismatch for {name}")
    outer_row_prefix = topology.cu_total_q_plan_rows if topology.direction == "q2k" else topology.cu_total_k_plan_rows
    if getattr(plan, "cu_total_m_blocks", None) is not outer_row_prefix:
        raise ValueError("arbitrary topology binding mismatch for cu_total_m_blocks")


def validate_arbitrary_attention_plan(
    *,
    block_sparse_tensors: object | None,
) -> ArbitraryPlanSignature:
    """Validate that an attention request carries a complete versioned plan.

    A payload without a versioned signature is deliberately rejected.  Shape
    checks alone cannot distinguish an SM90 WGMMA payload from an SM100
    tcgen05/TMEM payload that happens to have the same extent.
    """

    signature = getattr(block_sparse_tensors, "plan_signature", None)
    payload = getattr(block_sparse_tensors, "mask_block_masks", None)
    topology_tensors = getattr(block_sparse_tensors, "topology_tensors", None)
    has_signature = signature is not None
    has_payload = payload is not None
    has_topology = topology_tensors is not None

    plan_parts = {
        "plan_signature": has_signature,
        "mask_block_masks": has_payload,
        "topology_tensors": has_topology,
    }
    if not all(plan_parts.values()):
        missing = ", ".join(name for name, present in plan_parts.items() if not present)
        raise ValueError("arbitrary attention requires a plan returned by " f"create_mask_plan; missing {missing}")
    if not isinstance(signature, ArbitraryPlanSignature):
        raise TypeError("arbitrary plan_signature must be an ArbitraryPlanSignature")
    if not isinstance(topology_tensors, ArbitraryTopologyTensors):
        raise TypeError("arbitrary plan requires ArbitraryTopologyTensors")
    validate_arbitrary_topology_binding(
        block_sparse_tensors,
        signature,
        topology_tensors,
    )
    return signature


@dataclass(frozen=True)
class MaskPlanMetadata:
    """Immutable public description of a compiled mask plan."""

    mode: str
    arch: int
    device: torch.device
    dtype: torch.dtype
    batch_size: int
    total_q: int
    total_k: int
    max_seqlen_q: int
    max_seqlen_k: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    head_dim_v: int
    hmask: int
    nfunc: int
    pack_gqa: bool
    has_backward: bool


class MaskPlan:
    """Opaque owner of forward/backward topology and packed predicate payloads."""

    __slots__ = (
        "_packed_plan",
        "_cu_seqlens_q",
        "_cu_seqlens_k",
        "_metadata",
    )

    def __init__(
        self,
        *,
        packed_plan: object,
        geometry: PlanGeometry,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        signature = validate_arbitrary_attention_plan(
            block_sparse_tensors=packed_plan,
        )
        self._packed_plan = packed_plan
        self._cu_seqlens_q = geometry.cu_seqlens_q
        self._cu_seqlens_k = geometry.cu_seqlens_k
        self._metadata = MaskPlanMetadata(
            mode="varlen" if geometry.is_varlen else "fixed",
            arch=geometry.arch,
            device=device,
            dtype=dtype,
            batch_size=geometry.batch_size,
            total_q=geometry.total_q,
            total_k=geometry.total_k,
            max_seqlen_q=geometry.max_seqlen_q,
            max_seqlen_k=geometry.max_seqlen_k,
            num_q_heads=geometry.num_q_heads,
            num_kv_heads=geometry.num_kv_heads,
            head_dim=geometry.head_dim,
            head_dim_v=geometry.head_dim_v,
            hmask=geometry.hmask,
            nfunc=geometry.nfunc,
            pack_gqa=bool(getattr(packed_plan, "pack_gqa")),
            has_backward=getattr(packed_plan, "bwd_tensors", None) is not None,
        )
        expected_arch_family = "sm90" if geometry.arch == 90 else "sm100"
        if signature.arch_family != expected_arch_family:
            raise ValueError("planner and runtime architecture do not match")

    @property
    def metadata(self) -> MaskPlanMetadata:
        return self._metadata

    def with_kv_ready(self, ready: torch.Tensor, *, block_size: int) -> MaskPlan:
        """Bind system-acquire KV gates without changing interval topology.

        The caller publishes a nonzero signal only after both K and V for that
        communication block are visible, and resets signals before reuse. This
        view and the caller own the signal storage; compiled caches do not.
        """
        import copy

        metadata = self.metadata
        if metadata.mode != "fixed" or metadata.batch_size != 1:
            raise NotImplementedError("KV readiness currently requires fixed-length batch size 1")
        if metadata.arch not in (90, 100, 103) or (metadata.arch != 90 and metadata.head_dim == 256):
            raise NotImplementedError("KV readiness currently requires SM90 or generic SM100/SM103 kernels")
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("communication block_size must be a positive integer")
        blocks = (metadata.total_k + block_size - 1) // block_size
        if ready.device != metadata.device or ready.dtype != torch.int32:
            raise ValueError("KV ready signals must be int32 on the plan device")
        if ready.ndim != 2 or ready.shape != (metadata.num_kv_heads, blocks) or ready.stride(-1) != 1:
            raise ValueError(f"KV ready signals require [Hkv, communication blocks] = {(metadata.num_kv_heads, blocks)} with contiguous blocks")
        updates = {"kv_ready": ready, "comm_block_size": block_size}
        packed = self._packed_plan
        if packed.bwd_tensors is not None:
            updates["bwd_tensors"] = packed.bwd_tensors._replace(kv_ready=ready, comm_block_size=block_size)
        bound = copy.copy(self)
        bound._packed_plan = packed._replace(**updates)
        return bound

    def dkv_producer_counts(self, *, block_size: int) -> torch.Tensor:
        """Prepare the number of active physical CTA writers per Hkv/K block.

        K2Q activity belongs to a native cluster, so both physical halves count
        even when a head is active in only one half. Ghost tail CTAs do not count.
        This is a preparation operation and allocates its returned int32 tensor.
        """
        metadata = self.metadata
        packed = self._packed_plan.bwd_tensors
        if packed is None or metadata.mode != "fixed" or metadata.batch_size != 1:
            raise NotImplementedError("dKV publication requires a backward plan for fixed-length batch size 1")
        if metadata.arch not in (90, 100, 103) or (metadata.arch != 90 and metadata.head_dim == 256):
            raise NotImplementedError("dKV publication currently requires SM90 or generic SM100/SM103 kernels")
        tile = packed.block_size[1] if metadata.arch == 90 else packed.plan_signature.tile_n
        if type(block_size) is not int or block_size <= 0 or (tile % block_size and block_size % tile):
            raise ValueError(f"dKV block_size must divide or be a multiple of the physical KV tile ({tile})")
        active = (packed.mask_block_cnt + packed.full_block_cnt) > 0
        if metadata.arch == 90:
            # SM90 executes a zero epilogue for inactive K2Q tasks, including
            # ordered GQA reductions. Those writes must also precede publication.
            active = torch.ones_like(active)
        heads = metadata.num_q_heads
        if active.shape[0] == 1:
            active = active.expand(heads, -1)
        native_tiles = (metadata.total_k + tile - 1) // tile
        cluster_size = 1 if metadata.arch == 90 else packed.plan_signature.cta_group_size
        physical = active.repeat_interleave(cluster_size, dim=-1)[:, :native_tiles]
        counts = physical.reshape(metadata.num_kv_heads, heads // metadata.num_kv_heads, native_tiles).sum(1, dtype=torch.int32)
        if block_size <= tile:
            counts = counts.repeat_interleave(tile // block_size, dim=-1)
        else:
            group = block_size // tile
            padding = (-native_tiles) % group
            counts = torch.nn.functional.pad(counts, (0, padding)).reshape(metadata.num_kv_heads, -1, group).sum(-1, dtype=torch.int32)
        return counts[:, : (metadata.total_k + block_size - 1) // block_size].contiguous()

    def with_dkv_done(
        self,
        done: torch.Tensor,
        *,
        block_size: int,
        expected: torch.Tensor | None = None,
        completed: torch.Tensor | None = None,
        completed_count: torch.Tensor | None = None,
    ) -> MaskPlan:
        """Bind backward system-release publication after physical tile writes.

        GQA publishes raw FP32 accumulator completion, before conversion. The
        caller must reset counters before execution. Optional completion flags
        are incremented by the last writer, using ``dkv_producer_counts`` as
        ``expected``. Zero-producer flags must be initialized by the caller.
        """
        import copy

        metadata = self.metadata
        packed = self._packed_plan.bwd_tensors
        if packed is None or metadata.mode != "fixed" or metadata.batch_size != 1:
            raise NotImplementedError("dKV publication requires a backward plan for fixed-length batch size 1")
        if metadata.arch not in (90, 100, 103) or (metadata.arch != 90 and metadata.head_dim == 256):
            raise NotImplementedError("dKV publication currently requires SM90 or generic SM100/SM103 kernels")
        tile = packed.block_size[1] if metadata.arch == 90 else packed.plan_signature.tile_n
        if type(block_size) is not int or block_size <= 0 or (tile % block_size and block_size % tile):
            raise ValueError(f"dKV block_size must divide or be a multiple of the physical KV tile ({tile})")
        shape = (metadata.num_kv_heads, (metadata.total_k + block_size - 1) // block_size)
        if (expected is None, completed is None, completed_count is None).count(True) not in (0, 3):
            raise ValueError("expected, completed and completed_count must be provided together")
        for name, tensor in (("done", done), ("expected", expected), ("completed", completed), ("completed_count", completed_count)):
            if tensor is None:
                continue
            expected_shape = (1,) if name == "completed_count" else shape
            if tensor.device != metadata.device or tensor.dtype != torch.int32 or tuple(tensor.shape) != expected_shape or tensor.stride(-1) != 1:
                raise ValueError(f"{name} requires int32 {expected_shape} on the plan device with contiguous blocks")
        bound = copy.copy(self)
        bound._packed_plan = self._packed_plan._replace(
            bwd_tensors=packed._replace(
                dkv_done=done,
                dkv_expected=expected,
                dkv_completed=completed,
                dkv_completed_count=completed_count,
                dkv_block_size=block_size,
            )
        )
        return bound

    @property
    def backward_block_size(self) -> tuple[int, int] | None:
        """Logical Q/KV token span, including cooperative CTA grouping."""
        packed = self._packed_plan.bwd_tensors
        return None if packed is None else packed.block_size

    def dkv_accumulator_permutations(self, *, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return natural-to-native scalar indices for one external dK/dV block.

        The caller owns these int32 device tensors. This preparation-only
        operation derives the layout from the native postprocess, synchronizes,
        and checks that each mapping is bijective. It is not capture-safe.
        """
        from cudnn.flex_attention.dispatch import _get_bwd_postprocess_kernel, torch2cute_dtype_map
        from cudnn.flex_attention.kernels.sm90.bwd.backward_config import resolve_sm90_bwd_consumer_config
        from cudnn.flex_attention.kernels.sm100.bwd.backward_config import resolve_sm100_bwd_consumer_config

        metadata = self.metadata
        if self._packed_plan.bwd_tensors is None:
            raise ValueError("accumulator layout requires a plan built with backward metadata")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("accumulator layout must be prepared before CUDA Graph capture")
        if metadata.mode != "fixed" or metadata.arch not in (90, 100, 103):
            raise NotImplementedError("external accumulator layouts require fixed-length SM90/SM100/SM103 plans")
        resolver = resolve_sm90_bwd_consumer_config if metadata.arch == 90 else resolve_sm100_bwd_consumer_config
        config = resolver(
            arch=metadata.arch,
            dtype=metadata.dtype,
            head_dim=metadata.head_dim,
            head_dim_v=metadata.head_dim_v,
            num_q_heads=metadata.num_q_heads,
            num_kv_heads=metadata.num_kv_heads,
            is_varlen=False,
        )
        if type(block_size) is not int or block_size <= 0 or block_size % config.tile_n:
            raise ValueError(f"accumulator block_size must be a positive multiple of the native KV tile ({config.tile_n})")
        hopper = metadata.arch == 90
        atom = config.atom_layout_n_dkv if hopper else 1
        swap = config.dkv_swap_ab if hopper else False
        cluster = 1 if hopper else config.cta_group_size
        outputs = {}
        for dimension in (metadata.head_dim, metadata.head_dim_v):
            if dimension in outputs:
                continue
            padded = (dimension + 15) // 16 * 16
            # Export padded rows as well: callers can validate their own compact
            # stride or select only the live columns, without guessing an MMA layout.
            total = block_size * padded
            logical = torch.arange(total, device=metadata.device, dtype=torch.int64)
            permutation = torch.zeros_like(logical)
            output = torch.empty((1, block_size, 1, padded), device=metadata.device, dtype=metadata.dtype)
            post = _get_bwd_postprocess_kernel(
                torch2cute_dtype_map[metadata.dtype],
                padded,
                config.tile_n,
                256 if hopper else 128,
                atom,
                swap,
                hopper and atom == 1,
                False,
                False,
                cluster,
                metadata.arch,
            )
            # Bit planes remain exact through FP16/BF16; a scalar index ramp would
            # lose low bits when converted by the native postprocess.
            for bit in range(max(1, (total - 1).bit_length())):
                accum = ((logical >> bit) & 1).float().reshape(1, 1, total)
                post(accum, output, 1.0, None, block_size)
                permutation |= output.reshape(-1).to(torch.int64) << bit
            if not torch.equal(permutation.sort().values, logical):
                raise RuntimeError("native dKV accumulator mapping is not bijective")
            outputs[dimension] = permutation.to(torch.int32).contiguous()
        return outputs[metadata.head_dim], outputs[metadata.head_dim_v]

    def with_backward_work_order(self, work: torch.Tensor, state: torch.Tensor) -> MaskPlan:
        """Prepare native cluster/head work and matching deterministic dQ tickets.

        ``work`` is a complete permutation of [KV cluster, Q head, batch] tasks.
        Q heads sharing a KV head must appear in ascending order for each cluster.
        ``state`` is caller-owned int32 [1 + tasks]: reset the cursor to zero and
        the assignment slots to -1 before every execute. A resident cluster claims
        a task from this cursor; physical CUDA block order carries no semantics.
        This cold operation copies compact topology to the host, never a QxK mask.
        """
        import copy

        metadata = self.metadata
        packed = self._packed_plan.bwd_tensors
        if packed is None or metadata.mode != "fixed" or metadata.batch_size != 1:
            raise NotImplementedError("prepared backward work requires fixed batch size 1 and backward metadata")
        if metadata.arch not in (90, 100, 103) or (metadata.arch != 90 and metadata.head_dim == 256):
            raise NotImplementedError("prepared backward work currently requires SM90 or generic SM100/SM103 kernels")
        clusters = packed.mask_block_cnt.shape[1]
        heads = metadata.num_q_heads
        tasks = clusters * heads
        for name, tensor, shape in (("work", work, (tasks, 3)), ("state", state, (tasks + 1,))):
            if tensor.device != metadata.device or tensor.dtype != torch.int32 or tuple(tensor.shape) != shape or not tensor.is_contiguous():
                raise ValueError(f"{name} requires contiguous int32 {shape} on the plan device")
        from cudnn.flex_attention.runtime.fake_tensor import is_fake_mode

        if is_fake_mode():
            # Compilation consumes descriptors; tickets are materialized from
            # the real sparse topology when the runtime plan is prepared.
            bound = copy.copy(self)
            order_heads = heads if packed.mask_block_cnt.shape[0] == 1 else 1
            bound._packed_plan = self._packed_plan._replace(
                bwd_tensors=packed._replace(
                    bwd_work_desc=work.clone(),
                    bwd_work_state=state,
                    bwd_dq_order=torch.empty((order_heads, packed.mask_block_idx.numel()), dtype=torch.int32, device=metadata.device),
                    bwd_dq_order_full=torch.empty((order_heads, packed.full_block_idx.numel()), dtype=torch.int32, device=metadata.device),
                )
            )
            return bound
        descriptors = [tuple(row) for row in work.cpu().tolist()]
        wanted = {(cluster, head, 0) for cluster in range(clusters) for head in range(heads)}
        if set(descriptors) != wanted:
            raise ValueError("work must contain every native cluster/head/batch exactly once")
        group = heads // metadata.num_kv_heads
        next_head = {}
        for cluster, head, _ in descriptors:
            key = cluster, head // group
            expected_head = next_head.get(key, head // group * group)
            if head != expected_head:
                raise ValueError("work must preserve ascending Q-head order within each cluster/KV-head group")
            next_head[key] = expected_head + 1

        mask_heads = packed.mask_block_cnt.shape[0]
        order_heads = heads if mask_heads == 1 else 1
        partial_order = torch.zeros((order_heads, packed.mask_block_idx.numel()), dtype=torch.int32)
        full_order = torch.zeros((order_heads, packed.full_block_idx.numel()), dtype=torch.int32)
        partial_offsets, full_offsets = packed.mask_block_offset.cpu().tolist(), packed.full_block_offset.cpu().tolist()
        partial_counts, full_counts = packed.mask_block_cnt.cpu().tolist(), packed.full_block_cnt.cpu().tolist()
        partial_indices, full_indices = packed.mask_block_idx.cpu().tolist(), packed.full_block_idx.cpu().tolist()
        tickets = [{} for _ in range(heads)]
        for cluster, head, _ in descriptors:
            mask_head = 0 if mask_heads == 1 else head
            order_head = head if mask_heads == 1 else 0
            row = mask_head * clusters + cluster
            for counts, offsets, indices, output in (
                (partial_counts, partial_offsets, partial_indices, partial_order.numpy()),
                (full_counts, full_offsets, full_indices, full_order.numpy()),
            ):
                start = offsets[row]
                for entry in range(start, start + counts[mask_head][cluster]):
                    q_block = indices[entry]
                    ticket = tickets[head].get(q_block, 0)
                    output[order_head, entry] = ticket
                    tickets[head][q_block] = ticket + 1
        bound = copy.copy(self)
        bound._packed_plan = self._packed_plan._replace(
            bwd_tensors=packed._replace(
                bwd_work_desc=work.clone(),
                bwd_work_state=state,
                bwd_dq_order=partial_order.to(metadata.device),
                bwd_dq_order_full=full_order.to(metadata.device),
            )
        )
        return bound

    def debug_snapshot(self) -> dict[str, object]:
        """Return cloned tensors for diagnostics without exposing mutable plan state."""

        names = (
            "mask_block_cnt",
            "mask_block_offset",
            "mask_block_idx",
            "mask_block_masks",
            "full_block_cnt",
            "full_block_offset",
            "full_block_idx",
            "cu_total_m_blocks",
            "dq_write_order",
            "dq_write_order_full",
            "sequence_desc",
            "fwd_work_desc",
        )

        def snapshot_view(view: object | None) -> dict[str, torch.Tensor | None] | None:
            if view is None:
                return None
            return {name: (value.detach().clone() if isinstance((value := getattr(view, name, None)), torch.Tensor) else None) for name in names}

        return {
            "metadata": self._metadata,
            "forward": snapshot_view(self._packed_plan),
            "backward": snapshot_view(getattr(self._packed_plan, "bwd_tensors", None)),
            "dq": snapshot_view(getattr(self._packed_plan, "dq_tensors", None)),
            "cu_seqlens_q": (self._cu_seqlens_q.detach().clone() if self._cu_seqlens_q is not None else None),
            "cu_seqlens_k": (self._cu_seqlens_k.detach().clone() if self._cu_seqlens_k is not None else None),
        }

    @property
    def _is_varlen(self) -> bool:
        q_prefix_present = self._cu_seqlens_q is not None
        k_prefix_present = self._cu_seqlens_k is not None
        if q_prefix_present != k_prefix_present:
            raise RuntimeError("MaskPlan must contain both cu_seqlens_q and cu_seqlens_k, or neither")
        return q_prefix_present

    def _validate_runtime(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        metadata = self._metadata
        mode = "varlen" if self._is_varlen else "fixed"
        if mode != metadata.mode:
            raise RuntimeError(f"MaskPlan mode is {metadata.mode}, but its cumulative sequence lengths select {mode}")
        expected_rank = 3 if mode == "varlen" else 4
        if any(tensor.ndim != expected_rank for tensor in (q, k, v)):
            raise ValueError(f"{mode} q, k, and v must be rank-{expected_rank}")
        if any(tensor.device != metadata.device for tensor in (q, k, v)):
            raise ValueError("q, k, and v must use the MaskPlan CUDA device")
        if any(tensor.dtype != metadata.dtype for tensor in (q, k, v)):
            raise TypeError("q, k, and v must use the MaskPlan dtype")
        if q.shape[-2:] != (metadata.num_q_heads, metadata.head_dim):
            raise ValueError("q head geometry does not match MaskPlan")
        if k.shape[-2:] != (metadata.num_kv_heads, metadata.head_dim):
            raise ValueError("k head geometry does not match MaskPlan")
        if v.shape[-2:] != (metadata.num_kv_heads, metadata.head_dim_v):
            raise ValueError("v head geometry does not match MaskPlan")
        if mode == "fixed":
            expected_q = (metadata.batch_size, metadata.total_q // metadata.batch_size)
            expected_k = (metadata.batch_size, metadata.total_k // metadata.batch_size)
            if q.shape[:2] != expected_q or k.shape[:2] != expected_k or v.shape[:2] != expected_k:
                raise ValueError("fixed sequence geometry does not match MaskPlan")
        elif q.shape[0] != metadata.total_q or k.shape[0] != metadata.total_k or v.shape[0] != metadata.total_k:
            raise ValueError("varlen total sequence geometry does not match MaskPlan")
        needs_backward = torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (q, k, v))
        if needs_backward and not metadata.has_backward:
            raise ValueError("MaskPlan was built without backward payloads")

    @property
    def _runtime_args(self) -> tuple[object, torch.Tensor | None, torch.Tensor | None]:
        return self._packed_plan, self._cu_seqlens_q, self._cu_seqlens_k


__all__ = [
    "ArbitraryPlanRuntimeBinding",
    "ArbitraryPlanSignature",
    "ArbitraryPlanTopology",
    "ArbitraryTopologyTensors",
    "MaskPlan",
    "MaskPlanMetadata",
    "canonical_blackwell_arch_family",
    "resolve_arbitrary_pack_gqa",
    "validate_arbitrary_attention_plan",
    "validate_arbitrary_plan_runtime_binding",
    "validate_arbitrary_plan_signature",
    "validate_arbitrary_topology_binding",
]
