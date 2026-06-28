"""Separate, KTC-free, GPU-resident Feature-Engineering step for ``MLFRAME_FE_GPU_STRICT``.

A dedicated STRICT FE path that honors a hard residency contract (operands + y uploaded ONCE per device, zero
bulk D2H mid-pipeline, bounded scalar D2H for the branchy selection, all compute on GPU kernels, no KTC
crossover, selection-equivalent), with multi-GPU sharding + per-device hw-spec-aware chunking. See the package
modules + the plan for the contract + phasing. Phase 0 is a scaffold (the entry stub falls back to the existing
per-family FE step until later phases implement the resident pipeline)."""
from ._audit import BULK_BYTES, ResidencyReport, residency_audit
from ._entry import (
    fe_gpu_strict_bytematch_enabled,
    fe_gpu_strict_resident_enabled,
    run_fe_step_gpu_strict,
)
from ._state import ResidentFEState

__all__ = [
    "ResidentFEState",
    "run_fe_step_gpu_strict",
    "fe_gpu_strict_resident_enabled",
    "fe_gpu_strict_bytematch_enabled",
    "residency_audit",
    "ResidencyReport",
    "BULK_BYTES",
]
