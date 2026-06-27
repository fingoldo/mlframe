"""Entry point for the separate KTC-free GPU-resident FE step (``MLFRAME_FE_GPU_STRICT`` +
``MLFRAME_FE_GPU_STRICT_RESIDENT``).

Phase 0: this is a SCAFFOLD. ``run_fe_step_gpu_strict`` raises ``NotImplementedError`` so the caller's
try/except falls back to the existing per-family FE step -> zero behavior change until a later phase implements
the resident pipeline and the resident flag is turned on. The branch in ``_run_fe_step`` is gated behind the
default-OFF resident flag, so STRICT itself is unaffected too."""
from __future__ import annotations

import os


def fe_gpu_strict_resident_enabled() -> bool:
    """Whether the SEPARATE resident GPU-strict FE step is active. DEFAULT OFF (Phase 0). Requires both
    ``MLFRAME_FE_GPU_STRICT`` (the strict gate) AND ``MLFRAME_FE_GPU_STRICT_RESIDENT=1`` (this opt-in). Kept
    independent so the resident path can be rolled out / rolled back without touching the existing strict path."""
    if os.environ.get("MLFRAME_FE_GPU_STRICT_RESIDENT", "").strip().lower() not in ("1", "true", "on", "yes"):
        return False
    try:
        from .._fe_gpu_strict import fe_gpu_strict_enabled
        return bool(fe_gpu_strict_enabled())
    except Exception:
        return False


def run_fe_step_gpu_strict(self, **kwargs):
    """One FE step, fully GPU-resident, multi-GPU + hw-spec aware. Returns the SAME contract as
    ``_run_fe_step`` (``data, cols, nbins, X, selected_vars, n_recommended_features`` + mutated
    ``engineered_recipes``).

    PHASE 0 STUB: not yet implemented. Raises ``NotImplementedError`` so ``_run_fe_step`` falls back to the
    existing per-family path (no behavior change). Implemented incrementally in Phases 1-3."""
    raise NotImplementedError("run_fe_step_gpu_strict: resident GPU-strict FE step not yet implemented (Phase 0 scaffold)")
