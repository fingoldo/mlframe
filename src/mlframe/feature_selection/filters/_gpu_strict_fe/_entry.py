"""Entry point for the separate KTC-free GPU-resident FE step (``MLFRAME_FE_GPU_STRICT`` +
``MLFRAME_FE_GPU_STRICT_RESIDENT``).

Phase 0: this is a SCAFFOLD. ``run_fe_step_gpu_strict`` raises ``NotImplementedError`` so the caller's
try/except falls back to the existing per-family FE step -> zero behavior change until a later phase implements
the resident pipeline and the resident flag is turned on. The branch in ``_run_fe_step`` is gated behind the
default-OFF resident flag, so STRICT itself is unaffected too."""
from __future__ import annotations

import os


def fe_gpu_strict_resident_enabled() -> bool:
    """Whether the resident GPU-strict FE stages are active. DEFAULT ON under ``MLFRAME_FE_GPU_STRICT``.

    The resident GPU stages (recipe replay, fourier detection, prewarp ALS, usability pool, pure-form
    retention, the CMI perm-null) are selection-equivalent to the CPU path and are now the DEFAULT behaviour
    of STRICT: whenever ``MLFRAME_FE_GPU_STRICT`` is on they engage. ``MLFRAME_FE_GPU_STRICT_RESIDENT=0`` is
    the explicit OPT-OUT (kept so the resident path can still be disabled per-fit for diagnosis / rollback
    without touching the byte-identical DEFAULT non-strict path). STRICT itself is a selection-equivalent
    force-GPU mode (the CPU/CUDA backends agree to ~1e-9); the byte-identical contract lives on the non-strict
    default path, which this never touches."""
    if os.environ.get("MLFRAME_FE_GPU_STRICT_RESIDENT", "1").strip().lower() not in ("1", "true", "on", "yes"):
        return False
    try:
        from .._fe_gpu_strict import fe_gpu_strict_enabled
        return bool(fe_gpu_strict_enabled())
    except Exception:
        return False


def fe_gpu_strict_bytematch_enabled() -> bool:
    """Whether the STRICT-resident gate MI uses the RANK binner for a byte-match with the CPU rank MI.

    DEFAULT OFF. Requires the resident path (``fe_gpu_strict_resident_enabled``) AND the opt-in
    ``MLFRAME_FE_GPU_STRICT_BYTEMATCH=1``. With it OFF the resident gate MI uses the FAST percentile-edge
    binner (selection-equivalent to CPU on F2 -- the gate's edge-vs-rank difference does not flip the F2
    selection, only the gate's lift MAGNITUDE on heavily-tied operator outputs). With it ON the gate MI bins by
    argsort equi-frequency RANK so it byte-matches the CPU njit rank MI, at the cost of an irreducible per-gate
    argsort (~1s on a full fit, GTX 1050 Ti). Read live (no frozen cache) so it tracks the env per call."""
    if os.environ.get("MLFRAME_FE_GPU_STRICT_BYTEMATCH", "").strip().lower() not in ("1", "true", "on", "yes"):
        return False
    return fe_gpu_strict_resident_enabled()


def run_fe_step_gpu_strict(self, **kwargs):
    """One FE step, fully GPU-resident, multi-GPU + hw-spec aware. Returns the SAME contract as
    ``_run_fe_step`` (``data, cols, nbins, X, selected_vars, n_recommended_features`` + mutated
    ``engineered_recipes``).

    PHASE 0 STUB: not yet implemented. Raises ``NotImplementedError`` so ``_run_fe_step`` falls back to the
    existing per-family path (no behavior change). Implemented incrementally in Phases 1-3."""
    raise NotImplementedError("run_fe_step_gpu_strict: resident GPU-strict FE step not yet implemented (Phase 0 scaffold)")
