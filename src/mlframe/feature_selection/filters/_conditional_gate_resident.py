"""Resident-candidate and row-argmax helpers of the conditional-gate / row-argmax scans.

Carved out of ``_conditional_gate_fe`` (which re-exports every name here) to keep that module under the line budget: the switch that uploads a scored candidate once, the resident handle,
and the host and device-born first-max index of the operand columns."""

from __future__ import annotations

import logging
import os

import numpy as np

from ._fe_usability_signal import _crit_np_dtype

logger = logging.getLogger("mlframe.feature_selection.filters._conditional_gate_fe")


def _gate_cands_resident() -> bool:
    """Whether the SCORED gate / row-argmax candidate float is uploaded ONCE and its resident cupy handle threaded
    through the marginal MI + the 12-perm null - so the candidate never re-crosses H2D at the ``_mi_classif_batch``
    :318 upload site (measured: 80 MB/x10 from ``cheap_row_argmax_scan``, 3 MB/x12 from ``_perm_null_hi`` on a 1M
    STRICT-resident F2 fit). Default ON under ``fe_gpu_strict_resident_enabled`` + ``_cmi_gpu_enabled`` (the same
    predicate pair the CMI-gate / raw-redundancy candidate residency uses); opt-out ``MLFRAME_FE_GATE_RESIDENT_CANDS=0``.
    Any import fault -> off (the exact host upload path runs)."""
    if os.environ.get("MLFRAME_FE_GATE_RESIDENT_CANDS", "1").strip().lower() not in ("1", "true", "on", "yes"):
        return False
    try:
        from ._gpu_strict_fe import fe_gpu_strict_resident_enabled
        from ._mi_greedy_cmi_fe import _cmi_gpu_enabled
        return bool(fe_gpu_strict_resident_enabled()) and bool(_cmi_gpu_enabled())
    except Exception as _flag_exc:
        # fallback: the GPU-strict flag probe is False on CPU-only hosts, so resident candidates stay disabled.
        logger.debug("gate GPU-strict-resident flag probe failed (%s); resident candidates disabled.", _flag_exc)
        return False

def _resident_cand(feat: np.ndarray, role):
    """Upload a SCORED candidate float column ONCE and return its RESIDENT cupy handle (content-keyed via
    ``resident_operand`` under ``role``), so ``_mi`` / ``_perm_null_hi`` read it device-side instead of re-uploading
    the same host float at every MI kernel call. Returns the ORIGINAL host array unchanged when residency is off, on a
    non-finite column (the resident percentile-edge binner needs an all-finite operand -> host path preserves the CPU
    NaN handling), or on ANY cupy fault -> the host MI path runs (selection-identical, no work hidden)."""
    if not _gate_cands_resident():
        return feat
    host = np.ascontiguousarray(np.asarray(feat, dtype=_crit_np_dtype())).ravel()
    if not np.all(np.isfinite(host)):
        return feat
    try:
        from ._fe_resident_operands import resident_operand
        import cupy as cp  # noqa: F401  (import guard: skip residency when cupy is absent)
        return resident_operand(host, role, dtype=_crit_np_dtype())
    except Exception:
        logger.debug("gate resident-candidate upload failed; host fallback", exc_info=True)
        return feat

def _argmax_host(columns):
    """First-max index of the host operand columns in the crit dtype: the parallel numba kernel for finite columns, ``np.argmax`` (its NaN semantics) otherwise."""
    from ._row_argmax_gpu import all_finite_cached, row_argmax_host

    cols = [np.ascontiguousarray(np.asarray(c, dtype=_crit_np_dtype())) for c in columns]
    if all(all_finite_cached(c) for c in cols):
        return row_argmax_host(cols).astype(_crit_np_dtype())
    return np.argmax(np.stack(cols, axis=1), axis=1).astype(_crit_np_dtype())

def _argmax_resident(arrs, tri):
    """DEVICE-BORN the row-argmax candidate ``argmax_k(operand_k)`` from the RESIDENT operand columns, returning
    the resident double index column, or ``None`` when residency is off / any operand is non-finite / cupy faults
    (the caller then takes the host ``np.argmax`` + one-shot resident upload).

    ``cp.argmax`` returns the FIRST occurrence of the max along the axis - identical to ``np.argmax`` - and the
    operand floats are byte-identical to the host columns, so the argmax INDEX codes are selection-identical. The
    win: the DISTINCT argmax candidate is BORN on device from the raw operands (each uploaded once, shared via the
    content-keyed cache) and never crosses H2D, instead of a host ``np.argmax`` whose float column re-uploads."""
    if not _gate_cands_resident():
        return None
    try:
        import cupy as cp

        from ._fe_resident_operands import resident_operand
        from ._row_argmax_gpu import all_finite_cached, row_argmax_cm, row_argmax_on_device
        _ops: list = []
        for c in tri:
            _h = np.ascontiguousarray(np.asarray(arrs[c], dtype=_crit_np_dtype()))
            if not all_finite_cached(_h):
                return None  # host np.argmax NaN semantics differ -> keep the host path for a non-finite operand
            if not _ops and not row_argmax_on_device(int(_h.shape[0])):
                return None  # the per-host kernel-tuning cache prefers the host numba kernel at this row count
            _ops.append(resident_operand(_h, ("argmax_op", str(c)), dtype=_crit_np_dtype()))
        return row_argmax_cm(cp, _ops)  # first-max index as double, one launch (cp.argmax over the interleaved stack ran a 15 ms cub reduction)
    except Exception:
        logger.debug("device-born row-argmax failed; host fallback", exc_info=True)
        return None
