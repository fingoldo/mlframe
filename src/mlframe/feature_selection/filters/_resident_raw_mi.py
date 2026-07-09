"""Resident-operand MI for FIT-CONSTANT raw baseline matrices (the class-B :311 H2D collapse, 2026-06-30).

Several FE scorers compute ``MI(col; y)`` over a matrix that is a PURE FIT-CONSTANT raw-operand baseline --
the raw numeric columns verbatim (the unified-gate noise floor ``raw_mi_noise_floor``), the gate-prune raw
relevance ranking (``_conditional_gate_fe._rank_and_prune``'s ``column_stack`` of the candidate columns), and
the orth-univariate uplift RAW baseline (``score_features_by_mi_uplift``'s ``raw_X.to_numpy()``). Under
``MLFRAME_FE_GPU_STRICT`` each of those ``_mi_classif_batch`` calls already routes through the resident plug-in
MI, but the host matrix is ``cp.asarray``-uploaded FRESH on every call at ``_orth_mi_backends.py:311`` (the
matrix is treated as a per-call transient there; only ``y`` was routed to the resident-operand cache). For
these THREE callers the matrix is NOT a transient -- it is the SAME fit-constant raw columns re-scored across
the fit -- so it can ride the resident-operand cache and upload ONCE.

This module is the class-B route the per-sub-family residency plan identified: take a host ``(n, k)``
fit-constant matrix + a stable ROLE key, upload it ONCE via ``resident_operand`` (keyed role + shape +
content-fingerprint), and score it through the SAME percentile-EDGE (or argsort-RANK, when the caller's gate MI
uses rank binning) resident plug-in MI the host STRICT ``_mi_classif_batch`` routes to -- no estimator switch.
The y label vector rides the SAME ``"y_mi_classif"`` resident role the STRICT MI path already uses, so the
baseline shares the cached y buffer.

SELECTION-EQUIVALENCE: the resident plug-in over the SAME matrix + SAME y + SAME (edge|rank) binner is the
EXACT estimator the host STRICT ``_mi_classif_batch`` already invokes at :311 (this only removes the redundant
re-upload), so the per-column MI -- and every downstream median/MAD floor / argsort ranking / uplift baseline
built on it -- is byte-identical to the host STRICT path. The NON-strict default path is untouched (the caller
runs the host ``_mi_classif_batch``) -> byte-identical there too.

GATE: engages ONLY under ``fe_gpu_strict_resident_enabled`` (so the non-strict default never reaches it) AND
the caller's per-family opt-out. On ANY cupy failure / no cupy it returns ``None`` and the caller takes the
EXACT host scorer. NEVER ``free_all_blocks`` (the mempool teardown owns that).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["resident_raw_baseline_mi"]


def resident_raw_baseline_mi(
    mat: np.ndarray,
    y: Any,
    role_key: Any,
    *,
    nbins: int,
    rank_binning: bool = False,
) -> Optional[np.ndarray]:
    """Score per-column ``MI(col; y)`` of a FIT-CONSTANT host ``(n, k)`` raw baseline matrix through the resident
    plug-in MI, uploading the matrix ONCE via the resident-operand cache (keyed ``role_key``).

    ``role_key`` is a stable ROLE discriminator (e.g. ``("gate_prune_raw", cols_tuple)``); the matrix shape +
    dtype + a content fingerprint are folded in by ``resident_operand`` so a same-role matrix with different
    VALUES re-uploads rather than aliasing a stale buffer. ``rank_binning`` selects the argsort equi-frequency
    RANK resident binner (the gate MI byte-match path) instead of the percentile-EDGE binner -- it mirrors the
    exact dispatch of ``_orth_mi_backends._mi_classif_batch:333`` so the resident MI matches the host STRICT MI
    the caller would otherwise compute.

    Returns a host ``(k,)`` float64 MI array, OR ``None`` when STRICT-residency is off / cupy is unavailable /
    any cupy fault -- in which case the caller MUST fall back to the EXACT host ``_mi_classif_batch`` (the
    byte-identical default path)."""
    try:
        from ._gpu_strict_fe import fe_gpu_resident_raw_baseline_enabled

        if not fe_gpu_resident_raw_baseline_enabled():
            return None
    except Exception:
        return None
    try:
        import cupy as cp  # noqa: F401
    except Exception:
        return None
    try:
        host = np.ascontiguousarray(np.asarray(mat, dtype=np.float64))
        if host.ndim == 1:
            host = host[:, None]
        n, k = host.shape
        if n == 0 or k == 0:
            return np.zeros(k, dtype=np.float64)

        from ._fe_resident_operands import resident_operand, assemble_resident_matrix
        from ._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident

        # DEVICE-ASSEMBLE the raw baseline matrix from the resident PER-COLUMN operands rather than uploading it
        # whole: this same raw data is already resident column-by-column under the ("xbasis_op", col) role (the
        # basis / cross-basis device builders upload each source column once), so stacking the resident columns
        # on device content-hits that cache and the (n, k) matrix never crosses H2D. ``role_key`` carries the
        # column names as its second element (e.g. ("raw_noise_floor", tuple(cols))), and column j of ``host`` is
        # that column verbatim (the callers build ``raw_X[cols].to_numpy()``), so the per-column upload is the
        # SAME bytes -> content-keyed dedup -> selection-identical. Any shape/name mismatch -> upload the matrix.
        _names = role_key[1] if (isinstance(role_key, tuple) and len(role_key) >= 2 and isinstance(role_key[1], (tuple, list))) else None
        Xd = assemble_resident_matrix(host, _names, role_key, dtype=cp.float64)

        _yi = np.ascontiguousarray(np.asarray(y)).astype(np.int64).ravel()
        yd = resident_operand(_yi, "y_mi_classif", dtype=np.int64)
        _ymin = int(_yi.min()) if _yi.size else 0
        _ncls = (int(_yi.max()) - _ymin + 1) if _yi.size else 1

        if rank_binning:
            from ._gpu_resident_rank_bin import plugin_mi_classif_batch_rank_cuda_resident

            rank_mi = plugin_mi_classif_batch_rank_cuda_resident(Xd, yd, int(nbins), y_min=_ymin, n_classes=_ncls)
            if rank_mi is not None:
                return np.asarray(rank_mi, dtype=np.float64)
            # rank resident path unavailable -> host fallback (caller takes the CPU njit rank MI).
            return None
        return np.asarray(
            _plugin_mi_classif_batch_cuda_resident(Xd, yd, int(nbins), y_min=_ymin, n_classes=_ncls),
            dtype=np.float64,
        )
    except Exception as _exc:
        logger.debug("resident_raw_baseline_mi: GPU path failed (%s); host fallback", _exc)
        return None
