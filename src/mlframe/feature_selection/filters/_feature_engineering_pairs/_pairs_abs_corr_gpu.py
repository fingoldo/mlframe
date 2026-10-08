"""Device twin of ``_abs_corr_finite_njit`` for the deferred-float GPU pair path.

The pair emit step scores each leader column's |corr(y)| for the linear-usability tie-break. On the deferred-float path the candidate column only exists on
the device, so the host formulation copied the whole ``(n,)`` column back (25 columns x 120 KB per small fit, growing with n) to reduce it to ONE scalar.
This computes the same statistic on the device and returns just that scalar.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, cast

import numpy as np

from ._pairs_common import _DEGENERATE_REL_TOL

logger = logging.getLogger(__name__)


def abs_corr_finite_gpu(a_dev: Any, y: np.ndarray, y_finite: np.ndarray, min_n: int = 8) -> float:
    """|Pearson corr| of the device column ``a_dev`` against the host target ``y`` over rows where both are finite; same two-pass arithmetic and degenerate
    handling as ``_abs_corr_finite_njit`` (``0.0`` below ``min_n`` joint-finite rows or for a near-constant side). Only scalars cross to the host."""
    import cupy as cp

    from mlframe.feature_selection.filters._fe_resident_operands import resident_operand

    y_dev = resident_operand(y, ("abs_corr_y",), dtype=np.float64).ravel()
    fin_dev = resident_operand(np.asarray(y_finite, dtype=np.uint8), ("abs_corr_yfin",), dtype=np.uint8).ravel() != 0
    a = a_dev.astype(cp.float64, copy=False).ravel()
    # Masked sums instead of boolean-index compaction: ``a[mask]`` and ``int(count_nonzero)`` each block the stream on a device read, and this runs per
    # candidate. Everything stays on the device and ONE small vector comes back at the end.
    mask = fin_dev & cp.isfinite(a)
    m = mask.astype(cp.float64)
    n = m.sum()
    a0 = cp.where(mask, a, 0.0)
    y0 = cp.where(mask, y_dev, 0.0)
    da = (a0 - (a0 * m).sum() / n) * m
    dy = (y0 - (y0 * m).sum() / n) * m
    stats = cp.stack([n, (da * da).sum(), (dy * dy).sum(), (da * dy).sum(), cp.abs(a0).max(), cp.abs(y0).max()]).get()
    cnt, va, vy, cay, amax, ymax = (float(v) for v in stats)
    if cnt < min_n:
        return 0.0
    n_i = int(cnt)
    if va <= n_i * (_DEGENERATE_REL_TOL * amax) ** 2 or vy <= n_i * (_DEGENERATE_REL_TOL * ymax) ** 2:
        return 0.0
    denom = (va * vy) ** 0.5
    if denom <= 0.0:
        return 0.0
    return float(abs(cay / denom))


def abs_corr_or_none(a_dev: Any, y: Optional[np.ndarray], y_finite: Optional[np.ndarray]) -> Optional[float]:
    """``abs_corr_finite_gpu`` with the host helper's contract: ``0.0`` when there is no target or the lengths differ, and ``None`` on any device fault so
    the caller falls back to the host column."""
    if y is None or y_finite is None:
        return 0.0
    if int(a_dev.shape[0]) != int(y.shape[0]):
        return 0.0
    try:
        return abs_corr_finite_gpu(a_dev, y, y_finite, 8)
    except Exception as e:
        logger.debug("device |corr| failed, caller falls back to the host column: %s", e)
        return None


def candidate_abs_corr(resolve_col: Any, safe_abs_corr: Any, buf_col: int) -> Optional[float]:
    """|corr(y)| of candidate column ``buf_col`` via the device capability of a deferred-float ``resolve_col`` and the target held by ``safe_abs_corr``,
    or ``None`` when either is absent (host buffers, no target) or the device path faulted - the caller then reads the host column."""
    fn = getattr(resolve_col, "abs_corr", None)
    target = getattr(safe_abs_corr, "target", None)
    if fn is None or target is None or target[0] is None:
        return None
    return cast("Optional[float]", fn(buf_col, target[0], target[1]))


def abs_corr_zerofill_gpu(a_dev: Any, b: np.ndarray) -> float:
    """Device twin of ``_abs_corr_zerofill_njit``: |Pearson corr| of the device column ``a_dev`` against the host column ``b`` with every non-finite entry
    read as 0 (the zero-fill statistic the degenerate-pair veto was written against, NOT the masked one). Only scalars cross to the host."""
    import cupy as cp

    from mlframe.feature_selection.filters._fe_resident_operands import resident_operand

    a = a_dev.astype(cp.float64, copy=False).ravel()
    bd = resident_operand(np.ascontiguousarray(np.asarray(b, dtype=np.float64)), ("zerofill_operand",), dtype=np.float64).ravel()
    n = int(a.shape[0])
    if n == 0:
        return 0.0
    a = cp.where(cp.isfinite(a), a, 0.0)
    bd = cp.where(cp.isfinite(bd), bd, 0.0)
    da = a - a.sum() / n
    db = bd - bd.sum() / n
    stats = cp.stack([(da * da).sum(), (db * db).sum(), (da * db).sum(), cp.abs(a).max(), cp.abs(bd).max()]).get()
    va, vb, cab, amax, bmax = (float(v) for v in stats)
    if va <= n * (_DEGENERATE_REL_TOL * amax) ** 2 or vb <= n * (_DEGENERATE_REL_TOL * bmax) ** 2:
        return 0.0
    denom = (va * vb) ** 0.5
    if denom <= 0.0:
        return 0.0
    return float(abs(cab / denom))
