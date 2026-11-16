"""Parallel redundancy screen of the ordered-pair ratio and log-ratio pools.

``pairwise_ratio_features`` / ``pairwise_log_ratio_features`` used to build every one of the p (p - 1) candidate columns, run two numpy standard deviations and two correlation calls on it, and then keep
the few that pass, one ordered pair at a time in a Python loop. ``ratio_pair_screen`` makes the same per-pair decisions (candidate constant -> rejected; a source with no spread is skipped; a candidate
whose masked |Pearson| with either source exceeds the threshold -> rejected) for all pairs in one numba kernel parallel over the first operand, with one candidate buffer per task, so only the accepted
columns are materialised afterwards.
"""

from __future__ import annotations

import numba
import numpy as np

from ._feature_engineering_pairs._pairs_core_steps import _abs_corr_finite_njit

RATIO = 0
LOG_RATIO = 1


@numba.njit(cache=True, fastmath=False, nogil=True)
def _redundant_with(cand, mat, finite, src_std, src, threshold):
    """Whether the candidate column is more correlated than ``threshold`` (masked |Pearson|) with source row ``src``; a source without spread never makes it redundant."""
    if src_std[src] <= 1e-12:
        return False
    return _abs_corr_finite_njit(cand, mat[src], finite[src], 2) > threshold


@numba.njit(cache=True, fastmath=False, nogil=True, parallel=True)
def _screen_njit(mat, finite, src_std, kind, eps, threshold):
    """``accept[a, b]`` for every ordered pair of the rows of ``mat`` (p, n): the redundancy verdict of the ratio (``kind`` 0) or log-ratio (1) column of ``a`` over ``b``."""
    p, n = mat.shape
    accept = np.zeros((p, p), dtype=np.bool_)
    for a in numba.prange(p):
        cand = np.empty(n, dtype=mat.dtype)
        for b in range(p):
            if a == b:
                continue
            for i in range(n):
                av = mat[a, i]
                bv = mat[b, i]
                if kind == 0:
                    den = (1.0 if bv >= 0.0 else -1.0) * max(abs(bv), eps)
                    v = av / den
                else:
                    v = np.log1p(abs(av) + eps) - np.log1p(abs(bv) + eps)
                cand[i] = v if np.isfinite(v) else 0.0
            if cand.std() <= 1e-12:
                continue
            ok = not _redundant_with(cand, mat, finite, src_std, np.int64(a), threshold) and not _redundant_with(cand, mat, finite, src_std, np.int64(b), threshold)
            accept[a, b] = ok
    return accept


def ratio_pair_screen_host(columns: list, kind: int, eps: float, threshold: float) -> np.ndarray:
    """Host (numba) form: boolean (p, p) matrix of the ordered pairs whose ratio (``kind`` ``RATIO``) or log-ratio (``LOG_RATIO``) column passes the redundancy check; ``columns`` are the p source arrays."""
    mat = np.ascontiguousarray(np.vstack([np.asarray(c).reshape(1, -1) for c in columns]))
    finite = np.isfinite(mat)
    src_std = np.asarray([float(c.std()) for c in mat])
    return np.asarray(_screen_njit(mat, finite, src_std, int(kind), float(eps), float(threshold)))


def ratio_pair_screen(columns: list, kind: int, eps: float, threshold: float) -> np.ndarray:
    """Boolean (p, p) accept matrix of the ratio / log-ratio pool: the device kernel of ``_ratio_delta_screen_gpu`` when the GPU strategies are engaged and the tuned choice (``ratio_pair_screen_backend``) says so, else the numba form."""
    n = int(columns[0].shape[0]) if columns else 0
    try:
        from ._fe_gpu_strict import fe_gpu_strict_enabled
        from ._ratio_delta_screen_gpu import ratio_pair_screen_device, ratio_pair_screen_on_device

        if columns and fe_gpu_strict_enabled() and ratio_pair_screen_on_device(n):
            import cupy as cp

            return ratio_pair_screen_device(cp.asarray(np.ascontiguousarray(np.vstack([np.asarray(c).reshape(1, -1) for c in columns]))), kind, eps, threshold)
    except ImportError:
        pass
    return ratio_pair_screen_host(columns, kind, eps, threshold)
