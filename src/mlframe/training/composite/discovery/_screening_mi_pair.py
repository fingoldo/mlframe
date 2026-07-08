"""Pair-MI kernel for composite-target discovery screening (carved from ``screening.py``).

``_mi_from_binned_pair`` is the pair-MI kernel called ~9.8k times per discovery run (per-feature MI
AND inside the per-permutation null loop in ``_auto_base``). It dispatches to a ``numba.njit`` single-
pass joint-histogram+MI kernel, bit-identical to the numpy reference within FP reduction-order error.
"""
from __future__ import annotations

import numpy as np

try:
    import numba as _numba

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _numba = None
    _HAS_NUMBA = False


def _mi_from_binned_pair_numpy(
    x_idx: np.ndarray, y_idx: np.ndarray, *, nbins: int,
) -> float:
    """Numpy reference for :func:`_mi_from_binned_pair` (kept callable for tests / benches).

    ``x_idx`` / ``y_idx`` may arrive as int16 (the narrowed prebin code
    dtype). The flattened index ``x_idx*nbins + y_idx`` reaches ``nbins**2 - 1``,
    which overflows int16 at ``nbins >= 182`` (and even int16*python-scalar can
    wrap on intermediate products), so the combo is computed in int64 explicitly.
    This is purely a width promotion -- the index values are unchanged, so MI is
    bit-identical to the legacy int64-code path. ``np.bincount`` requires int.
    """
    combo = x_idx.astype(np.int64) * nbins + y_idx
    joint_counts = np.bincount(combo, minlength=nbins * nbins).reshape(nbins, nbins)
    n_total = float(joint_counts.sum())
    if n_total <= 0:
        return 0.0
    pxy = joint_counts.astype(np.float64) / n_total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    log_terms = np.zeros_like(pxy)
    log_terms[nz] = np.log(pxy[nz] / (px * py)[nz])
    return max(0.0, float((pxy * log_terms).sum()))


_mi_from_binned_pair_njit_kernel = None


if _HAS_NUMBA:
    @_numba.njit(cache=True, fastmath=False, nogil=True)
    def _mi_from_binned_pair_njit_kernel(x_idx, y_idx, nbins):
        # Single-pass joint histogram + marginals, then MI = sum pxy*log(pxy/(px*py)).
        # Reproduces the numpy reference's arithmetic term-for-term: the joint counts are
        # integer-exact (no FP), px/py are the same row/col sums, and each non-zero cell adds
        # pxy*log(pxy/(px*py)) -- the only divergence from numpy is FP reduction ORDER of the
        # final accumulation (numpy reduces the (nbins,nbins) product array; this walks cells
        # row-major), which lands ~1e-16 on the natural-log MI scale, far under the 1e-12 gate.
        n = x_idx.shape[0]
        joint = np.zeros(nbins * nbins, dtype=np.int64)
        for i in range(n):
            joint[int(x_idx[i]) * nbins + int(y_idx[i])] += 1
        n_total = 0
        for k in range(nbins * nbins):
            n_total += joint[k]
        if n_total <= 0:
            return 0.0
        inv_n = 1.0 / float(n_total)
        # Row (px) and column (py) marginal probabilities.
        px = np.zeros(nbins, dtype=np.float64)
        py = np.zeros(nbins, dtype=np.float64)
        for a in range(nbins):
            base = a * nbins
            for b in range(nbins):
                c = joint[base + b]
                if c != 0:
                    p = c * inv_n
                    px[a] += p
                    py[b] += p
        mi = 0.0
        for a in range(nbins):
            base = a * nbins
            pxa = px[a]
            for b in range(nbins):
                c = joint[base + b]
                if c != 0:
                    pxy = c * inv_n
                    mi += pxy * np.log(pxy / (pxa * py[b]))
        if mi < 0.0:
            return 0.0
        return mi


def _mi_from_binned_pair(
    x_idx: np.ndarray, y_idx: np.ndarray, *, nbins: int,
) -> float:
    """MI from two already-binned integer arrays (0..nbins-1).

    Hot kernel: called ~9.8k times per discovery run (per-feature MI AND inside the
    per-permutation null loop in ``_auto_base``, so the cost multiplies with
    ``n_targets x auto_base_top_k x npermutations``). Dispatches to a ``numba.njit``
    single-pass histogram+MI kernel (``cache=True``) which is ~4x faster than the numpy
    ``bincount``+log path at the production sample size and matches it within ULP-scale FP
    reduction-order error (<1e-9 worst-case, pinned by test_mi_kernel_divergence_bound; the
    joint integer counts are identical, only the final-sum/marginal reduction order differs --
    numpy reduces the (nbins,nbins) array pairwise, this walks cells row-major). That delta is
    far below any MI ranking threshold (~1e-3), so it cannot move a feature-selection decision.
    The numpy reference stays callable as
    ``_mi_from_binned_pair_numpy`` for tests / benches; falls back to it when numba is
    unavailable.
    """
    if not _HAS_NUMBA:
        return _mi_from_binned_pair_numpy(x_idx, y_idx, nbins=nbins)
    # The njit kernel indexes element-by-element (``int(x_idx[i])``) so it consumes a strided slice / any integer dtype directly; the prior ``np.ascontiguousarray`` forced a full O(n) copy of every strided ``feature_binned[:, j]`` column on each of the ~2.4k hot-loop calls -- pure waste, bit-identical to the contiguous path (verified across nbins x {int16,int32,int64}). Pass straight through.
    return float(_mi_from_binned_pair_njit_kernel(x_idx, y_idx, int(nbins)))  # type: ignore[misc]  # redefined to a real njit kernel under `if _HAS_NUMBA:` above; only None when the early return already fired
