"""Out-of-fold nonparametric warp ``E[y | x]`` of one numeric column (the shared service of the warp operator and, later, of the 2-D cell table).

The warp replaces a column by the calibrated mean of the (rank-scaled) target in its quantile bin, interpolated linearly between bin centres. A non-monotone effect such as ``sin(9.4 x)``
becomes monotone-in-the-target and so linearly usable, and a 10-bin MI that cannot resolve 1.5 periods of the raw column sees the oscillation.

Leak safety: bin EDGES depend on ``x`` only (they are computed once, from the rows given); the table values are cross-fitted. A training row in fold ``f`` gets its value from the table fitted
on the other folds (bin sums minus the fold's own sums, shrunk toward the mean of those rows); rows that were not used to fit anything (``x`` rows beyond the fitting rows) and new data get the
full table, which never saw them. The recipe stores the full table only (``cx``, ``cy``) and the fill for non-finite ``x``; replay is a pure function of the source column.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["N_BINS", "SHRINK", "N_FOLDS", "quantile_edges", "fit_oof_warp1d", "apply_warp1d"]

N_BINS = 20  # bins of the warp table
SHRINK = 10.0  # pseudo-counts that pull a bin mean toward the mean of the fitted rows
N_FOLDS = 5  # cross-fitting folds


def quantile_edges(x: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Unique interior equal-frequency edges of the finite values of ``x`` (up to ``n_bins - 1`` of them)."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.empty(0)
    return np.unique(np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))


@njit(cache=True, nogil=True)
def _bin_codes(x, edges):
    """Bin code of each value (``-1`` for a non-finite one): the number of edges ``<= x``."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.int64)
    ne = edges.shape[0]
    for i in range(n):
        v = x[i]
        if not np.isfinite(v):
            out[i] = -1
            continue
        lo = 0
        hi = ne
        while lo < hi:
            mid = (lo + hi) >> 1
            if edges[mid] <= v:
                lo = mid + 1
            else:
                hi = mid
        out[i] = lo
    return out


@njit(cache=True, nogil=True)
def _table(sx, sy, cnt, mu, m, cx, cy):
    """Compress the bins with rows into ``(cx, cy)`` (bin mean of x, shrunk bin mean of y, in bin order); returns how many bins hold rows."""
    k = 0
    for b in range(cnt.shape[0]):
        if cnt[b] > 0.0:
            cx[k] = sx[b] / cnt[b]
            cy[k] = (sy[b] + m * mu) / (cnt[b] + m)
            k += 1
    return k


@njit(cache=True, nogil=True)
def _interp(v, cx, cy, k, fill):
    """``np.interp`` of one value on the first ``k`` entries of ``(cx, cy)``; ``fill`` for a non-finite value."""
    if not np.isfinite(v) or k == 0:
        return fill
    if v <= cx[0]:
        return cy[0]
    if v >= cx[k - 1]:
        return cy[k - 1]
    lo = 0
    hi = k - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if cx[mid] <= v:
            lo = mid
        else:
            hi = mid
    d = cx[hi] - cx[lo]
    if d <= 0.0:
        return cy[lo]
    return cy[lo] + (cy[hi] - cy[lo]) * (v - cx[lo]) / d


@njit(cache=True, nogil=True)
def _fit_core(x, y, b, fold, k_folds, nbin, m):
    """Cross-fitted warp of the fitting rows plus the full table.

    ``b`` are the bin codes of ``x`` (``-1`` where ``x`` or ``y`` is not finite), ``fold`` the fold of each row. Returns ``(oof, cx, cy, k, fill)``: the out-of-fold value of every fitting row,
    the full table (first ``k`` entries valid) and the mean of the fitted target (the fill for a non-finite ``x``)."""
    n = x.shape[0]
    sx = np.zeros((k_folds, nbin))
    sy = np.zeros((k_folds, nbin))
    cn = np.zeros((k_folds, nbin))
    fy = np.zeros(k_folds)
    fc = np.zeros(k_folds)
    for i in range(n):
        bi = b[i]
        if bi >= 0:
            f = fold[i]
            sx[f, bi] += x[i]
            sy[f, bi] += y[i]
            cn[f, bi] += 1.0
            fy[f] += y[i]
            fc[f] += 1.0
    tsx = sx.sum(axis=0)
    tsy = sy.sum(axis=0)
    tcn = cn.sum(axis=0)
    ty = fy.sum()
    tc = fc.sum()
    mu_all = ty / tc if tc > 0.0 else 0.0
    cx_all = np.empty(nbin)
    cy_all = np.empty(nbin)
    k_all = _table(tsx, tsy, tcn, mu_all, m, cx_all, cy_all)
    cxf = np.empty((k_folds, nbin))
    cyf = np.empty((k_folds, nbin))
    kf = np.empty(k_folds, dtype=np.int64)
    muf = np.empty(k_folds)
    for f in range(k_folds):
        c_tr = tc - fc[f]
        muf[f] = (ty - fy[f]) / c_tr if c_tr > 0.0 else mu_all
        kf[f] = _table(tsx - sx[f], tsy - sy[f], tcn - cn[f], muf[f], m, cxf[f], cyf[f])
    oof = np.empty(n)
    for i in range(n):
        f = fold[i]
        oof[i] = _interp(x[i], cxf[f], cyf[f], kf[f], muf[f])
    return oof, cx_all, cy_all, k_all, mu_all


def fit_oof_warp1d(x: np.ndarray, y: np.ndarray, *, n_bins: int = N_BINS, shrink: float = SHRINK, n_folds: int = N_FOLDS, seed: int = 0) -> dict:
    """Fit the warp of the fitting column ``x`` against the bounded, rank-scaled target ``y`` (same length) and return ``{"oof", "cx", "cy", "fill"}``.

    ``oof`` is the cross-fitted value of every fitting row (a row in fold ``f`` takes the table fitted on the other folds); ``(cx, cy)`` is the full table and ``fill`` the mean target, used for a
    non-finite ``x``. The bin edges come from ``x`` alone. Rows outside the fitting rows should be given the full table (``apply_warp1d``): it never saw them."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    edges = quantile_edges(x, n_bins)
    b = np.where(np.isfinite(y), _bin_codes(x, edges), -1)
    fold = (np.random.default_rng(seed).permutation(len(x)) % int(n_folds)).astype(np.int64)
    oof, cx, cy, k, fill = _fit_core(x, np.where(np.isfinite(y), y, 0.0), b, fold, int(n_folds), int(len(edges) + 1), float(shrink))
    return {"oof": oof, "cx": np.ascontiguousarray(cx[:k]), "cy": np.ascontiguousarray(cy[:k]), "fill": float(fill)}


@njit(cache=True, nogil=True)
def _apply(x, cx, cy, fill):
    """Table lookup of every value (``fill`` where ``x`` is not finite)."""
    n = x.shape[0]
    out = np.empty(n)
    k = cx.shape[0]
    for i in range(n):
        out[i] = _interp(x[i], cx, cy, k, fill)
    return out


def apply_warp1d(x: np.ndarray, cx: np.ndarray, cy: np.ndarray, fill: float) -> np.ndarray:
    """The warp table ``(cx, cy)`` applied to ``x``: piecewise-linear between the bin centres, clamped to the end values, ``fill`` for a non-finite ``x``."""
    return _apply(np.ascontiguousarray(x, dtype=np.float64), np.ascontiguousarray(cx, dtype=np.float64), np.ascontiguousarray(cy, dtype=np.float64), float(fill))
