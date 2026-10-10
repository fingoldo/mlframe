"""njit kernels of the row-statistic operator: one row statistic over a subset of (standardised) columns, scored by plug-in MI.

``eval_candidates`` scores a batch of ``(subset, statistic)`` candidates in parallel without storing a single candidate column: for each one it computes the statistic of a ``N_EDGE_SAMPLE``
sample of the even rows to take the bin edges, then one pass over all rows bins the statistic, counts the joint histogram on the even rows (the rows the search selects on) and on the odd rows
(held out), and returns both MIs. The statistic of a row is computed in place from the ``k`` standardised values of the subset (sorting for the median is an insertion sort of at most ``MAX_SUBSET``
values), so the work per candidate is ``n * k`` and nothing needs the full-column argsort a quantile binning would.
"""

from __future__ import annotations

import numpy as np

from ._offset_product_kernels import _bin_of
from numba import njit, prange

__all__ = ["STAT_NAMES", "MAX_SUBSET", "LSE_BETA", "N_EDGE_SAMPLE", "eval_candidates", "stat_column"]

STAT_NAMES = ("min", "max", "med", "rng", "std", "lse_pos", "lse_neg")
MAX_SUBSET = 8  # columns in a subset (the search stops growing here)
LSE_BETA = 4.0  # sharpness of the soft max / soft min
N_EDGE_SAMPLE = 2048  # even rows whose statistic defines the bin edges

_MIN, _MAX, _MED, _RNG, _STD, _LSE_POS, _LSE_NEG = 0, 1, 2, 3, 4, 5, 6


@njit(cache=True, nogil=True)
def _stat_of(vals, k, stat):
    """The statistic ``stat`` of the first ``k`` entries of ``vals`` (``vals`` may be reordered by the median)."""
    if stat == _MIN:
        r = vals[0]
        for j in range(1, k):
            r = min(r, vals[j])
        return r
    if stat == _MAX:
        r = vals[0]
        for j in range(1, k):
            r = max(r, vals[j])
        return r
    if stat == _RNG:
        lo = vals[0]
        hi = vals[0]
        for j in range(1, k):
            lo = min(lo, vals[j])
            hi = max(hi, vals[j])
        return hi - lo
    if stat == _MED:
        for a in range(1, k):
            v = vals[a]
            b = a - 1
            while b >= 0 and vals[b] > v:
                vals[b + 1] = vals[b]
                b -= 1
            vals[b + 1] = v
        if k % 2 == 1:
            return vals[k // 2]
        return 0.5 * (vals[k // 2 - 1] + vals[k // 2])
    if stat == _STD:
        m = 0.0
        for j in range(k):
            m += vals[j]
        m /= k
        s = 0.0
        for j in range(k):
            d = vals[j] - m
            s += d * d
        return np.sqrt(s / k)
    beta = LSE_BETA if stat == _LSE_POS else -LSE_BETA
    mx = beta * vals[0]
    for j in range(1, k):
        mx = max(mx, beta * vals[j])
    acc = 0.0
    for j in range(k):
        acc += np.exp(beta * vals[j] - mx)
    return (mx + np.log(acc / k)) / beta


@njit(cache=True, nogil=True)
def _row_value(Zt, subset, k, stat, i, vals):
    """The statistic of row ``i`` over the columns ``subset[:k]`` of the ``(p, n)`` block ``Zt``."""
    for j in range(k):
        vals[j] = Zt[subset[j], i]
    return _stat_of(vals, k, stat)


@njit(cache=True, nogil=True)
def _mi(counts, nb, ky, n_tot):
    """Plug-in MI (nats) of an ``(nb, ky)`` count table holding ``n_tot`` rows."""
    if n_tot <= 0:
        return 0.0
    px = np.zeros(nb)
    py = np.zeros(ky)
    for b in range(nb):
        for c in range(ky):
            px[b] += counts[b, c]
            py[c] += counts[b, c]
    mi = 0.0
    for b in range(nb):
        for c in range(ky):
            kk = counts[b, c]
            if kk > 0:
                mi += (kk / n_tot) * np.log(kk * n_tot / (px[b] * py[c]))
    return mi


@njit(parallel=True, cache=True)
def eval_candidates(Zt, subsets, sizes, stats, ycodes, ky, nb, out_even, out_odd):
    """Even-row and odd-row plug-in MI of each candidate.

    ``Zt``: ``(p, n)`` standardised columns; ``subsets``: ``(B, MAX_SUBSET)`` column indices, the first ``sizes[b]`` valid; ``stats``: ``(B,)`` statistic ids; ``ycodes``: class codes in
    ``[0, ky)`` of the ``n`` rows. Bin edges are the interior quantiles of the statistic over a sample of the even rows."""
    B = subsets.shape[0]
    n = Zt.shape[1]
    n_even = (n + 1) // 2
    m = min(N_EDGE_SAMPLE, n_even)
    for b in prange(B):
        k = sizes[b]
        vals = np.empty(MAX_SUBSET)
        sub = np.empty(m)
        for j in range(m):
            sub[j] = _row_value(Zt, subsets[b], k, stats[b], 2 * ((j * n_even) // m), vals)
        sub.sort()
        edges = np.empty(nb - 1)
        for q in range(1, nb):
            edges[q - 1] = sub[min(m - 1, (q * m) // nb)]
        counts = np.zeros((2, nb, ky), dtype=np.int64)
        for i in range(n):
            x = _row_value(Zt, subsets[b], k, stats[b], i, vals)
            counts[i & 1, _bin_of(x, edges), ycodes[i]] += 1
        out_even[b] = _mi(counts[0], nb, ky, n_even)
        out_odd[b] = _mi(counts[1], nb, ky, n - n_even)


@njit(parallel=True, cache=True)
def stat_column(Zt, subset, k, stat):
    """The statistic of every row of ``Zt`` over the columns ``subset[:k]`` (rows in parallel)."""
    n = Zt.shape[1]
    out = np.empty(n)
    for i in prange(n):
        vals = np.empty(MAX_SUBSET)
        out[i] = _row_value(Zt, subset, k, stat, i, vals)
    return out
