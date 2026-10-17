"""Parallel quantile coding of a block of candidate columns for the family-wise permutation test.

``_family_maxt_gate._quantile_codes`` codes one column with ``np.unique`` / ``np.quantile`` / ``np.searchsorted`` (about 6 ms for 40000 rows, three sorts); a pair or modular family has
thousands of columns, so the loop was 22 s of a 50000-row conjunction fit. Here each column of a block is sorted once, its distinct levels counted from the sorted copy, and the interior
edges taken from the same sorted copy with numpy's ``linear`` quantile arithmetic (virtual index, gamma, ``lerp`` with the ``>= 0.5`` branch) so the codes equal the numpy path's.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _edge_value(s: np.ndarray, q: float) -> float:
    """``np.quantile(s_unsorted, q)`` (method ``linear``) read off the sorted copy ``s``, in numpy's arithmetic order."""
    n = s.shape[0]
    virtual = (n - 1) * q
    lo = np.floor(virtual)
    if virtual >= n - 1:
        i0 = n - 1
        i1 = n - 1
    elif virtual < 0:
        i0 = 0
        i1 = 0
    else:
        i0 = int(lo)
        i1 = i0 + 1
    gamma = virtual - i0
    a = float(s[i0])
    b = float(s[i1])
    diff = b - a
    out = a + diff * gamma
    if gamma >= 0.5:
        out = b - diff * (1.0 - gamma)
    return out


@njit(cache=True, parallel=True)
def quantile_codes_block(v: np.ndarray, n_bins: int, codes: np.ndarray, nx: np.ndarray) -> None:
    """Fill ``codes[j]`` (int16) and ``nx[j]`` for every row ``j`` of ``v`` ``(columns, n)`` exactly as ``_quantile_codes`` does for a finite column."""
    k = v.shape[0]
    n = v.shape[1]
    step = 1.0 / n_bins
    for j in prange(k):
        s = np.sort(v[j])
        d = 1
        for i in range(1, n):
            if s[i] != s[i - 1]:
                d += 1
        if d <= n_bins:
            levels = np.empty(d, dtype=s.dtype)
            levels[0] = s[0]
            m = 1
            for i in range(1, n):
                if s[i] != s[i - 1]:
                    levels[m] = s[i]
                    m += 1
            for i in range(n):
                lo = 0
                hi = d
                x = v[j, i]
                while lo < hi:
                    mid = (lo + hi) // 2
                    if levels[mid] < x:
                        lo = mid + 1
                    else:
                        hi = mid
                codes[j, i] = lo
            nx[j] = d
            continue
        edges = np.empty(n_bins + 1, dtype=np.float64)  # Intentional float64: quantile edges in numpy's arithmetic (equal codes to the numpy path)
        m = 0
        for b in range(n_bins + 1):
            q = 1.0 if b == n_bins else b * step
            e = _edge_value(s, q)
            if m == 0 or e != edges[m - 1]:
                edges[m] = e
                m += 1
        hi_code = max(m - 2, 0)
        top = 0
        for i in range(n):
            x = v[j, i]
            lo = 0
            hi = m
            while lo < hi:
                mid = (lo + hi) // 2
                if edges[mid] <= x:
                    lo = mid + 1
                else:
                    hi = mid
            c = lo - 1
            if c < 0:
                c = 0
            elif c > hi_code:
                c = hi_code
            codes[j, i] = c
            if c > top:
                top = c
        nx[j] = top + 1
