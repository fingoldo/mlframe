"""Compiled plug-in conditional MI over strata, and its within-stratum permutation null (triplet cross-basis scorer).

The scorer evaluates I(x; y | s) = sum over strata of (m_s / n) * I(x; y within s) for an engineered column and 199 within-stratum permutations of
it. Each evaluation used a Python loop over strata with a scikit-learn ``mutual_info_score`` call (2.4 ms each): 3248 evaluations were 79 s of a
154 s F2 fit. Here one pass over the rows fills an (strata, x levels, y levels) count table and the MI is read off it, in a parallel loop over
permutations. The permutations themselves are drawn by the caller (``permutation_sources``) from the same generator, in the same order, as the
per-stratum ``rng.permutation`` they replace, so the null is the same data.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _cmi_from_counts(counts: np.ndarray, n: int, min_rows: int) -> float:
    """Sum over strata with at least ``min_rows`` rows of (m / n) * plug-in MI (nats) of that stratum's x-y table, clipped at zero as scikit-learn does."""
    n_s, n_x, n_y = counts.shape
    total = 0.0
    px = np.zeros(n_x, dtype=np.int64)
    py = np.zeros(n_y, dtype=np.int64)
    for sv in range(n_s):
        m = 0
        for a in range(n_x):
            px[a] = 0
        for b in range(n_y):
            py[b] = 0
        for a in range(n_x):
            for b in range(n_y):
                c = counts[sv, a, b]
                px[a] += c
                py[b] += c
                m += c
        if m < min_rows:
            continue
        mi = 0.0
        for a in range(n_x):
            if px[a] == 0:
                continue
            for b in range(n_y):
                c = counts[sv, a, b]
                if c == 0:
                    continue
                mi += (c / m) * math.log((c * m) / (px[a] * py[b]))
        if mi > 0.0:
            total += (m / n) * mi
    return total


@njit(cache=True)
def stratified_cmi(x: np.ndarray, y: np.ndarray, s: np.ndarray, min_rows: int) -> float:
    """Plug-in I(x; y | s) summed over the strata of ``s`` (non-negative integer codes), skipping strata below ``min_rows`` rows."""
    n = x.shape[0]
    counts = np.zeros((int(s.max()) + 1, int(x.max()) + 1, int(y.max()) + 1), dtype=np.int64)
    for i in range(n):
        counts[s[i], x[i], y[i]] += 1
    cmi: float = _cmi_from_counts(counts, n, min_rows)
    return cmi


@njit(cache=True, parallel=True)
def stratified_cmi_null(x: np.ndarray, y: np.ndarray, s: np.ndarray, sources: np.ndarray, min_rows: int) -> np.ndarray:
    """I(x_p; y | s) for every row ``p`` of ``sources``, where ``x_p[i] = x[sources[p, i]]`` is x permuted inside each stratum of ``s``."""
    n_perm = sources.shape[0]
    n = x.shape[0]
    n_s = int(s.max()) + 1
    n_x = int(x.max()) + 1
    n_y = int(y.max()) + 1
    out = np.zeros(n_perm, dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    for p in prange(n_perm):
        counts = np.zeros((n_s, n_x, n_y), dtype=np.int64)
        for i in range(n):
            counts[s[i], x[sources[p, i]], y[i]] += 1
        out[p] = _cmi_from_counts(counts, n, min_rows)
    return out


def permutation_sources(s: np.ndarray, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """(n_perm, n) int32 source positions: row p permutes the positions of every stratum of ``s`` among themselves.

    One ``rng.permutation`` per stratum and draw, strata in ascending order, as the within-stratum shuffle this replaces consumed the generator.
    """
    groups = [np.flatnonzero(s == sv) for sv in np.unique(s)]
    src = np.empty((int(n_perm), s.shape[0]), dtype=np.int32)
    for p in range(int(n_perm)):
        for idx in groups:
            src[p, idx] = rng.permutation(idx)
    return src


__all__ = ["permutation_sources", "stratified_cmi", "stratified_cmi_null"]
