"""Compiled plug-in conditional MI over strata, and its within-stratum permutation null (triplet cross-basis scorer).

The scorer evaluates I(x; y | s) = sum over strata of (m_s / n) * I(x; y within s) for an engineered column and 199 within-stratum permutations of
it. Each evaluation used a Python loop over strata with a scikit-learn ``mutual_info_score`` call (2.4 ms each): 3248 evaluations were 79 s of a
154 s F2 fit. Here one pass over the rows fills an (strata, x levels, y levels) count table and the MI is read off it, in a parallel loop over
permutations. The permutations themselves are drawn by the caller (``permutation_sources``) from the same generator, in the same order, as the
per-stratum ``rng.permutation`` they replace, so the null is the same data.
"""

from __future__ import annotations


import numpy as np
from numba import njit, prange

from ._numba_utils_shuffle import bounded, next_state, seed_state


from ._numba_utils import (  # noqa: F401 -- helpers hosted in _numba_utils_cross_module for the cross-module njit invariant
    _cmi_from_counts,
    stratified_cmi,
    stratified_cmi_null,
)


@njit(cache=True, parallel=True, nogil=True)
def _permutation_sources_njit(order: np.ndarray, starts: np.ndarray, lens: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    """Fisher-Yates shuffle of every stratum's positions, one draw per parallel iteration; draw ``p`` is seeded ``seed + p`` so the result does not depend on the thread schedule (index draws: ``_numba_utils_shuffle``)."""
    n = order.shape[0]
    src = np.empty((n_perm, n), dtype=np.int32)
    for p in prange(n_perm):
        x = seed_state(seed + p)
        work = order.copy()
        for g in range(starts.shape[0]):
            a = starts[g]
            for i in range(lens[g] - 1, 0, -1):
                x = next_state(x)
                j = bounded(x, i + 1)
                t = work[a + i]
                work[a + i] = work[a + j]
                work[a + j] = t
        for k in range(n):
            src[p, order[k]] = work[k]
    return src


def permutation_seed(rng: np.random.Generator) -> int:
    """The one draw that seeds every permutation of a null: permutation ``p`` is seeded ``seed + p``, so any slice of the null is the same data however it is chunked."""
    return int(rng.integers(0, 2**31 - 1))


def strata_layout(s: np.ndarray) -> tuple:
    """``(order, starts, lens)`` of the strata of ``s``: the rows in stratum order (stable), where each stratum starts and its length. Depends only on ``s``, so a null drawn in several chunks computes it once."""
    n = s.shape[0]
    order = np.argsort(s, kind="stable").astype(np.int64)
    sorted_s = s[order]
    starts = np.flatnonzero(np.r_[True, sorted_s[1:] != sorted_s[:-1]]).astype(np.int64)
    lens = np.diff(np.r_[starts, n]).astype(np.int64)
    return order, starts, lens


def permutation_sources_chunk(s: np.ndarray, first: int, count: int, seed: int, layout: "tuple | None" = None) -> np.ndarray:
    """(count, n) int32 source positions of permutations ``first .. first + count - 1`` of the null seeded by ``seed``: row p permutes the positions of every stratum of ``s`` among themselves. ``layout`` is ``strata_layout(s)`` when the caller already has it."""
    order, starts, lens = layout if layout is not None else strata_layout(s)
    return np.asarray(_permutation_sources_njit(order, starts, lens, int(count), int(seed) + int(first)))


def permutation_sources(s: np.ndarray, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """(n_perm, n) int32 source positions: row p permutes the positions of every stratum of ``s`` among themselves."""
    return permutation_sources_chunk(s, 0, n_perm, permutation_seed(rng))


__all__ = ["permutation_seed", "strata_layout", "permutation_sources_chunk", "permutation_sources", "stratified_cmi", "stratified_cmi_null"]
