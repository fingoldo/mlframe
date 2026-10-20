"""Compiled plug-in conditional MI over strata, and its within-stratum permutation null (triplet cross-basis scorer).

The scorer evaluates I(x; y | s) = sum over strata of (m_s / n) * I(x; y within s) for an engineered column and 199 within-stratum permutations of
it. Each evaluation used a Python loop over strata with a scikit-learn ``mutual_info_score`` call (2.4 ms each): 3248 evaluations were 79 s of a
154 s F2 fit. Here one pass over the rows fills an (strata, x levels, y levels) count table and the MI is read off it, in a parallel loop over
permutations. The permutations themselves are drawn by the caller (``permutation_sources``) from the same generator, in the same order, as the
per-stratum ``rng.permutation`` they replace, so the null is the same data.
"""

from __future__ import annotations


import numpy as np


from ._numba_utils import (  # noqa: F401 -- helpers hosted in _numba_utils_cross_module for the cross-module njit invariant
    _cmi_from_counts,
    stratified_cmi,
    stratified_cmi_null,
)


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
