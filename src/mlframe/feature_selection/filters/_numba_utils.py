"""Shared cross-module ``@njit`` helpers.

INVARIANT: every ``@njit`` helper called from more than one ``filters/*`` submodule lives here; single-module njit helpers stay in their owning module.

Rationale: numba dispatcher objects bind to the importing module at compile time. Duplicating a helper across modules compiles two distinct dispatchers
under the same Python name, invalidates the cache on every cross-module dispatch, causes silent recompiles, longer cold starts, and on some Windows
configurations file-lock races against ``__pycache__``. Consolidating shared kernels here keeps the dispatcher graph clean and cache invalidation
predictable. Enforced by ``mlframe/tests/test_meta/test_filters_numba_invariant.py``.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit()
def arr2str(arr) -> str:
    """Collision-safe ``@njit`` stringification of an integer array. Elements separated by ``_`` so distinct multisets always produce distinct strings; the
    naive concatenation collapsed ``sorted([1, 11])`` and ``sorted([1, 1, 1])`` to the same ``"111"`` and aliased entropy-cache slots in ``conditional_mi``.
    Returns ``""`` for empty input. The pyutilz analogue (``pyutilz.data.numbalib.arr2str``) is pure-Python by design (numba string ops are O(N^2)) and
    cannot replace this one because it's called from inside ``conditional_mi``'s @njit body.
    """
    out = ""
    for k, el in enumerate(arr):
        if k > 0:
            out += "_"
        out += str(el)
    return out


@njit()
def count_cand_nbins(X, factors_nbins) -> int:
    """Sum the bin count across the factors named in candidate ``X``. Used by ``screen_predictors`` to gate the confirmation step (permutation testing is
    skipped when the conditioning set has more bins than ``MAX_CONFIRMATION_CAND_NBINS``)."""
    sum_cand_nbins = 0
    for factor in X:
        sum_cand_nbins += factors_nbins[factor]
    return sum_cand_nbins


@njit()
def unpack_and_sort(x, z):
    """Concatenate two integer iterables into a sorted numpy array. Used by ``conditional_mi`` cache-key construction: the canonical representation of the
    union ``X u Z`` must be ordering-independent so different call paths hit the same cache slot.
    """
    res = np.empty(len(x) + len(z), dtype=np.int64)
    idx = 0
    for el in x:
        res[idx] = el
        idx += 1
    for el in z:
        res[idx] = el
        idx += 1
    return np.sort(res)
