"""Shared cross-module ``@njit`` helpers.

INVARIANT
---------
ALL ``@njit`` helpers called from more than one ``filters/*`` submodule
live here. Single-module njit helpers stay in their owning module.

The cross-module rule exists because numba dispatcher objects are bound
to the importing module at compile time. When the same helper is
duplicated across modules, numba compiles two distinct dispatchers under
the same Python name and the cache invalidates on every cross-module
dispatch -- which produces silent recompiles, longer cold starts, and on
some Windows configurations file-lock races against the
``__pycache__`` directory.

Consolidating shared kernels here keeps the dispatcher graph clean and
makes cache invalidation predictable.

The companion meta test
``mlframe/tests/test_meta/test_filters_numba_invariant.py`` greps the
filters submodules for cross-module ``@njit`` imports and fails if a
shared helper drifts into a non-``_numba_utils`` location.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit()
def arr2str(arr) -> str:
    """Collision-safe ``@njit`` stringification of an integer array.

    B7b investigation (post-plan): the analogous helper in
    ``pyutilz.data.numbalib.arr2str`` is **pure Python** (not jitted) by
    design -- per its docstring numba's string ops are O(N^2) and the
    pyutilz version is meant for one-off Python contexts. Our local
    version cannot be replaced because it's called from inside
    ``conditional_mi``'s @njit body. Verdict: B7b not applicable; keep
    local implementation.


    Each element is separated by ``_`` so that distinct multisets always
    produce distinct strings. The pre-B12 implementation collapsed
    ``sorted([1, 11])`` and ``sorted([1, 1, 1])`` to the same ``"111"``
    string, which then aliased entropy-cache slots in ``conditional_mi``.
    Per-scenario collision counts are recorded in
    ``mlframe/feature_selection/_benchmarks/_results/collision_census_pre_refactor.json``.

    Returns ``""`` for an empty input. The leading ``_`` (after the first
    element) is omitted so single-element multisets keep a trivial
    representation.
    """
    out = ""
    for k, el in enumerate(arr):
        if k > 0:
            out += "_"
        out += str(el)
    return out


@njit()
def count_cand_nbins(X, factors_nbins) -> int:
    """Sum the bin count across the factors named in candidate ``X``.

    Used by ``screen_predictors`` to gate the confirmation step (when the
    conditioning set has more bins than ``MAX_CONFIRMATION_CAND_NBINS``,
    permutation testing is skipped).
    """
    sum_cand_nbins = 0
    for factor in X:
        sum_cand_nbins += factors_nbins[factor]
    return sum_cand_nbins


@njit()
def unpack_and_sort(x, z):
    """Concatenate two integer iterables into a sorted numpy array.

    Used by ``conditional_mi`` cache-key construction: the canonical
    representation of the union ``X ∪ Z`` must be ordering-independent so
    different call paths hit the same cache slot.
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
