"""Key-building ``@njit`` helpers shared by ``info_theory`` and ``_numba_utils``.

Leaf module (numpy and numba only): ``info_theory`` imports these to build conditional-MI cache keys, so they cannot live in ``_numba_utils``
(which imports ``info_theory`` for its evaluation kernels). ``_numba_utils`` re-exports both names.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from numba import njit


@njit(cache=True)
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


@njit(cache=True)
def unpack_and_sort(x: Sequence[int], z: Sequence[int]) -> np.ndarray:
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
