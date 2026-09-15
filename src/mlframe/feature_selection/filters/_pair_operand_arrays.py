"""Operand arrays for every unordered pair of a variable list, built without the C(k, 2) index arrays."""

from __future__ import annotations

import numpy as np

# Pairs converted to Python ints per chunk when a per-pair dict is filled from the operand arrays.
_DICT_FILL_CHUNK = 1_000_000


def pair_operand_arrays(k_vars: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(pair_a, pair_b)`` for every ``i < j`` over ``k_vars``, in ``itertools.combinations`` order.

    Equal to ``k_vars[np.triu_indices(k, 1)]`` for both halves, but written row by row into the two outputs, so the two C(k, 2)
    int64 index arrays ``np.triu_indices`` returns are never allocated (at k=5000 those alone are 200 MB).
    """
    k_vars = np.asarray(k_vars, dtype=np.int64)
    k = int(k_vars.shape[0])
    m = k * (k - 1) // 2
    pa = np.empty(m, dtype=np.int64)
    pb = np.empty(m, dtype=np.int64)
    pos = 0
    for i in range(k - 1):
        width = k - 1 - i
        pa[pos : pos + width] = k_vars[i]
        pb[pos : pos + width] = k_vars[i + 1 :]
        pos += width
    return pa, pb


def fill_pair_bias(target: dict, pair_a: np.ndarray, pair_b: np.ndarray, bias: np.ndarray) -> None:
    """Set ``target[(min, max)] = float(bias[p])`` for every pair ``p``, converting at most ``_DICT_FILL_CHUNK`` pairs to Python at once."""
    n = int(pair_a.shape[0])
    for start in range(0, n, _DICT_FILL_CHUNK):
        stop = min(n, start + _DICT_FILL_CHUNK)
        for a, b, v in zip(pair_a[start:stop].tolist(), pair_b[start:stop].tolist(), bias[start:stop].tolist()):
            target[(a, b) if a <= b else (b, a)] = float(v)
