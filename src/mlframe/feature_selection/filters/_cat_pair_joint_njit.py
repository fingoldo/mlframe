"""Joint plug-in MI of many categorical column pairs with one target, one parallel pass.

For each pair ``(i, j)`` the two dense code columns are combined into one key (``code_i * radix_j + code_j``), re-numbered by first appearance in one pass and scored with the same njit plug-in MI core
the per-pair path calls (``_plug_in_mi_njit``), so every value is bit-equal to ``_plug_in_mi(joint_codes, y)``. The pairs are independent, hence one ``prange`` over them instead of a Python loop
with a kernel launch, an array allocation and a dictionary row per pair.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ._adaptive_nbins import _plug_in_mi_njit
from ._mi_greedy_cmi_fe_binning import _factorize_dense_njit


@njit(parallel=True, cache=True)
def pair_joint_mi(codes: np.ndarray, radix: np.ndarray, first: np.ndarray, second: np.ndarray, y_b: np.ndarray, k_y: int) -> np.ndarray:
    """``out[p] = MI(joint(codes[first[p]], codes[second[p]]); y_b)`` for every pair ``p``; ``codes`` is ``(columns, rows)`` int64, ``radix[c] = max(codes[c]) + 1``."""
    n_pairs = first.shape[0]
    out = np.zeros(n_pairs, dtype=np.float64)  # Intentional float64: MI statistic vector, bit-equal to the per-pair float(...) of _plug_in_mi
    for p in prange(n_pairs):
        key = codes[first[p]] * radix[second[p]] + codes[second[p]]
        joint_codes, n_cells = _factorize_dense_njit(key)
        out[p] = _plug_in_mi_njit(joint_codes, y_b, n_cells, k_y, False)
    return out
