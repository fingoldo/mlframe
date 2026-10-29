"""Stages of ``honest_null_columns_bycol``."""

from __future__ import annotations

# --- imports (managed) ---
import numpy as np
from numba import njit
# --- end imports ---


@njit(cache=True)
def _bycol_perm_label_stats(n_perm_rows, y_codes_perm, m, eval_rows, n, y_raw_perm):
    """Per-permutation target-code counts and the same-labels test for ``honest_null_columns_bycol``."""
    ny_perm = np.zeros(n_perm_rows, dtype=np.int64)
    ky_eff_perm = np.zeros(n_perm_rows, dtype=np.float64)  # Intentional float64: statistic input
    for b in range(n_perm_rows):
        yc_row = y_codes_perm[b]
        top = 0
        for t in range(m):
            v = yc_row[eval_rows[t]]
            if v + 1 > top:
                top = v + 1
        present = np.zeros(top, dtype=np.bool_)
        for t in range(m):
            present[yc_row[eval_rows[t]]] = True
        occupied = 0.0
        for c in range(top):
            if present[c]:
                occupied += 1.0
        ny_perm[b] = top
        ky_eff_perm[b] = occupied
    # The sorted-order joint below needs the evaluation rows to be every row, in order, and the tree's labels to be the screen's codes.
    all_rows = m == n
    if all_rows:
        for t in range(m):
            if eval_rows[t] != t:
                all_rows = False
                break
    same_labels = all_rows
    if same_labels:
        for b in range(n_perm_rows):
            for i in range(n):
                if y_raw_perm[b, i] != y_codes_perm[b, i]:
                    same_labels = False
                    break
            if not same_labels:
                break
    return ky_eff_perm, ny_perm, same_labels
