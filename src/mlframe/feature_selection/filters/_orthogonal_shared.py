"""Shared helpers for the ``_orthogonal_*_fe.py`` layer family (adaptive-degree, bootstrap-MI,
cluster-basis, CMIM, diff-basis, JMIM, quadruplet, routing, three-gate-MI, total-correlation,
triplet, ...): small utilities independently duplicated across those modules, consolidated here
so a fix can't silently drift out of sync across copies.
"""
from __future__ import annotations

import numpy as np


def coerce_y_classif(y) -> np.ndarray:
    """Dense int64 class labels for MI estimators.

    Integer dtypes pass straight through. Non-integer y (float / continuous /
    categorical) is DENSIFIED via ``np.unique(return_inverse=...)`` rather than
    truncated with ``.astype(int64)``: plain truncation merges distinct labels
    (1.2 and 1.8 -> 1) and destroys continuous-y signal entirely (every value
    in [0, 1) collapses to 0). The dense-rank mapping preserves every distinct
    value as its own class, which is the contract the MI estimator expects.
    """
    arr = np.asarray(y).ravel()
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64, copy=False)
    _, inv = np.unique(arr, return_inverse=True)
    return inv.astype(np.int64, copy=False)
