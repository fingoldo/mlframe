"""Shared target-encoding helper for the classification-MI FE families.

The FE families that score an engineered column's plug-in classification MI against ``y`` must first turn
``y`` into dense integer class codes. The trap the several families independently fell into was casting a
CONTINUOUS float target with ``astype(int64)`` BEFORE ``np.unique`` - which truncates ``0.7 -> 0`` and
collapses the target to a couple of buckets (MI destroyed), OR densifying a genuinely continuous target into
~n singleton classes (classification MI degenerate, every row its own class). Both destroy the signal.

``encode_y_for_classif_mi`` is the single correct path: integer targets densify directly; a continuous
(float/complex) target with more than a handful of distinct values is quantile-binned FIRST (never
int-truncated), then densified. Mirrors the temporal-agg family's continuous-y guard.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Above this many distinct float values a target is treated as continuous and quantile-binned rather than
# densified as-is (a float target with <=32 levels is already effectively discrete -> densify directly).
_CONTINUOUS_Y_DISTINCT_THRESHOLD = 32
_CONTINUOUS_Y_QCUT_BINS = 10


def encode_y_for_classif_mi(y: np.ndarray) -> np.ndarray:
    """Dense int64 class codes for a classification-MI target.

    Integer/bool ``y`` is densified directly. A continuous (float/complex) ``y`` with more than
    ``_CONTINUOUS_Y_DISTINCT_THRESHOLD`` distinct values is quantile-binned (never int-truncated, which would
    collapse ``[0, 1)`` to one class) then densified. Idempotent on already-dense integer codes.
    """
    arr = np.asarray(y).ravel()
    if np.issubdtype(arr.dtype, np.integer) or arr.dtype == bool:
        a = arr.astype(np.int64, copy=False)
        # Fast path for an already-dense 0..K-1 code vector (the overwhelmingly common case: this helper is
        # called per-pair x per-modulus x per-permutation inside the FE scans). np.unique would re-SORT the
        # whole column every call; an O(n) bincount occupancy check is far cheaper and returns the identical
        # array when the codes are already dense.
        if a.size:
            mn = int(a.min())
            mx = int(a.max())
            if mn == 0 and mx < 65536 and np.count_nonzero(np.bincount(a, minlength=mx + 1)) == mx + 1:
                return a
        _, inv = np.unique(a, return_inverse=True)
        return inv.astype(np.int64, copy=False)
    if arr.dtype.kind in "fc" and int(np.unique(arr).size) > _CONTINUOUS_Y_DISTINCT_THRESHOLD:
        try:
            import pandas as pd

            # qcut(labels=False) returns an ndarray for ndarray input (Series for Series) - np.asarray covers both.
            arr = np.asarray(pd.qcut(arr, q=_CONTINUOUS_Y_QCUT_BINS, labels=False, duplicates="drop"))
        except Exception as e:  # nosec B110 - qcut can fail on degenerate distributions; fall back to plain densify
            logger.debug("pd.qcut continuous-target binning failed, falling back to plain densify: %s", e)
    _, inv = np.unique(arr, return_inverse=True)
    return inv.astype(np.int64, copy=False)
