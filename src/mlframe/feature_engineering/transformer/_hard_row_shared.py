"""Shared top-K-within-subset selector for the ``*_hard_row``/``*_cbhr`` transformer family:
independently duplicated across those modules, consolidated here so a fix can't silently drift out
of sync across copies.
"""
from __future__ import annotations

import numpy as np


def topk_within_subset(values: np.ndarray, subset_idx: np.ndarray, k: int) -> np.ndarray:
    """Indices (into ``values``) of the top-``k`` largest values restricted to ``subset_idx``, deterministically
    tiebroken by position; returns empty when the subset is empty."""
    if subset_idx.size == 0:
        return np.array([], dtype=np.int64)
    k_eff = min(k, subset_idx.size)
    sub_values = values[subset_idx]
    # Wave 62 (2026-05-20): lexsort with within-subset index tiebreak so tied residuals (duplicate rows) give
    # deterministic top-K within subset.
    sub_top = np.lexsort((np.arange(len(sub_values)), -sub_values))[:k_eff]
    return np.asarray(subset_idx[sub_top])
