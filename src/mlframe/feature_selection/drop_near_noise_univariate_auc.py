"""``drop_near_noise_univariate_auc``: cheap univariate-AUC prescreen before MRMR/DCD.

Source: 4th_santander-customer-transaction-prediction.md -- "I removed some vars from train which predictions
by long model had AUC near .5 (before grouping)." A feature whose OWN univariate AUC sits at chance carries
essentially no linear/monotone signal about the target in isolation -- dropping it before the expensive
MRMR/DCD redundancy-aware search is a cheap first-pass filter for independent-feature datasets (won't catch a
feature that's only informative in COMBINATION with others, which is exactly what MRMR itself is for; this
is a pre-filter, not a replacement).

Reuses ``preprocessing.align_feature_direction.batch_univariate_auc`` (the same vectorized rank-based AUC
computation added for the feature-direction-alignment entry) rather than reimplementing per-column AUC.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from mlframe.preprocessing.align_feature_direction import batch_univariate_auc


def drop_near_noise_univariate_auc(df: pd.DataFrame, y: np.ndarray, columns: Optional[Sequence[str]] = None, tolerance: float = 0.02) -> List[str]:
    """Return column names whose univariate AUC against ``y`` falls within ``tolerance`` of chance (0.5).

    Parameters
    ----------
    df
        Feature frame.
    y
        Binary target, same row order as ``df``.
    columns
        Columns to screen; defaults to every numeric column of ``df``.
    tolerance
        A column is flagged when ``abs(auc - 0.5) <= tolerance``.

    Returns
    -------
    list of str
        Column names to consider dropping as near-noise before the full selection pipeline.
    """
    cols = list(columns) if columns is not None else list(df.select_dtypes(include=[np.number]).columns)
    y_arr = np.asarray(y)
    X = df[cols].to_numpy(dtype=np.float64)
    aucs = batch_univariate_auc(X, y_arr)
    return [col for col, auc in zip(cols, aucs) if abs(auc - 0.5) <= tolerance]


__all__ = ["drop_near_noise_univariate_auc"]
