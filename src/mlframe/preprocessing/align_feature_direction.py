"""``align_feature_direction``: flip sign of negatively-target-oriented features before pooling.

Source: 4th_santander-customer-transaction-prediction.md -- "I reversed features which had individual AUC
less than .5 - the idea was to get all features sorted similarly against target to help boosting." Boosting
splits are per-feature-threshold and orientation-agnostic in principle, but techniques that POOL features
together (a long-format melt across many independently-modeled columns, a composite mean/sum across a feature
block, a shared-embedding model) implicitly assume a consistent orientation -- a feature negatively correlated
with the target contributes the WRONG sign to a pooled aggregate unless flipped first.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def align_feature_direction(df: pd.DataFrame, y: np.ndarray, columns: Optional[Sequence[str]] = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Flip sign of every column whose univariate AUC against ``y`` is below 0.5.

    Parameters
    ----------
    df
        Feature frame.
    y
        Binary target, same row order as ``df``.
    columns
        Columns to screen; defaults to every numeric column of ``df``.

    Returns
    -------
    tuple
        ``(aligned_df, flip_signs)`` -- ``aligned_df`` is ``df`` (shallow copy) with flipped columns negated;
        ``flip_signs`` is ``{column: +1 or -1}`` (the sign APPLIED, i.e. ``-1`` means that column was
        flipped) -- store this and reapply the SAME signs at inference/test time (never recompute AUC on
        test rows, which would leak the test target).
    """
    cols = list(columns) if columns is not None else list(df.select_dtypes(include=[np.number]).columns)
    y_arr = np.asarray(y)

    # A per-column sklearn.roc_auc_score loop pays heavy per-call overhead (type_of_target validation,
    # label_binarize, array_api_compat wrapping) that's identical/redundant across every column -- measured
    # as the dominant cProfile cost at n_cols=500. AUC has a closed-form rank-based formula (Mann-Whitney U
    # statistic): auc = (sum_of_ranks_among_positives - n_pos*(n_pos+1)/2) / (n_pos*n_neg). Computing ranks
    # for the WHOLE (n_rows, n_cols) matrix in one vectorized np.argsort(axis=0) pass, instead of one
    # sklearn call per column, replaces N heavyweight Python-level calls with a single C-level batch op.
    X = df[cols].to_numpy(dtype=np.float64)
    order = np.argsort(X, axis=0)
    ranks = np.empty_like(order, dtype=np.float64)
    rank_values = np.broadcast_to((np.arange(X.shape[0]) + 1)[:, None], order.shape)
    np.put_along_axis(ranks, order, rank_values, axis=0)  # 1-based ranks; ties broken arbitrarily (matches roc_auc_score's tie handling closely enough for a sign decision)

    is_pos = y_arr == 1
    n_pos = int(is_pos.sum())
    n_neg = len(y_arr) - n_pos
    sum_ranks_pos = ranks[is_pos].sum(axis=0)
    aucs = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    flip_signs: Dict[str, int] = {}
    flipped_cols: List[str] = []
    for col, auc in zip(cols, aucs):
        sign = -1 if auc < 0.5 else 1
        flip_signs[col] = sign
        if sign == -1:
            flipped_cols.append(col)

    out = df.copy(deep=False)
    if flipped_cols:
        out[flipped_cols] = -out[flipped_cols]
    return out, flip_signs


def apply_feature_direction(df: pd.DataFrame, flip_signs: Dict[str, int]) -> pd.DataFrame:
    """Reapply previously-fitted flip signs (e.g. to held-out/test rows) -- never recomputes AUC."""
    out = df.copy(deep=False)
    flipped_cols = [col for col, sign in flip_signs.items() if sign == -1 and col in out.columns]
    if flipped_cols:
        out[flipped_cols] = -out[flipped_cols]
    return out


__all__ = ["align_feature_direction", "apply_feature_direction"]
