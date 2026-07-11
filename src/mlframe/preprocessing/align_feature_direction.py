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


def batch_univariate_auc(X: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
    """Per-column univariate AUC against a binary target, for the WHOLE ``(n_rows, n_cols)`` matrix at once.

    A per-column ``sklearn.roc_auc_score`` loop pays heavy per-call overhead (``type_of_target`` validation,
    ``label_binarize``, ``array_api_compat`` wrapping) that's identical/redundant across every column --
    measured as the dominant cProfile cost at n_cols=500. AUC has a closed-form rank-based formula
    (Mann-Whitney U statistic): ``auc = (sum_of_ranks_among_positives - n_pos*(n_pos+1)/2) / (n_pos*n_neg)``.
    Computing ranks for the whole matrix in one vectorized ``np.argsort(axis=0)`` pass, instead of one
    sklearn call per column, replaces N heavyweight Python-level calls with a single C-level batch op.
    """
    order = np.argsort(X, axis=0)
    ranks = np.empty_like(order, dtype=np.float64)
    rank_values = np.broadcast_to((np.arange(X.shape[0]) + 1)[:, None], order.shape)
    np.put_along_axis(ranks, order, rank_values, axis=0)  # 1-based ranks; ties broken arbitrarily (matches roc_auc_score's tie handling closely enough for a sign/threshold decision)

    is_pos = y_arr == 1
    n_pos = int(is_pos.sum())
    n_neg = len(y_arr) - n_pos
    sum_ranks_pos = ranks[is_pos].sum(axis=0)
    return np.asarray((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


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
    X = df[cols].to_numpy(dtype=np.float64)
    aucs = batch_univariate_auc(X, y_arr)

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


def check_feature_direction_stability(
    df: pd.DataFrame,
    y: np.ndarray,
    columns: Optional[Sequence[str]] = None,
    n_folds: int = 5,
    seed: int = 0,
) -> Dict[str, Dict[str, object]]:
    """Opt-in: verify each column's AUC-direction sign is stable across K stratified folds.

    ``align_feature_direction`` computes the sign from ONE full-data AUC estimate. For a feature whose true
    AUC is close to 0.5 (near-chance), that single estimate is itself noisy -- a different sample could just
    as easily have produced an AUC on the other side of 0.5, flipping the sign the wrong way. Such a feature's
    "alignment" is not trustworthy: downstream pooling would be flipping a coin, not correcting a real bias.

    This re-estimates the per-column sign on each of ``n_folds`` stratified folds (each large enough to be a
    meaningful resample, small enough that a genuinely strong direction stays stable) and flags columns whose
    fold-level sign disagrees with the majority sign in ANY fold.

    Returns
    -------
    dict
        ``{column: {"full_sign": int, "fold_signs": List[int], "n_sign_flips": int, "stable": bool,
        "mean_fold_auc": float}}``. ``stable=False`` means at least one fold's AUC crossed 0.5 relative to the
        majority direction -- treat that column's flip in ``align_feature_direction``'s output as unreliable.
    """
    from sklearn.model_selection import StratifiedKFold

    cols = list(columns) if columns is not None else list(df.select_dtypes(include=[np.number]).columns)
    y_arr = np.asarray(y)
    X = df[cols].to_numpy(dtype=np.float64)

    full_aucs = batch_univariate_auc(X, y_arr)
    full_signs = np.where(full_aucs < 0.5, -1, 1)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_sign_matrix = np.empty((n_folds, len(cols)), dtype=np.int64)
    fold_auc_matrix = np.empty((n_folds, len(cols)), dtype=np.float64)
    for fold_idx, (_, fold_rows) in enumerate(cv.split(X, y_arr)):
        fold_aucs = batch_univariate_auc(X[fold_rows], y_arr[fold_rows])
        fold_auc_matrix[fold_idx] = fold_aucs
        fold_sign_matrix[fold_idx] = np.where(fold_aucs < 0.5, -1, 1)

    result: Dict[str, Dict[str, object]] = {}
    for j, col in enumerate(cols):
        fold_signs = fold_sign_matrix[:, j].tolist()
        n_flips = int(np.sum(fold_sign_matrix[:, j] != full_signs[j]))
        result[col] = {
            "full_sign": int(full_signs[j]),
            "fold_signs": fold_signs,
            "n_sign_flips": n_flips,
            "stable": n_flips == 0,
            "mean_fold_auc": float(fold_auc_matrix[:, j].mean()),
        }
    return result


__all__ = ["align_feature_direction", "apply_feature_direction", "check_feature_direction_stability"]
