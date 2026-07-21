"""``align_feature_direction``: flip sign of negatively-target-oriented features before pooling.

Source: 4th_santander-customer-transaction-prediction.md -- "I reversed features which had individual AUC
less than .5 - the idea was to get all features sorted similarly against target to help boosting." Boosting
splits are per-feature-threshold and orientation-agnostic in principle, but techniques that POOL features
together (a long-format melt across many independently-modeled columns, a composite mean/sum across a feature
block, a shared-embedding model) implicitly assume a consistent orientation -- a feature negatively correlated
with the target contributes the WRONG sign to a pooled aggregate unless flipped first.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

FlipSpec = Union[int, Tuple[str, float, int]]  # int: plain sign flip; ("fold", center, sign): non-monotonic fold transform


def batch_univariate_auc(X: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
    """Per-column univariate AUC against a binary target, for the WHOLE ``(n_rows, n_cols)`` matrix at once.

    A per-column ``sklearn.roc_auc_score`` loop pays heavy per-call overhead (``type_of_target`` validation,
    ``label_binarize``, ``array_api_compat`` wrapping) that's identical/redundant across every column --
    measured as the dominant cProfile cost at n_cols=500. AUC has a closed-form rank-based formula
    (Mann-Whitney U statistic): ``auc = (sum_of_ranks_among_positives - n_pos*(n_pos+1)/2) / (n_pos*n_neg)``.
    Computing ranks for the whole matrix in one vectorized ``np.argsort(axis=0)`` pass, instead of one
    sklearn call per column, replaces N heavyweight Python-level calls with a single C-level batch op.
    """
    is_pos = y_arr == 1
    n_pos = int(is_pos.sum())
    n_neg = len(y_arr) - n_pos
    if n_pos == 0 or n_neg == 0:
        # A single-class y makes AUC undefined (n_pos*n_neg == 0). Pre-fix this divided silently,
        # producing inf/nan AUCs (RuntimeWarning, not raised) that then compare False against 0.5
        # everywhere, so callers silently treated every column as "no flip needed" instead of
        # erroring.
        raise ValueError(f"batch_univariate_auc: y_arr is single-class (n_pos={n_pos}, n_neg={n_neg}); AUC is undefined. Filter to a genuinely two-class y before calling.")

    order = np.argsort(X, axis=0)
    ranks = np.empty_like(order, dtype=np.float64)
    rank_values = np.broadcast_to((np.arange(X.shape[0]) + 1)[:, None], order.shape)
    np.put_along_axis(ranks, order, rank_values, axis=0)  # 1-based ranks; ties broken arbitrarily (matches roc_auc_score's tie handling closely enough for a sign/threshold decision)

    sum_ranks_pos = ranks[is_pos].sum(axis=0)
    return np.asarray((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _best_fold_point(x: np.ndarray, y_arr: np.ndarray, n_candidates: int) -> Tuple[float, float]:
    """Search candidate fold centers (quantiles of ``x``) for the one whose ``|x - c|`` best separates ``y_arr``.

    A U/inverted-U relationship (e.g. both tails of ``x`` push the same direction, the middle the other way) has
    linear AUC near 0.5 -- a monotonic sign flip can never recover it, because no single sign orients the whole
    column consistently. Folding around the right center turns it into a monotonic "distance from center" signal
    that DOES have a strong linear-AUC-detectable direction, and is stable to reapply on unseen rows (unlike
    e.g. a spline fit) since only a single scalar center needs to be stored.
    """
    candidates = np.quantile(x, np.linspace(0.05, 0.95, n_candidates))
    best_auc = 0.5
    best_gap = 0.0
    best_c = float(np.median(x))
    for c in candidates:
        folded = np.abs(x - c)
        auc = float(batch_univariate_auc(folded[:, None], y_arr)[0])
        gap = abs(auc - 0.5)
        if gap > best_gap:
            best_gap = gap
            best_auc = auc
            best_c = float(c)
    return best_c, best_auc


def batch_mutual_information(X: np.ndarray, y_arr: np.ndarray, random_state: int = 0) -> np.ndarray:
    """Per-column mutual information against a binary target, for the whole ``(n_rows, n_cols)`` matrix at once.

    Unlike ``batch_univariate_auc`` (a monotonic-relationship detector -- AUC is a rank statistic, blind to
    any relationship that isn't consistently increasing or decreasing), MI captures ANY statistical dependence,
    including non-monotonic ones (U-shaped, inverted-U, multi-modal) where AUC sits near 0.5 despite the column
    carrying real signal.
    """
    from sklearn.feature_selection import mutual_info_classif

    return np.asarray(mutual_info_classif(X, y_arr, discrete_features=False, random_state=random_state))


def align_feature_direction(
    df: pd.DataFrame,
    y: np.ndarray,
    columns: Optional[Sequence[str]] = None,
    use_mutual_information: bool = False,
    mi_near_chance_gap: float = 0.05,
    mi_relevance_threshold: float = 0.01,
    fold_n_candidates: int = 25,
    random_state: int = 0,
    nonlinear_report: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, FlipSpec]]:
    """Flip sign of every column whose univariate AUC against ``y`` is below 0.5.

    Parameters
    ----------
    df
        Feature frame.
    y
        Binary target, same row order as ``df``.
    columns
        Columns to screen; defaults to every numeric column of ``df``.
    use_mutual_information
        Opt-in (default ``False``, behavior bit-identical to the plain AUC-sign path when omitted). When
        ``True``, columns whose linear AUC is near-chance (``|auc - 0.5| < mi_near_chance_gap`` -- exactly
        the columns a plain sign flip would treat as noise) are re-screened with ``batch_mutual_information``.
        A column with MI ``>= mi_relevance_threshold`` is genuinely informative despite near-chance linear AUC
        -- almost always a non-monotonic relationship a sign flip cannot fix -- so instead of a sign flip it's
        replaced by ``sign * |x - center|`` for a ``center`` chosen (via ``_best_fold_point``) to maximize that
        folded feature's own linear-AUC separation. The fold is only applied if it beats the original AUC gap.
    mi_near_chance_gap
        Only columns within this AUC distance of 0.5 are eligible for MI re-screening (avoids wasted MI calls
        on already-well-oriented linear columns).
    mi_relevance_threshold
        Minimum MI (nats) for a near-chance column to be treated as a genuine non-monotonic relationship.
    fold_n_candidates
        Number of quantile-spaced candidate fold centers tried per eligible column.
    random_state
        Seed passed to ``sklearn.feature_selection.mutual_info_classif`` (uses a randomized kNN estimator).
    nonlinear_report
        Optional out-param dict; when supplied, columns folded via the MI path get an entry
        ``{"linear_auc": ..., "mutual_information": ..., "fold_center": ..., "fold_auc": ...}``.

    Returns
    -------
    tuple
        ``(aligned_df, flip_signs)`` -- ``aligned_df`` is ``df`` (shallow copy) with flipped/folded columns
        transformed; ``flip_signs`` maps each column to either an ``int`` (``+1``/``-1``, the sign applied) or,
        only when ``use_mutual_information=True`` folded a column, a ``("fold", center, sign)`` tuple -- store
        this and reapply via ``apply_feature_direction`` at inference/test time (never recompute on test rows,
        which would leak the test target).
    """
    cols = list(columns) if columns is not None else list(df.select_dtypes(include=[np.number]).columns)
    y_arr = np.asarray(y)
    X = df[cols].to_numpy(dtype=np.float64)
    aucs = batch_univariate_auc(X, y_arr)

    flip_signs: Dict[str, FlipSpec] = {}
    flipped_cols: List[str] = []
    fold_cols: Dict[str, Tuple[str, float, int]] = {}

    near_chance_idx: List[int] = []
    if use_mutual_information:
        near_chance_idx = [j for j, auc in enumerate(aucs) if abs(auc - 0.5) < mi_near_chance_gap]

    mis = np.empty(0)
    if near_chance_idx:
        mis = batch_mutual_information(X[:, near_chance_idx], y_arr, random_state=random_state)

    relevant_nonlinear_idx = {j for j, mi in zip(near_chance_idx, mis) if mi >= mi_relevance_threshold}

    for j, (col, auc) in enumerate(zip(cols, aucs)):
        if j in relevant_nonlinear_idx:
            center, fold_auc = _best_fold_point(X[:, j], y_arr, fold_n_candidates)
            if abs(fold_auc - 0.5) > abs(auc - 0.5):
                sign = -1 if fold_auc < 0.5 else 1
                fold_cols[col] = ("fold", center, sign)
                flip_signs[col] = fold_cols[col]
                if nonlinear_report is not None:
                    mi_value = float(mis[near_chance_idx.index(j)])
                    nonlinear_report[col] = {"linear_auc": float(auc), "mutual_information": mi_value, "fold_center": center, "fold_auc": fold_auc}
                continue
        sign = -1 if auc < 0.5 else 1
        flip_signs[col] = sign
        if sign == -1:
            flipped_cols.append(col)

    out = df.copy(deep=False)
    if flipped_cols:
        out[flipped_cols] = -out[flipped_cols]
    for col, (_mode, center, sign) in fold_cols.items():
        out[col] = sign * np.abs(out[col].to_numpy(dtype=np.float64) - center)
    return out, flip_signs


def apply_feature_direction(df: pd.DataFrame, flip_signs: Dict[str, FlipSpec]) -> pd.DataFrame:
    """Reapply previously-fitted flip signs/folds (e.g. to held-out/test rows) -- never recomputes AUC/MI."""
    out = df.copy(deep=False)
    flipped_cols = [col for col, spec in flip_signs.items() if spec == -1 and col in out.columns]
    if flipped_cols:
        out[flipped_cols] = -out[flipped_cols]
    for col, spec in flip_signs.items():
        if col not in out.columns or not isinstance(spec, tuple):
            continue
        _mode, center, sign = spec
        out[col] = sign * np.abs(out[col].to_numpy(dtype=np.float64) - center)
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


__all__ = ["align_feature_direction", "apply_feature_direction", "check_feature_direction_stability", "batch_mutual_information"]
