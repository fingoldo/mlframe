"""Estimator-type-aware cross-fold importance aggregation for the RFECV wrapper.

The legacy cross-fold aggregator (``get_actual_features_ranking`` in
``_helpers_importance.py``) builds a feature x run table and votes (Borda / mean /
...). That naive mean+vote has two known failure modes:

  * TREE / GBM gain importance is noisy fold-to-fold; a feature with a high raw
    mean but huge cross-fold variance is less trustworthy than a steadily-modest
    feature, yet the mean ranks it higher.
  * LINEAR ``coef_`` is already abs'd before voting, so a feature whose sign
    FLIPS across folds (positive in 3, negative in 2 -> genuinely unstable) is
    indistinguishable from a consistently-signed one of equal magnitude.

This module adds a family-aware dispatcher (``importance_agg="dispatched"``):

  * tree   : per-feature mean across folds DOWN-WEIGHTED by the cross-fold
             coefficient of variation -> ``mean / (1 + cv)`` (cv = std/|mean|).
             High fold-to-fold variance -> ranked lower than its raw mean.
  * linear : SIGN-HARMONY. Average the SIGNED coef across folds then take the
             magnitude, multiplied by the sign-agreement fraction
             (max(frac_pos, frac_neg)). A sign-flipping feature is demoted.
  * kernel : no native importance -> defer to the legacy vote (permutation path
             already produced non-negative importances; nothing family-specific
             to add).

The legacy path stays the default until benched-and-won (REJECTED != DELETED):
``importance_agg="legacy"`` keeps the historical vote.
"""
from __future__ import annotations

import logging
from typing import Union

import numpy as np
import pandas as pd

from ._enums import VotesAggregation

logger = logging.getLogger(__name__)

# Estimator class-name fragments mapped to family. Checked against type(est).__name__
# (after unwrapping Pipelines / TransformedTargetRegressor). Order: linear first so a
# "LogisticRegression" doesn't match a stray "Regress" tree token, etc.
_LINEAR_TOKENS = (
    "LinearRegression", "LogisticRegression", "Ridge", "Lasso", "ElasticNet",
    "SGDClassifier", "SGDRegressor", "LinearSVC", "LinearSVR", "Perceptron",
    "PassiveAggressive", "ARDRegression", "BayesianRidge", "Lars", "OrthogonalMatchingPursuit",
)
_TREE_TOKENS = (
    "RandomForest", "ExtraTrees", "GradientBoosting", "DecisionTree", "HistGradientBoosting",
    "XGB", "LGBM", "LightGBM", "CatBoost", "Bagging", "AdaBoost",
)


def detect_estimator_family(estimator) -> str:
    """Return 'tree' / 'linear' / 'kernel' for an estimator.

    Unwraps Pipeline / TransformedTargetRegressor / wrappers, then matches the
    inner class name against known tokens. Falls back to attribute sniffing
    (coef_ -> linear, feature_importances_ -> tree) and finally 'kernel' for
    SVMs / kNN / anything without a native importance attribute.
    """
    m = estimator
    for _ in range(8):
        inner = None
        for attr in ("_final_estimator", "regressor_", "best_estimator_", "base_estimator"):
            cand = getattr(m, attr, None)
            if cand is not None and cand is not m:
                inner = cand
                break
        if inner is None:
            break
        m = inner
    name = type(m).__name__
    if any(tok in name for tok in _LINEAR_TOKENS):
        return "linear"
    if any(tok in name for tok in _TREE_TOKENS):
        return "tree"
    # Attribute sniff: a fitted estimator exposes coef_ / feature_importances_.
    if hasattr(m, "feature_importances_"):
        return "tree"
    if hasattr(m, "coef_"):
        return "linear"
    # SVC / SVR with a non-linear kernel, KNeighbors*, GaussianProcess*, ...
    return "kernel"


def get_signed_linear_coef(
    model,
    current_features: list,
    train_data=None,
    multiclass_coef_aggregation: str = "max",
    coef_scale_source: str = "train",
) -> Union[dict, None]:
    """Extract SIGNED, scale-corrected linear coef for sign-harmony aggregation.

    Mirrors the coef_ branch of ``get_feature_importances`` (multiclass collapse +
    train-std scaling) but PRESERVES the sign, so the cross-fold aggregator can
    detect sign flips. Returns dict[feature -> signed_coef] or None if the model
    has no usable ``coef_`` (e.g. a tree slipped through family detection).

    Multiclass collapse keeps the sign of the dominant-magnitude class
    ('max' -> coef of the argmax-|coef| class; 'sum' -> signed sum across classes).
    """
    from operator import attrgetter

    m = model
    for _ in range(8):
        if hasattr(m, "_final_estimator") and m._final_estimator is not m:
            m = m._final_estimator
            continue
        if hasattr(m, "regressor_") and getattr(m, "regressor_") is not m:
            m = m.regressor_
            continue
        break
    coef = getattr(m, "coef_", None)
    if coef is None:
        return None
    coef = np.asarray(coef, dtype=float)
    if coef.ndim > 1:
        if multiclass_coef_aggregation == "max":
            # Per feature, take the coef of the class with the largest |coef| (sign kept).
            idx = np.argmax(np.abs(coef), axis=0)
            coef = coef[idx, np.arange(coef.shape[1])]
        else:
            coef = coef.sum(axis=0)
    coef = np.ravel(coef)
    # Scale correction with train stds (sign-preserving multiply by a positive std).
    if coef_scale_source != "none":
        src = train_data
        if src is not None:
            try:
                arr = src.values if hasattr(src, "values") else np.asarray(src)
                stds = np.nanstd(arr, axis=0)
                stds = np.where(stds > 1e-12, stds, 1.0)
                if len(stds) == len(coef):
                    coef = coef * stds
            except (TypeError, ValueError):
                pass
    if len(coef) != len(current_features):
        return None
    return {feat: float(c) for feat, c in zip(current_features, coef)}


def _table_from_runs(feature_importances: dict) -> pd.DataFrame:
    """Build a feature(rows) x run(cols) DataFrame from the per-run dicts."""
    return pd.DataFrame(feature_importances)


def aggregate_tree(feature_importances: dict, k_cv: float = 1.0, eps: float = 1e-12) -> dict:
    """Variance-down-weighted mean of tree/GBM gain importances across folds.

    score = mean / (1 + k_cv * cv), cv = std / (|mean| + eps).

    A feature whose gain is steady across folds keeps (almost) its full mean; one
    whose gain swings wildly fold-to-fold (cv high) is discounted toward 0. Single
    -fold features (1 run) get cv=0 -> raw mean (no information to penalise).
    """
    table = _table_from_runs(feature_importances)
    if table.empty:
        return {}
    means = table.mean(axis=1, skipna=True)
    # ddof=0 so a 1-run feature yields std 0 rather than NaN.
    stds = table.std(axis=1, skipna=True, ddof=0).fillna(0.0)
    cv = stds / (means.abs() + eps)
    scores = means / (1.0 + k_cv * cv)
    return {feat: float(scores[feat]) for feat in table.index}


def aggregate_linear(signed_importances: dict, eps: float = 1e-12) -> dict:
    """Sign-harmony aggregation of SIGNED linear coef across folds.

    score = |mean(signed_coef)| * sign_agreement,
    sign_agreement = max(frac_positive, frac_negative) over the runs where the
    coef is non-zero (zeros are sign-neutral and excluded from the agreement
    fraction but still pull the signed mean toward 0).

    A feature positive in 3 folds and negative in 2 has agreement 0.6 and a
    signed mean near 0 -> demoted hard vs a consistently-positive feature
    (agreement 1.0, full magnitude).
    """
    table = _table_from_runs(signed_importances)
    if table.empty:
        return {}
    # Vectorised over features: the per-row ``table.loc[feat]`` lookup built a fresh Series per feature
    # (the RFECV linear-aggregation hotspot at p~300). numpy column reductions are bit-identical here:
    # mean over finite entries, sign-agreement over |coef|>eps entries, both order-independent.
    M = table.to_numpy(dtype=float)
    finite = np.isfinite(M)
    cnt_finite = finite.sum(axis=1)
    M_fin = np.where(finite, M, 0.0)
    sum_fin = M_fin.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_signed = sum_fin / cnt_finite
    pos = (M > eps) & finite
    neg = (M < -eps) & finite
    n_pos = pos.sum(axis=1)
    n_nz = n_pos + neg.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_pos = n_pos / n_nz
    agreement = np.maximum(frac_pos, 1.0 - frac_pos)
    agreement = np.where(n_nz == 0, 1.0, agreement)  # all-zero coef row: no disharmony
    scores = np.abs(mean_signed) * agreement
    scores = np.where(cnt_finite == 0, 0.0, scores)  # all-non-finite row -> 0.0
    return dict(zip(table.index.tolist(), scores.tolist()))


def aggregate_stability(feature_importances: dict, cut_k: int, eps: float = 1e-12) -> dict:
    """Stability-aware elimination score: mean importance * fold-selection-frequency.

    For each run (CV fold) the features are ranked by that fold's importance; a feature's
    ``fold_selection_frequency`` is the fraction of runs in which it lands in the top ``cut_k``
    (i.e. would survive an elimination keeping ``cut_k`` features in that fold alone). The
    elimination score is ``mean_importance * frequency``.

    This protects a steadily-mid-rank feature (in the top-k of every fold) from being evicted
    in favour of a high-mean-but-high-variance feature that spiked in one fold (top-k in 1 fold
    of 5 -> frequency 0.2 -> score cut by 0.2). Differs from importance_agg='dispatched' tree
    discounting, which penalises VALUE coefficient-of-variation; this penalises RANK volatility
    around the elimination cut, a discrete "did it make the cut" signal rather than a value CV.

    Single-run features get frequency 1.0 (in-or-out of the only fold's top-k decides 1.0 vs 0.0).
    """
    table = _table_from_runs(feature_importances)
    if table.empty:
        return {}
    means = table.mean(axis=1, skipna=True).fillna(0.0)
    k = max(1, int(cut_k))
    n_runs = table.shape[1]
    # Per-run survival indicator: rank each column descending, feature in top-k -> 1.
    # rank(ascending=False) gives 1=highest; <= k means it survives the cut in that fold.
    # NaN entries (feature absent in that run) never count as surviving.
    ranks = table.rank(axis=0, ascending=False, method="min", na_option="bottom")
    survived = (ranks <= k) & table.notna()
    freq = survived.sum(axis=1) / float(max(1, n_runs))
    scores = means * freq
    # Vectorised dict build: per-feature Series.__getitem__ in a comprehension dominated the
    # profile (100k getitem calls / ~0.7s at p=500x40). zip over the index + numpy values is ~5x cheaper.
    return dict(zip(scores.index.tolist(), scores.to_numpy(dtype=float).tolist()))


def aggregate_importances_dispatched(
    feature_importances: dict,
    family: str,
    votes_aggregation_method: VotesAggregation,
    *,
    signed_importances: Union[dict, None] = None,
    k_cv: float = 1.0,
    fi_missing_policy: str = "worst",
    run_weights: Union[dict, None] = None,
) -> list:
    """Estimator-type-aware cross-fold importance ranking.

    Returns a list of feature keys ordered best (most important) first, matching
    the contract of ``get_actual_features_ranking``.

    family:
        'tree'   -> variance-down-weighted mean (aggregate_tree).
        'linear' -> sign-harmony on signed_importances (aggregate_linear); if
                    signed_importances is missing/empty, falls back to the legacy
                    vote (we cannot recover sign from abs'd values).
        'kernel' / other -> legacy vote.
    """
    from ._helpers_importance import get_actual_features_ranking

    if family == "tree" and feature_importances:
        scores = aggregate_tree(feature_importances, k_cv=k_cv)
    elif family == "linear" and signed_importances:
        scores = aggregate_linear(signed_importances)
    else:
        return get_actual_features_ranking(
            feature_importances=feature_importances,
            votes_aggregation_method=votes_aggregation_method,
            fi_missing_policy=fi_missing_policy,
            run_weights=run_weights,
        )
    if not scores:
        return []
    # Deterministic order: descending score, then lexicographic on key name.
    return sorted(
        scores.keys(),
        key=lambda k: (-(scores[k] if np.isfinite(scores[k]) else -np.inf), str(k)),
    )
