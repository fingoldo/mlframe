"""``imputation_sensitivity_check``: flag imputation choices whose CV score is unstable across folds/time.

Source: 1st_favorita-grocery-sales-forecasting.md -- CPMP's observation that how different teams filled a
missing "onpromotion" flag correlated with their public/private leaderboard gap (a train/test-distribution-
shift risk hiding behind an innocuous-looking fill-value choice). Generalizes the competition-specific LB-gap
framing into a reusable diagnostic: given several candidate imputation choices for a missingness-heavy
feature, refit under each and compare fold-to-fold CV score VARIANCE -- a choice whose CV score swings wildly
across folds/time splits is a red flag for exactly this kind of train/test-shift risk, even before any
holdout leaderboard exists to reveal it.

A fold-to-fold-only check has a blind spot: when ``cv`` shuffles rows (the default) or otherwise mixes eras
within every fold, each fold sees a representative blend of every regime, so a fill choice whose error is
merely CONSISTENT across folds (same average bias each time) reads as "stable" even though it would fail hard
against a genuine held-out future/shifted segment -- exactly the public/private leaderboard-gap failure mode
the source anecdote is about. The optional ``shift_split`` param adds a second, opt-in check against one real
train/holdout split (e.g. "train on the past, evaluate on the future") to catch that blind spot directly.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold


def _split(cv, X: pd.DataFrame):
    return cv.split(X)


def imputation_sensitivity_check(
    estimator: Any,
    X_variants: Dict[str, pd.DataFrame],
    y: np.ndarray,
    scoring: Callable[[np.ndarray, np.ndarray], float],
    cv: Optional[object] = None,
    risk_z_threshold: float = 0.5,
    shift_split: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    shift_risk_z_threshold: float = 0.5,
) -> pd.DataFrame:
    """Compare per-fold CV score stability across several imputation-choice variants of ``X``.

    Parameters
    ----------
    estimator
        Unfitted sklearn-compatible estimator, cloned per fold/variant.
    X_variants
        ``{choice_name: X_with_that_imputation_applied}`` -- same row order/target across all variants, only
        the imputed feature(s) differ.
    y
        ``(n,)`` target, shared across variants.
    scoring
        ``scoring(y_true, y_pred) -> float``, HIGHER is better.
    cv
        sklearn-style CV splitter; defaults to ``KFold(n_splits=5, shuffle=True, random_state=0)``. Pass a
        time-ordered splitter (no shuffle) to match the source's own train/test-shift framing.
    risk_z_threshold
        A variant's ``fold_std`` is flagged ``is_risky=True`` when its z-score (relative to the OTHER
        variants' `fold_std`) exceeds this threshold -- i.e. it is notably less stable than its peers.
    shift_split
        Opt-in. ``(train_idx, holdout_idx)`` row-index arrays into ``X_variants``/``y`` describing ONE real
        distribution-shift holdout (e.g. a chronological "past vs. future" cut, or a covariate-shift split) --
        deliberately NOT a rotating CV fold, since folds that mix regimes can hide exactly this risk. When
        given, each variant is additionally fit once on ``train_idx`` and scored once on ``holdout_idx``; the
        drop from that variant's ``fold_mean`` to its ``shift_score`` (``shift_gap``) is compared across
        variants the same way ``fold_std`` is, adding ``shift_score``, ``shift_gap`` and ``is_shift_risky``
        columns. Omitted by default -- output is then bit-identical to the fold-only check.
    shift_risk_z_threshold
        A variant's ``shift_gap`` is flagged ``is_shift_risky=True`` when its z-score (relative to the OTHER
        variants' ``shift_gap``) exceeds this threshold. Only used when ``shift_split`` is given.

    Returns
    -------
    pd.DataFrame
        One row per variant (indexed by choice name), columns ``fold_mean``, ``fold_std``, ``is_risky`` (plus
        ``shift_score``, ``shift_gap``, ``is_shift_risky`` when ``shift_split`` is given), sorted by
        ``fold_std`` descending (riskiest first).
    """
    cv_splitter = KFold(n_splits=5, shuffle=True, random_state=0) if cv is None else cv

    results: Dict[str, np.ndarray] = {}
    for name, X in X_variants.items():
        fold_scores = []
        for train_idx, test_idx in _split(cv_splitter, X):
            model = clone(estimator)
            model.fit(X.iloc[train_idx], y[train_idx])
            preds = model.predict(X.iloc[test_idx])
            fold_scores.append(scoring(y[test_idx], preds))
        results[name] = np.asarray(fold_scores, dtype=np.float64)

    names = list(results.keys())
    means = np.array([results[n].mean() for n in names])
    stds = np.array([results[n].std(ddof=1) if len(results[n]) > 1 else 0.0 for n in names])

    if len(stds) > 1 and stds.std(ddof=1) > 0:
        z_scores = (stds - stds.mean()) / stds.std(ddof=1)
    else:
        z_scores = np.zeros_like(stds)
    is_risky = z_scores > risk_z_threshold

    columns: Dict[str, np.ndarray] = {"fold_mean": means, "fold_std": stds, "is_risky": is_risky}

    if shift_split is not None:
        shift_train_idx, shift_holdout_idx = shift_split
        shift_scores = np.empty(len(names), dtype=np.float64)
        for i, name in enumerate(names):
            X = X_variants[name]
            model = clone(estimator)
            model.fit(X.iloc[shift_train_idx], y[shift_train_idx])
            preds = model.predict(X.iloc[shift_holdout_idx])
            shift_scores[i] = scoring(y[shift_holdout_idx], preds)

        shift_gap = means - shift_scores
        if len(shift_gap) > 1 and shift_gap.std(ddof=1) > 0:
            shift_z_scores = (shift_gap - shift_gap.mean()) / shift_gap.std(ddof=1)
        else:
            shift_z_scores = np.zeros_like(shift_gap)
        is_shift_risky = shift_z_scores > shift_risk_z_threshold

        columns["shift_score"] = shift_scores
        columns["shift_gap"] = shift_gap
        columns["is_shift_risky"] = is_shift_risky

    out = pd.DataFrame(columns, index=names)
    return out.sort_values("fold_std", ascending=False)


__all__ = ["imputation_sensitivity_check"]
