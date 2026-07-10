"""Additive-vs-interaction signal decomposition via ``num_leaves=2`` LightGBM ("no-interaction" baseline).

``num_leaves=2`` restricts every tree to a single split (a depth-1 stump); summed across boosting rounds this
degenerates to a GAM-like ADDITIVE model -- each feature's marginal contribution, with NO feature
interactions representable at all. Comparing this "additive baseline"'s CV score against a normal
full-interaction model quantifies how much of the dataset's signal is purely additive vs interaction-driven
-- a cheap gate before investing in expensive interaction feature engineering (polynomial/hermite features,
deep trees): if the additive model already captures most of the score, interaction engineering has low
expected payoff for this dataset.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from mlframe.models.lgbm_defaults import default_lgbm_params


def additive_interaction_diagnostic(
    X: Any,
    y: np.ndarray,
    cv_splits: Any,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    objective: str = "regression",
    full_model_overrides: Optional[Dict[str, Any]] = None,
    additive_model_overrides: Optional[Dict[str, Any]] = None,
) -> dict:
    """Compare a ``num_leaves=2`` additive-only LightGBM against a normal full-interaction LightGBM by CV score.

    Parameters
    ----------
    X, y
        Full feature/target arrays.
    cv_splits
        Iterable of ``(train_idx, test_idx)`` index pairs.
    metric_fn
        ``metric_fn(y_true, y_pred) -> float``, HIGHER is better (e.g. AUC, R^2) -- used to compute the
        additive-vs-full RATIO in a natural [0, 1]-ish scale; for a loss metric, pass ``-metric_fn`` (or wrap
        to negate) so higher-is-better holds.
    objective
        Passed to :func:`mlframe.models.lgbm_defaults.default_lgbm_params`.
    full_model_overrides, additive_model_overrides
        Extra LightGBM params merged into each model's config (``additive_model_overrides`` always has
        ``num_leaves=2`` forced last, so it can't be overridden away from the additive constraint).

    Returns
    -------
    dict
        ``full_model_cv_score``, ``additive_model_cv_score`` (mean CV score per model type),
        ``additive_signal_ratio`` (``additive_score / full_score`` when both are positive; the fraction of
        the full model's score an additive-only model already captures -- close to 1.0 means the dataset's
        signal is mostly additive and interaction engineering has low expected payoff; well below 1.0 means
        real interaction signal exists), ``recommend_interaction_engineering`` (bool: ratio < 0.9).
    """
    import lightgbm as lgb

    is_frame = hasattr(X, "iloc")
    full_params = default_lgbm_params(objective=objective, **(full_model_overrides or {}))
    additive_params = default_lgbm_params(objective=objective, **(additive_model_overrides or {}))
    additive_params["num_leaves"] = 2

    def _cv_score(params: dict) -> float:
        fold_scores = []
        for train_idx, test_idx in cv_splits:
            X_train = X.iloc[train_idx] if is_frame else X[train_idx]
            X_test = X.iloc[test_idx] if is_frame else X[test_idx]
            model = lgb.LGBMRegressor(**params) if objective == "regression" else lgb.LGBMClassifier(**params)
            model.fit(X_train, y[train_idx])
            if hasattr(model, "predict_proba") and objective != "regression":
                pred = model.predict_proba(X_test)[:, 1]
            else:
                pred = model.predict(X_test)
            fold_scores.append(float(metric_fn(y[test_idx], pred)))
        return float(np.mean(fold_scores))

    full_score = _cv_score(full_params)
    additive_score = _cv_score(additive_params)

    # ratio can legitimately be negative (an additive-only model scoring WORSE than a constant predictor is
    # itself strong evidence the signal is interaction-driven, not additive) -- only undefined when the full
    # model itself has no signal to compare against.
    ratio = float(additive_score / full_score) if full_score > 0 else float("nan")

    return {
        "full_model_cv_score": full_score,
        "additive_model_cv_score": additive_score,
        "additive_signal_ratio": ratio,
        "recommend_interaction_engineering": (not np.isnan(ratio)) and ratio < 0.9,
    }


__all__ = ["additive_interaction_diagnostic"]
