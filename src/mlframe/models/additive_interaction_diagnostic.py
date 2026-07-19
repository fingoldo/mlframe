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

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from mlframe.models.lgbm_defaults import default_lgbm_params


def _drop_column(X: Any, is_frame: bool, feature_names: Sequence[str], col_idx: int) -> Any:
    """Return ``X`` with the ``col_idx``-th feature removed, preserving frame-ness for LightGBM."""
    if is_frame:
        return X.drop(columns=[feature_names[col_idx]])
    return np.delete(np.asarray(X), col_idx, axis=1)


def additive_interaction_diagnostic(
    X: Any,
    y: np.ndarray,
    cv_splits: Any,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    objective: str = "regression",
    full_model_overrides: Optional[Dict[str, Any]] = None,
    additive_model_overrides: Optional[Dict[str, Any]] = None,
    per_feature_report: bool = False,
    per_feature_names: Optional[Sequence[str]] = None,
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
    per_feature_report
        Opt-in (default ``False``, output unchanged when omitted). When ``True``, additionally runs a
        leave-one-feature-out (LOFO) decomposition: for every feature, both the full and additive models are
        retrained without it, giving each feature's total contribution to each model's CV score. A feature's
        ``interaction_contribution`` is its full-model LOFO delta minus its additive-model LOFO delta -- the
        slice of its score contribution that ONLY the full (interaction-capable) model can extract, i.e. how
        much of the additive-vs-full lift traces back to that specific feature. Cost is
        ``2 * n_features * n_folds`` extra LightGBM fits on top of the base diagnostic's ``2 * n_folds``, so
        keep ``n_features`` and CV folds modest when enabling this on large feature sets.
    per_feature_names
        Column names to label the per-feature report with. Required when ``X`` is not a DataFrame/has no
        ``.columns``; ignored (and inferred from ``X.columns``) otherwise. Only consulted when
        ``per_feature_report=True``.

    Returns
    -------
    dict
        ``full_model_cv_score``, ``additive_model_cv_score`` (mean CV score per model type),
        ``additive_signal_ratio`` (``additive_score / full_score`` when both are positive; the fraction of
        the full model's score an additive-only model already captures -- close to 1.0 means the dataset's
        signal is mostly additive and interaction engineering has low expected payoff; well below 1.0 means
        real interaction signal exists), ``recommend_interaction_engineering`` (bool: ratio < 0.9). When
        ``per_feature_report=True``, also ``per_feature_interaction_report``: a list of per-feature dicts
        (``feature``, ``full_model_lofo_delta``, ``additive_model_lofo_delta``, ``interaction_contribution``)
        sorted by ``interaction_contribution`` descending -- an actionable triage table of which features
        drive the model's interaction lift, vs. which are purely additive.
    """
    import lightgbm as lgb

    is_frame = hasattr(X, "iloc")
    full_params = default_lgbm_params(objective=objective, **(full_model_overrides or {}))
    additive_params = default_lgbm_params(objective=objective, **(additive_model_overrides or {}))
    additive_params["num_leaves"] = 2

    def _cv_score(params: dict, X_override: Optional[Any] = None) -> float:
        """Fit an LGBM model per CV fold with ``params`` and return the mean metric across folds."""
        X_source = X if X_override is None else X_override
        fold_scores = []
        for train_idx, test_idx in cv_splits:
            X_train = X_source.iloc[train_idx] if is_frame else X_source[train_idx]
            X_test = X_source.iloc[test_idx] if is_frame else X_source[test_idx]
            model = lgb.LGBMRegressor(**params) if objective == "regression" else lgb.LGBMClassifier(**params)
            model.fit(X_train, y[train_idx])
            if hasattr(model, "predict_proba") and objective != "regression":
                pred = np.asarray(model.predict_proba(X_test))[:, 1]
            else:
                pred = np.asarray(model.predict(X_test))
            fold_scores.append(float(metric_fn(y[test_idx], pred)))
        return float(np.mean(fold_scores))

    full_score = _cv_score(full_params)
    additive_score = _cv_score(additive_params)

    # ratio can legitimately be negative (an additive-only model scoring WORSE than a constant predictor is
    # itself strong evidence the signal is interaction-driven, not additive) -- only undefined when the full
    # model itself has no signal to compare against.
    ratio = float(additive_score / full_score) if full_score > 0 else float("nan")

    result: Dict[str, Any] = {
        "full_model_cv_score": full_score,
        "additive_model_cv_score": additive_score,
        "additive_signal_ratio": ratio,
        "recommend_interaction_engineering": (not np.isnan(ratio)) and ratio < 0.9,
    }

    if per_feature_report:
        if is_frame:
            feature_names: List[str] = list(X.columns)
        elif per_feature_names is not None:
            feature_names = list(per_feature_names)
        else:
            feature_names = [f"feature_{i}" for i in range(np.asarray(X).shape[1])]

        n_features = len(feature_names)
        report: List[Dict[str, Any]] = []
        for col_idx in range(n_features):
            X_loo = _drop_column(X, is_frame, feature_names, col_idx)
            full_loo_score = _cv_score(full_params, X_override=X_loo)
            additive_loo_score = _cv_score(additive_params, X_override=X_loo)
            full_delta = full_score - full_loo_score
            additive_delta = additive_score - additive_loo_score
            report.append(
                {
                    "feature": feature_names[col_idx],
                    "full_model_lofo_delta": float(full_delta),
                    "additive_model_lofo_delta": float(additive_delta),
                    "interaction_contribution": float(full_delta - additive_delta),
                }
            )
        report.sort(key=lambda row: row["interaction_contribution"], reverse=True)
        result["per_feature_interaction_report"] = report

    return result


__all__ = ["additive_interaction_diagnostic"]
