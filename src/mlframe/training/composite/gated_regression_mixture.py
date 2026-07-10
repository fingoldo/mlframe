"""``GatedRegressionMixture``: gate classifier routes to branch regressors, gate probability stacked as a feature.

Source: Elo Merchant Category Recommendation 5th place -- an outlier-probability binary classifier used for
(a) routing rows into a "low probability" vs "high probability" regression model by threshold, (b) as an
input feature to the high-probability regressor (stacked), and (c) downweighting rare outliers in the
low-probability regressor's loss.

Distinct from :class:`GatedOutlierEstimator` (this session's earlier addition): that class BLENDS a
classifier probability with ONE regressor's prediction (``p*constant + (1-p)*regressor.predict(X)``) and
never exposes the probability to the regressor itself. This class instead does HARD threshold ROUTING to N
independently-configured branch regressors (each can be a different estimator, tuned differently, matching
the source's "low prob" vs "high prob" specialist split), and -- the source's distinguishing "stacked
feature" trick -- appends the gate's own OOF probability as an EXTRA INPUT COLUMN to each branch regressor,
so a branch regressor can use "how confident the gate was" as a smooth signal on top of the hard route.
Per-branch ``sample_weight`` multipliers implement the source's rare-outlier downweighting.

Leakage discipline: the gate classifier's probabilities used to route/feature-stack the TRAINING rows are
OOF (via ``sklearn.model_selection.cross_val_predict(method="predict_proba")``) -- no row's own label leaks
into its own gate probability.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

logger = logging.getLogger(__name__)

_LOW = "low"
_HIGH = "high"


def _concat_feature(X: Any, col_name: str, values: np.ndarray) -> Any:
    if isinstance(X, pd.DataFrame):
        out = X.copy()
        out[col_name] = values
        return out
    try:
        import polars as pl
        if isinstance(X, pl.DataFrame):
            return X.with_columns(pl.Series(col_name, values))
    except ImportError:
        pass
    return np.concatenate([np.asarray(X, dtype=np.float64), values.reshape(-1, 1)], axis=1)


class GatedRegressionMixture(BaseEstimator, RegressorMixin):
    """Gate classifier hard-routes rows to branch regressors; gate probability is stacked as a feature.

    Parameters
    ----------
    gate_classifier
        sklearn-compatible probabilistic classifier prototype, cloned and OOF-fit against a binary
        subpopulation label (e.g. "is this row an outlier").
    low_regressor, high_regressor
        sklearn-compatible regressor prototypes for the below-threshold ("low" gate probability) and
        at-or-above-threshold ("high") branches respectively.
    threshold
        Gate probability cutoff for routing (default 0.5).
    use_gate_feature
        If True (default), the gate's OOF probability is appended as an extra input column to BOTH branch
        regressors at fit time, and the fitted gate's probability likewise at predict time.
    branch_sample_weight
        ``{"low": w, "high": w}`` multipliers applied to that branch's rows in its regressor's
        ``sample_weight`` (matches the source's rare-outlier downweighting; default ``{"low": 1.0, "high":
        1.0}``, i.e. no reweighting).
    n_splits, random_state
        Gate classifier's OOF CV configuration.

    Attributes
    ----------
    gate_model_, branch_models_
        The fitted gate classifier and ``{"low": regressor, "high": regressor}``.
    """

    def __init__(
        self,
        gate_classifier: Any,
        low_regressor: Any,
        high_regressor: Any,
        threshold: float = 0.5,
        use_gate_feature: bool = True,
        branch_sample_weight: Optional[Dict[str, float]] = None,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        self.gate_classifier = gate_classifier
        self.low_regressor = low_regressor
        self.high_regressor = high_regressor
        self.threshold = threshold
        self.use_gate_feature = use_gate_feature
        self.branch_sample_weight = branch_sample_weight
        self.n_splits = n_splits
        self.random_state = random_state

    def _predict_proba_1(self, model: Any, X: Any) -> np.ndarray:
        return np.asarray(model.predict_proba(X), dtype=np.float64)[:, 1]

    def fit(self, X: Any, y: Any, subpop_label: np.ndarray) -> "GatedRegressionMixture":
        y_arr = np.asarray(y, dtype=np.float64)
        label_arr = np.asarray(subpop_label)
        weights = self.branch_sample_weight or {_LOW: 1.0, _HIGH: 1.0}

        # composite_oof_predictions calls .predict(), a poor proxy for a probability (a classifier's hard-
        # label predict discards calibration); use cross_val_predict(method="predict_proba") directly so the
        # gate feature/route reflects the classifier's true probabilistic output.
        from sklearn.model_selection import KFold, cross_val_predict

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        oof_proba = cross_val_predict(clone(self.gate_classifier), X, label_arr, cv=kf, method="predict_proba")[:, 1]

        self.gate_model_ = clone(self.gate_classifier)
        self.gate_model_.fit(X, label_arr)

        route = np.where(oof_proba >= self.threshold, _HIGH, _LOW)
        self.branch_models_: Dict[str, Any] = {}
        for branch, regressor in ((_LOW, self.low_regressor), (_HIGH, self.high_regressor)):
            mask = route == branch
            if not mask.any():
                logger.warning("GatedRegressionMixture: branch %s has no routed rows at fit time.", branch)
                continue
            X_branch = X.iloc[np.flatnonzero(mask)] if hasattr(X, "iloc") else np.asarray(X)[mask]
            if self.use_gate_feature:
                X_branch = _concat_feature(X_branch, "gate_proba", oof_proba[mask])
            model = clone(regressor)
            sw = np.full(int(mask.sum()), weights.get(branch, 1.0), dtype=np.float64)
            try:
                model.fit(X_branch, y_arr[mask], sample_weight=sw)
            except TypeError:
                model.fit(X_branch, y_arr[mask])
            self.branch_models_[branch] = model
        return self

    def predict(self, X: Any) -> np.ndarray:
        proba = self._predict_proba_1(self.gate_model_, X)
        route = np.where(proba >= self.threshold, _HIGH, _LOW)
        n = proba.shape[0]
        out = np.zeros(n, dtype=np.float64)
        for branch in (_LOW, _HIGH):
            mask = route == branch
            if not mask.any() or branch not in self.branch_models_:
                continue
            X_branch = X.iloc[np.flatnonzero(mask)] if hasattr(X, "iloc") else np.asarray(X)[mask]
            if self.use_gate_feature:
                X_branch = _concat_feature(X_branch, "gate_proba", proba[mask])
            out[mask] = np.asarray(self.branch_models_[branch].predict(X_branch), dtype=np.float64)
        return out


__all__ = ["GatedRegressionMixture"]
