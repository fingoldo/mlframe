"""Composite-target modelling for CLASSIFICATION via base-margin residuals.

The regression ``CompositeTargetEstimator`` learns ``T = f(y, base)`` and inverts
it. The classification analogue does not transform a continuous y -- instead it
gives the inner classifier a BASE MARGIN (init_score / base_margin, i.e. a
log-odds offset) from a cheap base model, so the inner only has to learn the
RESIDUAL log-odds on top:

    logit(P(y=1 | x)) = base_margin(x) + inner_raw_margin(x)

When one feature (or a calibrated external score) already explains most of the
signal, anchoring the inner on that base lets it focus its capacity on the
correction regions instead of re-deriving the dominant effect -- the
classification twin of forcing the regression model to explain the residual
after the lag. Boosters expose the hook natively: LightGBM ``init_score=``,
XGBoost ``base_margin=``, CatBoost ``baseline=``. An inner without a raw-margin
path is rejected with a clear error (the residual contract is undefined without
one).

Design choices mirroring the regression wrapper:
- sklearn-compatible (fit / predict_proba / predict / get_params); the base
  model + the inner are looked up by config, never captured as closures, so
  clone / pickle stay clean.
- Binary only here (one base margin column); multiclass base-margin is a future
  extension (it needs a (n, K) margin and per-class offsets).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

logger = logging.getLogger(__name__)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable logistic.
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _inner_raw_margin(model: Any, X: Any) -> np.ndarray:
    """Raw (logit) margin from a fitted booster, family-dispatched."""
    # LightGBM
    try:
        import lightgbm as lgb
        if isinstance(model, lgb.LGBMClassifier):
            return np.asarray(model.predict(X, raw_score=True), dtype=np.float64).reshape(-1)
    except Exception:
        pass
    # XGBoost
    try:
        import xgboost as xgb
        if isinstance(model, xgb.XGBClassifier):
            return np.asarray(model.predict(X, output_margin=True), dtype=np.float64).reshape(-1)
    except Exception:
        pass
    # CatBoost
    try:
        import catboost as cb
        if isinstance(model, cb.CatBoostClassifier):
            return np.asarray(
                model.predict(X, prediction_type="RawFormulaVal"), dtype=np.float64,
            ).reshape(-1)
    except Exception:
        pass
    raise NotImplementedError(
        f"CompositeClassificationEstimator: inner {type(model).__name__!r} has no "
        "raw-margin path (LightGBM raw_score / XGBoost output_margin / CatBoost "
        "RawFormulaVal). The base-margin residual contract is undefined without "
        "one -- use a gradient-boosting inner or a plain classifier instead."
    )


def _fit_inner_with_init_score(model: Any, X: Any, y: np.ndarray, init_score: np.ndarray, sample_weight=None) -> None:
    """Fit a booster with a per-row base margin, family-dispatched kwarg name."""
    kw: dict[str, Any] = {}
    if sample_weight is not None:
        kw["sample_weight"] = sample_weight
    cls = type(model).__name__
    if "LGBM" in cls:
        model.fit(X, y, init_score=init_score, **kw)
    elif "XGB" in cls:
        model.fit(X, y, base_margin=init_score, **kw)
    elif "CatBoost" in cls:
        model.fit(X, y, baseline=init_score, **kw)
    else:
        raise NotImplementedError(
            f"CompositeClassificationEstimator: inner {cls!r} does not accept a "
            "base margin (init_score / base_margin / baseline)."
        )


class CompositeClassificationEstimator(BaseEstimator, ClassifierMixin):
    """Binary classifier that learns the residual log-odds over a base margin.

    Parameters
    ----------
    base_estimator
        The inner gradient-boosting classifier (LightGBM / XGBoost / CatBoost)
        trained on the residual. Cloned at fit.
    base_margin_estimator
        A cheap model producing the BASE log-odds. Anything exposing
        ``decision_function`` (e.g. sklearn ``LogisticRegression``) or
        ``predict_proba``; cloned at fit. Default: ``LogisticRegression``.
    base_margin_column
        Alternatively, the name of a column in X holding a PRECOMPUTED base
        margin (a calibrated external score's logit). When set,
        ``base_margin_estimator`` is ignored and the column is stripped from X
        before fitting the inner.
    """

    def __init__(
        self,
        base_estimator: Any = None,
        base_margin_estimator: Any = None,
        base_margin_column: str | None = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.base_margin_estimator = base_margin_estimator
        self.base_margin_column = base_margin_column

    # -- base margin extraction -------------------------------------------------
    def _margin_from_estimator(self, est: Any, X: Any) -> np.ndarray:
        if hasattr(est, "decision_function"):
            return np.asarray(est.decision_function(X), dtype=np.float64).reshape(-1)
        proba = np.asarray(est.predict_proba(X), dtype=np.float64)
        p1 = proba[:, 1] if proba.ndim == 2 else proba.reshape(-1)
        p1 = np.clip(p1, 1e-6, 1.0 - 1e-6)
        return np.log(p1 / (1.0 - p1))

    def _extract_margin_column(self, X: Any) -> np.ndarray:
        col = self.base_margin_column
        if hasattr(X, "get_column"):  # polars
            return np.asarray(X.get_column(col).to_numpy(), dtype=np.float64).reshape(-1)
        return np.asarray(X[col].to_numpy(), dtype=np.float64).reshape(-1)

    def _drop_margin_column(self, X: Any) -> Any:
        col = self.base_margin_column
        if hasattr(X, "drop") and hasattr(X, "get_column"):  # polars
            return X.drop(col) if col in X.columns else X
        if hasattr(X, "drop"):  # pandas
            return X.drop(columns=[col]) if col in X.columns else X
        return X

    # -- sklearn API ------------------------------------------------------------
    def fit(self, X: Any, y: Any, sample_weight=None) -> "CompositeClassificationEstimator":
        y_arr = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(y_arr)
        if self.classes_.size != 2:
            raise ValueError(
                "CompositeClassificationEstimator supports binary targets only; "
                f"got {self.classes_.size} classes."
            )
        y01 = (y_arr == self.classes_[1]).astype(np.float64)

        if self.base_margin_column is not None:
            base_margin = self._extract_margin_column(X)
            X_inner = self._drop_margin_column(X)
            self.base_margin_estimator_ = None
        else:
            from sklearn.linear_model import LogisticRegression
            self.base_margin_estimator_ = clone(
                self.base_margin_estimator
                if self.base_margin_estimator is not None
                else LogisticRegression(max_iter=1000)
            )
            self.base_margin_estimator_.fit(X, y01, sample_weight=sample_weight) \
                if sample_weight is not None else self.base_margin_estimator_.fit(X, y01)
            base_margin = self._margin_from_estimator(self.base_margin_estimator_, X)
            X_inner = X

        if self.base_estimator is None:
            raise ValueError("CompositeClassificationEstimator requires base_estimator (a GBDT classifier).")
        self.estimator_ = clone(self.base_estimator)
        _fit_inner_with_init_score(self.estimator_, X_inner, y01, base_margin, sample_weight)
        self.n_features_in_ = X_inner.shape[1]
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            from sklearn.exceptions import NotFittedError
            raise NotFittedError("CompositeClassificationEstimator.predict called before fit.")
        if self.base_margin_column is not None:
            base_margin = self._extract_margin_column(X)
            X_inner = self._drop_margin_column(X)
        else:
            base_margin = self._margin_from_estimator(self.base_margin_estimator_, X)
            X_inner = X
        return base_margin + _inner_raw_margin(self.estimator_, X_inner)

    def predict_proba(self, X: Any) -> np.ndarray:
        p1 = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: Any) -> np.ndarray:
        idx = (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        return self.classes_[idx]
