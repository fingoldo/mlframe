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

from ._booster_margin import inner_raw_margin

logger = logging.getLogger(__name__)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid: splits on sign of ``z`` so ``exp`` never overflows on large-magnitude margins."""
    # Numerically stable logistic.
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax for the multiclass ``(n, K)`` margin (max-subtraction before exp)."""
    # Row-wise stable softmax for the multiclass (n, K) margin.
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return np.asarray(e / e.sum(axis=1, keepdims=True))


def _inner_raw_margin(model: Any, X: Any) -> np.ndarray:
    """Raw (logit) margin from a fitted booster, family-dispatched.

    Returns ``(n,)`` for binary and ``(n, K)`` for multiclass -- shape preserved
    so the softmax/sigmoid caller can add the matching base margin.
    """
    return inner_raw_margin(
        model, X,
        lgbm_attr="LGBMClassifier", xgb_attr="XGBClassifier", catboost_attr="CatBoostClassifier",
        wrapper_name="CompositeClassificationEstimator", keep_2d=True,
    )


def _fit_inner_with_init_score(model: Any, X: Any, y: np.ndarray, init_score: np.ndarray, n_classes: int, sample_weight=None) -> None:
    """Fit a booster with a per-row base margin, family-dispatched.

    For multiclass ``init_score`` is ``(n, K)``. LightGBM wants it flattened
    class-major ``(n*K,)`` (``ravel(order="F")``); XGBoost and CatBoost accept
    the ``(n, K)`` matrix directly.
    """
    kw: dict[str, Any] = {}
    if sample_weight is not None:
        kw["sample_weight"] = sample_weight
    cls = type(model).__name__
    if "LGBM" in cls:
        score = init_score if n_classes <= 2 else np.asarray(init_score, dtype=np.float64).ravel(order="F")
        model.fit(X, y, init_score=score, **kw)
    elif "XGB" in cls:
        model.fit(X, y, base_margin=init_score, **kw)
    elif "CatBoost" in cls:
        model.fit(X, y, baseline=init_score, **kw)
    else:
        raise NotImplementedError(f"CompositeClassificationEstimator: inner {cls!r} does not accept a " "base margin (init_score / base_margin / baseline).")


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
        """Base log-odds margin: ``(n,)`` for binary, ``(n, K)`` for multiclass.

        Softmax is shift-invariant so ``log(proba)`` is a valid multiclass margin
        when ``decision_function`` is absent.
        """
        n_classes = int(getattr(self, "n_classes_", 2))
        if n_classes <= 2:
            if hasattr(est, "decision_function"):
                return np.asarray(est.decision_function(X), dtype=np.float64).reshape(-1)
            proba = np.asarray(est.predict_proba(X), dtype=np.float64)
            p1 = proba[:, 1] if proba.ndim == 2 else proba.reshape(-1)
            p1 = np.clip(p1, 1e-6, 1.0 - 1e-6)
            return np.log(p1 / (1.0 - p1))
        if hasattr(est, "decision_function"):
            df = np.asarray(est.decision_function(X), dtype=np.float64)
            if df.ndim == 2 and df.shape[1] == n_classes:
                return df
        proba = np.clip(np.asarray(est.predict_proba(X), dtype=np.float64), 1e-12, 1.0)
        return np.log(proba)

    def _extract_margin_column(self, X: Any) -> np.ndarray:
        """Pull the precomputed base-margin column out of ``X`` as a flat float64 array (polars or pandas)."""
        col = self.base_margin_column
        if hasattr(X, "get_column"):  # polars
            return np.asarray(X.get_column(col).to_numpy(), dtype=np.float64).reshape(-1)
        return np.asarray(X[col].to_numpy(), dtype=np.float64).reshape(-1)

    def _drop_margin_column(self, X: Any) -> Any:
        """Return ``X`` with the base-margin column removed (polars or pandas), so it never leaks into the inner as a feature."""
        col = self.base_margin_column
        if hasattr(X, "drop") and hasattr(X, "get_column"):  # polars
            return X.drop(col) if col in X.columns else X
        if hasattr(X, "drop"):  # pandas
            return X.drop(columns=[col]) if col in X.columns else X
        return X

    # -- sklearn API ------------------------------------------------------------
    def fit(self, X: Any, y: Any, sample_weight=None) -> "CompositeClassificationEstimator":
        """Fit the base margin (or extract it from ``base_margin_column``), then fit the inner booster on the residual log-odds via its native init-score hook."""
        y_arr = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(y_arr)
        self.n_classes_ = int(self.classes_.size)
        if self.n_classes_ < 2:
            raise ValueError("CompositeClassificationEstimator needs >= 2 classes; got " f"{self.n_classes_}.")
        # Label-encode to 0..K-1 in classes_ order (the order the inner booster
        # + the LogisticRegression base both emit margins in).
        y_enc = np.searchsorted(self.classes_, y_arr).astype(np.int64 if self.n_classes_ > 2 else np.float64)

        if self.base_margin_column is not None:
            if self.n_classes_ > 2:
                raise ValueError(
                    "base_margin_column is a single column and cannot carry the " "K-class base margin; use a base_margin_estimator for multiclass."
                )
            base_margin = self._extract_margin_column(X)
            X_inner = self._drop_margin_column(X)
            self.base_margin_estimator_ = None
        else:
            from sklearn.linear_model import LogisticRegression

            self.base_margin_estimator_ = clone(self.base_margin_estimator if self.base_margin_estimator is not None else LogisticRegression(max_iter=1000))
            if sample_weight is not None:
                self.base_margin_estimator_.fit(X, y_enc, sample_weight=sample_weight)
            else:
                self.base_margin_estimator_.fit(X, y_enc)
            base_margin = self._margin_from_estimator(self.base_margin_estimator_, X)
            X_inner = X

        if self.base_estimator is None:
            raise ValueError("CompositeClassificationEstimator requires base_estimator (a GBDT classifier).")
        self.estimator_ = clone(self.base_estimator)
        _fit_inner_with_init_score(
            self.estimator_, X_inner, y_enc, base_margin, self.n_classes_, sample_weight,
        )
        # ``n_features_in_`` reflects the full X seen at fit, identical across the base_margin_column and
        # base_margin_estimator paths (the dropped margin column is plumbing, not a learned feature dimension).
        cols = getattr(X, "columns", None)
        if cols is not None:
            self.n_features_in_ = len(cols)
        else:
            self.n_features_in_ = int(X.shape[1])
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        """Total margin = base margin (from the column or the fitted base estimator) + the inner's raw residual margin."""
        if not hasattr(self, "estimator_"):
            from sklearn.exceptions import NotFittedError
            raise NotFittedError("CompositeClassificationEstimator.predict called before fit.")
        if self.base_margin_column is not None:
            base_margin = self._extract_margin_column(X)
            X_inner = self._drop_margin_column(X)
        else:
            base_margin = self._margin_from_estimator(self.base_margin_estimator_, X)
            X_inner = X
        return np.asarray(base_margin + _inner_raw_margin(self.estimator_, X_inner))

    def predict_proba(self, X: Any) -> np.ndarray:
        """Convert the total margin to class probabilities: sigmoid for binary, softmax for multiclass."""
        if not hasattr(self, "estimator_"):
            from sklearn.exceptions import NotFittedError
            raise NotFittedError("CompositeClassificationEstimator.predict_proba called before fit.")
        margin = self.decision_function(X)
        if self.n_classes_ <= 2:
            p1 = _sigmoid(np.asarray(margin, dtype=np.float64).reshape(-1))
            return np.column_stack([1.0 - p1, p1])
        return _softmax(np.asarray(margin, dtype=np.float64))

    def predict(self, X: Any) -> np.ndarray:
        """Argmax class label over ``predict_proba``, mapped back through ``classes_``."""
        proba = self.predict_proba(X)
        return np.asarray(self.classes_[np.argmax(proba, axis=1)])

    def calibration_report(self, X: Any, y: Any, n_bins: int = 10) -> dict:
        """Top-label reliability diagram + Expected Calibration Error (ECE).

        Bins the predicted CONFIDENCE (max class probability) into ``n_bins``
        equal-width bins and compares mean confidence vs observed accuracy in
        each -- the standard reliability curve, valid for binary and multiclass.
        ECE is the count-weighted mean ``|confidence - accuracy|``; lower is
        better-calibrated. Returns the per-bin arrays + the scalar ECE so the
        caller can plot or gate on it.
        """
        proba = self.predict_proba(X)
        conf = proba.max(axis=1)
        pred = self.classes_[np.argmax(proba, axis=1)]
        y_true = np.asarray(y).reshape(-1)
        correct = (pred == y_true).astype(np.float64)
        edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
        idx = np.clip(np.digitize(conf, edges[1:-1]), 0, int(n_bins) - 1)
        bin_conf = np.full(int(n_bins), np.nan)
        bin_acc = np.full(int(n_bins), np.nan)
        bin_cnt = np.zeros(int(n_bins), dtype=np.int64)
        ece = 0.0
        n = conf.size
        for b in range(int(n_bins)):
            m = idx == b
            c = int(m.sum())
            bin_cnt[b] = c
            if c:
                bin_conf[b] = float(conf[m].mean())
                bin_acc[b] = float(correct[m].mean())
                ece += (c / n) * abs(bin_conf[b] - bin_acc[b])
        return {
            "bin_confidence": bin_conf, "bin_accuracy": bin_acc,
            "bin_count": bin_cnt, "ece": float(ece),
        }


# Split-conformal prediction SETS. Bound from ``composite/conformal_classification.py``
# (which imports nothing from this module, so no cycle). calibrate_conformal_set(
# X_cal, y_cal, alpha, score) fits the threshold from a held-out set; predict_set(
# X, alpha, score) returns per-row coverage-guaranteed label sets.
from .conformal_classification import (
    calibrate_conformal_set as _calibrate_conformal_set,
    predict_set as _predict_set,
)

CompositeClassificationEstimator.calibrate_conformal_set = _calibrate_conformal_set
CompositeClassificationEstimator.predict_set = _predict_set

# Inductive Venn-Abers probability calibration (binary). Bound from
# ``composite/venn_abers.py`` (no import cycle). calibrate_venn_abers(X_cal, y_cal)
# fits the two isotonic envelopes on a held-out set; predict_proba_interval(X)
# returns the calibrated [p_low, p_high] multiprobability; predict_proba_venn_abers(X)
# returns the regularised calibrated (n, 2) predict_proba.
from .venn_abers import (
    calibrate_venn_abers as _calibrate_venn_abers,
    predict_proba_interval as _predict_proba_interval,
    predict_proba_venn_abers as _predict_proba_venn_abers,
)

CompositeClassificationEstimator.calibrate_venn_abers = _calibrate_venn_abers
CompositeClassificationEstimator.predict_proba_interval = _predict_proba_interval
CompositeClassificationEstimator.predict_proba_venn_abers = _predict_proba_venn_abers
