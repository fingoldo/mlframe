"""Naive-Bayes log-odds combination of independent per-feature probability models.

COMPETITION / EXPLORATORY ONLY — NEVER wire this into production defaults.

Source: 2nd_santander-customer-transaction-prediction.md. Several top solutions
to that competition fit one calibrated probabilistic model per feature, then
combined predictions by SUMMING LOG-ODDS instead of averaging probabilities:

    P(Y=1|X) / P(Y=0|X)  ~  prod_i ( P(Y=1|x_i) / P(Y=0|x_i) )

This is only valid under CONDITIONAL independence of the features given the
label (not mere marginal/pairwise independence) -- a strong assumption. That
particular competition's synthetic data was engineered to have close-to-
conditionally-independent features, which is why the trick worked there.
Real production tabular data almost never satisfies conditional independence
(features are typically derived from overlapping underlying processes), so
naively combining per-feature models via log-odds summation on production
data will typically UNDER-perform (or at best match) simple probability
averaging, and can be badly overconfident/miscalibrated when features are
correlated. See the "honest negative" test in
``tests/competition/test_biz_val_naive_bayes_log_odds.py`` for a concrete
demonstration of this failure mode.

This module lives under ``mlframe.competition`` and must never be imported by
production mlframe modules or exported from mlframe's top-level ``__init__.py``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

__all__ = ["NaiveBayesLogOddsEnsembler"]

_EPS = 1e-12


def _clip_prob(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.clip(p, _EPS, 1.0 - _EPS)


def _to_log_odds(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    p = _clip_prob(p)
    return np.asarray(np.log(p) - np.log1p(-p), dtype=np.float64)


def _from_log_odds(logit: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray(1.0 / (1.0 + np.exp(-logit)), dtype=np.float64)


class NaiveBayesLogOddsEnsembler(BaseEstimator, ClassifierMixin):
    """Combine one calibrated model per feature (or feature block) via log-odds summation.

    COMPETITION / EXPLORATORY ONLY -- see module docstring. Valid only under
    CONDITIONAL feature independence given the label, an assumption almost
    never true for real production tabular data. Use only after an explicit
    feature-independence audit (e.g. mlframe's MI/redundancy tooling in
    ``mlframe.feature_selection``) confirms features are close to
    conditionally independent given the target -- otherwise prefer plain
    probability averaging or a joint model.

    Parameters
    ----------
    base_estimator:
        A binary-classification estimator prototype implementing
        ``fit``/``predict_proba``. Cloned once per feature (or feature block).
        Defaults to a calibrated logistic regression.
    feature_blocks:
        Optional grouping of column indices into blocks; one model is fit per
        block instead of per single feature. ``None`` fits one model per
        feature (the standard "one column at a time" setup from the source
        writeup).
    prior_correction:
        If True (default), subtracts ``(n_models - 1)`` times the class-prior
        log-odds from the summed log-odds before converting back to a
        probability. This is the standard Naive-Bayes prior-correction term
        that keeps the combined log-odds calibrated to the base rate instead
        of drifting with the number of feature models.
    """

    def __init__(
        self,
        base_estimator: Optional[ClassifierMixin] = None,
        feature_blocks: Optional[Sequence[Sequence[int]]] = None,
        calibrate: bool = True,
        prior_correction: bool = True,
    ) -> None:
        self.base_estimator = base_estimator
        self.feature_blocks = feature_blocks
        self.calibrate = calibrate
        self.prior_correction = prior_correction

    def _make_estimator(self) -> ClassifierMixin:
        proto: ClassifierMixin
        if self.base_estimator is None:
            proto = LogisticRegression(max_iter=1000)
        else:
            proto = clone(self.base_estimator)
        if self.calibrate:
            return CalibratedClassifierCV(proto, method="isotonic", cv=3)
        return proto

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> "NaiveBayesLogOddsEnsembler":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D")
        n_features = X_arr.shape[1]

        blocks: list[Sequence[int]]
        if self.feature_blocks is None:
            blocks = [(i,) for i in range(n_features)]
        else:
            blocks = list(self.feature_blocks)

        self.classes_ = np.unique(y_arr)
        if self.classes_.shape[0] != 2:
            raise ValueError("NaiveBayesLogOddsEnsembler only supports binary classification")

        self.blocks_ = blocks
        self.models_: list[ClassifierMixin] = []
        for block in blocks:
            block_idx = list(block)
            model = self._make_estimator()
            model.fit(X_arr[:, block_idx], y_arr)
            self.models_.append(model)

        prior = float(np.mean(y_arr == self.classes_[1]))
        self.prior_log_odds_ = float(_to_log_odds(np.array([prior]))[0])
        return self

    def _per_model_log_odds(self, X_arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        n_samples = X_arr.shape[0]
        n_models = len(self.models_)
        logits = np.empty((n_samples, n_models), dtype=np.float64)
        for j, (block, model) in enumerate(zip(self.blocks_, self.models_)):
            block_idx = list(block)
            proba = model.predict_proba(X_arr[:, block_idx])
            pos_col = list(model.classes_).index(self.classes_[1])
            logits[:, j] = _to_log_odds(proba[:, pos_col])
        return logits

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        X_arr = np.asarray(X, dtype=np.float64)
        logits = self._per_model_log_odds(X_arr)
        combined = logits.sum(axis=1)
        if self.prior_correction:
            n_models = logits.shape[1]
            combined = combined - (n_models - 1) * self.prior_log_odds_
        pos_proba = _from_log_odds(combined)
        return np.column_stack([1.0 - pos_proba, pos_proba])

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return np.asarray(self.classes_[idx])

    def predict_proba_average_baseline(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Companion baseline: plain probability averaging over the same fitted per-feature models.

        Exposed to make the log-odds-vs-averaging comparison trivial to run
        against the exact same fitted models (see biz_value tests).
        """
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples = X_arr.shape[0]
        n_models = len(self.models_)
        probas = np.empty((n_samples, n_models), dtype=np.float64)
        for j, (block, model) in enumerate(zip(self.blocks_, self.models_)):
            block_idx = list(block)
            proba = model.predict_proba(X_arr[:, block_idx])
            pos_col = list(model.classes_).index(self.classes_[1])
            probas[:, j] = proba[:, pos_col]
        pos_proba = probas.mean(axis=1)
        return np.column_stack([1.0 - pos_proba, pos_proba])
