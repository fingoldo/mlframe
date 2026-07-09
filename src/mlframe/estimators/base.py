"""Sklearn-compatible wrappers that retrofit early stopping onto estimators requiring a fixed eval_set (CatBoost, LightGBM, XGBoost).

Carves an internal validation split from the fit-time X/y inside ``fit`` itself, so the wrapped estimator can live
inside a plain sklearn ``Pipeline``/``GridSearchCV`` without the caller having to pre-split data or manage eval_set wiring.
"""

from __future__ import annotations

import inspect
import logging

logger = logging.getLogger(__name__)

import numpy as np

from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.base import RegressorMixin, ClassifierMixin, BaseEstimator, clone, is_classifier
from sklearn.utils import check_random_state

from sklearn.model_selection import train_test_split

class EstimatorWithEarlyStopping(BaseEstimator):
    """
    Adds early stopping in pipeline to estimators that only accept fixed evaluation set, like Catboost.
    """
    def __init__(self, base_estimator=None, test_size=0.05, train_size=None, random_state=None, shuffle=True, stratify=None, plot: bool = False):
        self.plot = plot
        self.base_estimator = base_estimator
        self.test_size, self.train_size, self.random_state, self.shuffle, self.stratify = test_size, train_size, random_state, shuffle, stratify

    def _resolve_stratify(self, y):
        """Auto-detect a stratify vector for the internal train/val split when the caller left ``stratify=None``.

        Mirrors ``EarlyStoppingWrapper._split`` (estimators/early_stopping.py): ``stratify`` cannot be passed at
        construction time since ``y`` only exists at ``fit``-time, so an explicit-``None`` default silently produced
        an unstratified split on every imbalanced classification target. Stratifies on ``y`` whenever the base
        estimator is a classifier (or ``y`` is low-cardinality -- some wrapped estimators like a bare CatBoostClassifier
        don't reliably report ``is_classifier``), every class has >=2 members, and the requested val split is large
        enough to hold at least one row per class; falls back to ``None`` (plain shuffle) otherwise.
        """
        y_arr = np.asarray(y)
        n = len(y_arr)
        try:
            is_clf_estimator = is_classifier(self.base_estimator)
        except Exception:
            is_clf_estimator = False
        low_cardinality = y_arr.dtype.kind in "OUS" or np.unique(y_arr).size <= max(2, min(20, n // 10))
        if not (is_clf_estimator or low_cardinality):
            return None
        classes, counts = np.unique(y_arr, return_counts=True)
        if counts.min() < 2:
            return None
        if isinstance(self.test_size, float):
            n_val = max(1, round(self.test_size * n))
        elif self.test_size is not None:
            n_val = int(self.test_size)
        else:
            n_val = n
        if n_val < classes.size:
            return None
        return y

    def fit(self, X, y, **fit_params):
        """Carve an internal validation split from ``X``/``y`` and fit the cloned base estimator with early stopping via ``eval_set``."""
        # A bare check_array(X) here would silently drop DataFrame column names / dtypes before the CatBoost/eval_set
        # dispatch, defeating native categorical-feature handling (cat_features indices matching named columns) and
        # any downstream reliance on feature_names_in_. Preserve the caller's frame; only ndarray-like inputs get validated.
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            X = check_array(X)

        stratify = self.stratify if self.stratify is not None else self._resolve_stratify(y)

        random_state = check_random_state(self.random_state)

        self.random_state_ = random_state
        fitted_estimator = clone(self.base_estimator)

        if "CatBoost" in type(fitted_estimator).__name__:
            # sample_weight must be split alongside X/y so train-fold weights align with
            # train rows and val-fold weights feed the eval_set; passing the full-length
            # weight to the train fold desyncs rows and errors on the length mismatch.
            sample_weight = fit_params.pop("sample_weight", None)
            arrays = [X, y] if sample_weight is None else [X, y, sample_weight]
            splits = train_test_split(
                *arrays, test_size=self.test_size, train_size=self.train_size, random_state=random_state, shuffle=self.shuffle, stratify=stratify
            )
            if sample_weight is None:
                X_train, X_val, y_train, y_val = splits
                eval_set = (X_val, y_val)
                fit_params_train = fit_params
            else:
                X_train, X_val, y_train, y_val, w_train, w_val = splits
                from catboost import Pool

                eval_set = Pool(X_val, y_val, weight=w_val)
                fit_params_train = {"sample_weight": w_train, **fit_params}
            fitted_estimator.fit(X_train, y_train, eval_set=eval_set, plot=self.plot, **fit_params_train)
        elif "eval_set" in inspect.signature(fitted_estimator.fit).parameters:
            # Generic eval-set estimators (LightGBM, XGBoost, ...) honour the same split params as
            # CatBoost; without this branch test_size/stratify/shuffle/random_state were silently dead.
            sample_weight = fit_params.pop("sample_weight", None)
            arrays = [X, y] if sample_weight is None else [X, y, sample_weight]
            splits = train_test_split(
                *arrays, test_size=self.test_size, train_size=self.train_size, random_state=random_state, shuffle=self.shuffle, stratify=stratify
            )
            if sample_weight is None:
                X_train, X_val, y_train, y_val = splits
                fit_params_train = fit_params
            else:
                X_train, X_val, y_train, y_val, w_train, w_val = splits
                fit_params_train = {"sample_weight": w_train, **fit_params}
            fitted_estimator.fit(X_train, y_train, eval_set=[(X_val, y_val)], **fit_params_train)
        else:
            logger.warning(
                "Early stopping not applicable: estimator of type %s accepts no eval_set; the split "
                "params (test_size/stratify/shuffle/random_state) do not apply and are ignored for this fit.",
                type(self.base_estimator),
            )
            fitted_estimator.fit(X, y, **fit_params)

        self.fitted_estimator_ = fitted_estimator

        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Delegate prediction to the fitted base estimator, preserving DataFrame columns like ``fit`` does."""
        check_is_fitted(self)
        if not hasattr(X, "columns"):
            X = check_array(X)

        return self.fitted_estimator_.predict(X)


class RegressorWithEarlyStopping(EstimatorWithEarlyStopping, RegressorMixin):
    """Regressor flavour of ``EstimatorWithEarlyStopping``.

    Wraps a base regressor so early stopping works inside an sklearn pipeline: ``fit`` carves an
    internal validation split from X/y and feeds it to the base estimator's ``eval_set``.
    """

    pass


class ClassifierWithEarlyStopping(EstimatorWithEarlyStopping, ClassifierMixin):
    """Classifier flavour of ``EstimatorWithEarlyStopping`` with ``predict_proba``/``decision_function`` passthrough.

    Wraps a base classifier so early stopping works inside an sklearn pipeline: ``fit`` carves an
    internal validation split from X/y and feeds it to the base estimator's ``eval_set``.
    """

    def predict_proba(self, X):
        """Delegate class-probability prediction to the fitted base estimator."""
        check_is_fitted(self)
        return self.fitted_estimator_.predict_proba(X)

    def decision_function(self, X):
        """Delegate decision-function scoring to the fitted base estimator (raises if it doesn't support one)."""
        check_is_fitted(self)
        if not hasattr(self.fitted_estimator_, "decision_function"):
            raise AttributeError(f"Wrapped estimator {type(self.fitted_estimator_).__name__} has no decision_function")
        return self.fitted_estimator_.decision_function(X)
