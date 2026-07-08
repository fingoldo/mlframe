"""Sklearn-compatible wrappers that retrofit early stopping onto estimators requiring a fixed eval_set (CatBoost, LightGBM, XGBoost).

Carves an internal validation split from the fit-time X/y inside ``fit`` itself, so the wrapped estimator can live
inside a plain sklearn ``Pipeline``/``GridSearchCV`` without the caller having to pre-split data or manage eval_set wiring.
"""

from __future__ import annotations

import inspect
import logging

logger = logging.getLogger(__name__)

from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.base import RegressorMixin, ClassifierMixin, BaseEstimator, clone
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

    def fit(self, X, y, **fit_params):
        """Carve an internal validation split from ``X``/``y`` and fit the cloned base estimator with early stopping via ``eval_set``."""
        X = check_array(X)

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
                *arrays, test_size=self.test_size, train_size=self.train_size, random_state=random_state, shuffle=self.shuffle, stratify=self.stratify
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
                *arrays, test_size=self.test_size, train_size=self.train_size, random_state=random_state, shuffle=self.shuffle, stratify=self.stratify
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
        """Delegate prediction to the fitted base estimator."""
        check_is_fitted(self)
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
