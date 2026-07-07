"""Feature selection within ML pipelines. Wrappers methods. Currently includes recursive feature elimination."""

from __future__ import annotations

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Callable, Union

import pandas as pd, numpy as np
from sklearn.base import is_classifier, is_regressor
from sklearn.dummy import DummyClassifier, DummyRegressor

LARGE_CONST: float = 1e30


def get_best_dummy_score(
    estimator: object,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.DataFrame, np.ndarray, pd.Series],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.DataFrame, np.ndarray, pd.Series],
    scoring: Callable,
    verbose: bool = False,
) -> float:
    """Given estimator type & train and test sets, finds the best respective dummy estimator"""
    best_dummy_score = -LARGE_CONST

    if is_classifier(estimator):
        dummy_model_type = DummyClassifier
        strategies = "most_frequent prior stratified uniform"
    elif is_regressor(estimator):
        dummy_model_type = DummyRegressor
        strategies = "mean median"
    else:
        raise TypeError(f"get_best_dummy_score: estimator must be a sklearn classifier or regressor, " f"got {type(estimator).__name__}")

    for strategy in strategies.split():
        model = dummy_model_type(strategy=strategy)
        model.fit(X=X_train, y=y_train)
        dummy_score = scoring(model, X_test, y_test)
        if verbose:
            logger.info("strategy=%s, score=%.6f", strategy, dummy_score)
        if dummy_score > best_dummy_score:
            best_dummy_score = dummy_score

    return best_dummy_score
