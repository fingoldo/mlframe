"""Feature selection within ML pipelines. Wrappers methods. Currently includes recursive feature elimination."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

while True:
    try:

        # ----------------------------------------------------------------------------------------------------------------------------
        # Normal Imports
        # ----------------------------------------------------------------------------------------------------------------------------

        from typing import *

        import pandas as pd, numpy as np
        from sklearn.base import is_classifier, is_regressor
        from sklearn.dummy import DummyClassifier, DummyRegressor

    except Exception as e:

        logger.warning(e)

        if "cannot import name" in str(e):
            raise (e)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Packages auto-install
        # ----------------------------------------------------------------------------------------------------------------------------

        from pyutilz.pythonlib import ensure_installed

        ensure_installed("numpy pandas scikit-learn")

    else:
        break

LARGE_CONST: float = 1e30


def get_best_dummy_score(
    estimator: object,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.DataFrame, np.ndarray, pd.Series],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.DataFrame, np.ndarray, pd.Series],
    scoring: object,
    verbose:bool=False,
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
        strategies = None
        logger.error(f"Unexpected estimator type in get_best_dummy_score: {estimator}")

    if strategies:
        for strategy in strategies.split():
            model = dummy_model_type(strategy=strategy)
            model.fit(X=X_train, y=y_train)
            dummy_score = scoring(model, X_test, y_test)
            if verbose:
                logger.info(f"strategy={strategy}, score={dummy_score:.6f}")
            if dummy_score > best_dummy_score:
                best_dummy_score = dummy_score

    return best_dummy_score
