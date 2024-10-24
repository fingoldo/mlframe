"""Dealing with outliers in ML pipelines."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.pythonlib import ensure_installed

ensure_installed("imbalanced-learn scikit-learn")


from sklearn.ensemble import IsolationForest


# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

from imblearn import FunctionSampler
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def reject_outliers(
    X: object,
    y: object,
    model: object = None,
    verbose: bool = True,
):
    """Function used to resample the dataset by dropping the outliers. Should be a part of imblearn Pipeline:

    from imblearn import FunctionSampler
    from imblearn.pipeline import Pipeline
    pipe = Pipeline([("out", FunctionSampler(func=reject_outliers, validate=False)), ("est", clf)])

    """

    if model is None:
        model = Pipeline([("imp", SimpleImputer()), ("est", IsolationForest())])

    model.fit(X)
    y_pred = model.predict(X)
    idx = y_pred == 1

    if verbose:
        logger.info("Outlier rejection: received %s samples, kept %s", len(X), idx.sum())

    return X[idx], y[idx]
