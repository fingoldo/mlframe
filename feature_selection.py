"""Feature selection within ML pipelines."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.pythonlib import ensure_installed

ensure_installed("numpy pandas")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import pandas as pd, numpy as np


from pyutilz.system import tqdmu
from mlframe.boruta_shap import BorutaShap


def find_impactful_features(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    feature_selector: object = None,
    model: object = None,
    importance_measure: str = "shap",
    classification: bool = False,
    n_trials: int = 150,
    normalize: bool = True,
    train_or_test="train",
    verbose: bool = True,
    fit_params: dict = {},
) -> dict:
    """
    Create a dict of inputs impacting every and all target (multitarget supported).
    Wrapped models are not supported (like TransformedTargetRegressor).
    """

    if verbose:
        logger.info("Starting impact analysis for %s row(s), %s feature(s), %s target(s)", X.shape[0], X.shape[1], Y.shape[1])

    if not feature_selector:
        feature_selector = BorutaShap(
            model=model,
            importance_measure=importance_measure,
            classification=classification,
            n_trials=n_trials,
            normalize=normalize,
            verbose=False,
            train_or_test=train_or_test,
            fit_params=fit_params,
        )

    if False:  # when multioutput is not supported
        max_targets = 0
        res = {"accepted": {}, "tentative": {}}
        wholeset_accepted, wholeset_tentative = [], []

        for var in tqdmu(range(Y.shape[1]), desc="target #"):
            if max_targets:
                if var >= max_targets:
                    break
            feature_selector.fit(X=X, y=Y.iloc[:, var], n_trials=n_trials, normalize=normalize, verbose=False, train_or_test=train_or_test)

            res["accepted"][var] = feature_selector.accepted
            res["tentative"][var] = feature_selector.tentative

            if verbose:
                logger.info(
                    "%s feature(s) found impactful on target %s: %s, %s tentative: %s",
                    len(feature_selector.accepted),
                    var,
                    sorted(feature_selector.accepted),
                    len(feature_selector.tentative),
                    sorted(feature_selector.tentative),
                )

            wholeset_accepted.extend(feature_selector.accepted)
            wholeset_tentative.extend(feature_selector.tentative)

        res["wholeset"] = {"accepted": set(wholeset_accepted), "tentative": set(wholeset_tentative)}
        res["mostvoted_accepted"] = Counter(wholeset_accepted)
        res["mostvoted_tentative"] = Counter(wholeset_tentative)
    else:
        feature_selector.fit(X=X, y=Y)
        res = {"accepted": sorted(feature_selector.accepted), "tentative": sorted(feature_selector.tentative)}
        if verbose:
            logger.info(res)
    return res
