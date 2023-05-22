"""Everything related to gaining predictions from already trained models."""

# pylint: disable=wrong-import-order,wrong-import-position,unidiomatic-typecheck,pointless-string-statement

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.pythonlib import (
    ensure_installed,
)  # lint: disable=ungrouped-imports,disable=wrong-import-order

ensure_installed("numpy pandas numba scipy sklearn antropy")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import joblib
import glob, os
import pandas as pd, numpy as np
from os.path import join, isfile, isdir, splitext

from pyutilz.system import tqdmu

# ----------------------------------------------------------------------------------------------------------------------------
# Disk ops
# ----------------------------------------------------------------------------------------------------------------------------


def read_trained_models(featureset: str, X: pd.DataFrame, features_file_name: str = "features.dump", inference_folder: str = "infer"):
    """Read trained models from a folder, along with required features names.
    Ensure that the models conform passed dataset.
    """

    models = {}

    fpath = join(inference_folder, featureset)
    if not isdir(fpath):
        return models

    features_file = join(fpath, features_file_name)
    if isfile(features_file):
        try:
            features = joblib.load(features_file)
        except Exception as e:
            logger.warning("Could not read features file %s of featureset %s", features_file_name, featureset)
        else:
            if isinstance(features, pd.core.indexes.base.Index):
                features = features.values.tolist()
            # print(features)
    else:
        logger.warning("Did not find features file for %s", featureset)

    if features is None:
        features = X.columns.values.tolist()
    else:
        X = X[features]

    for model_name in tqdmu(os.listdir(fpath), desc="Reading trained models"):
        model_file = join(fpath, model_name)
        if isfile(model_file) and model_name != features_file_name:
            # load model
            try:
                model = joblib.load(model_file)
            except Exception as e:
                logger.warning("Could not read model file %s of featureset %s: %s", model_file, featureset, e)
            else:
                # if it has feature_names attribute, check that it matches the featureset & features data
                if hasattr(model, "feature_names_in_"):
                    feature_names_in_ = model.feature_names_in_
                    if isinstance(feature_names_in_, np.ndarray):
                        feature_names_in_ = feature_names_in_.tolist()
                    if feature_names_in_ != features:
                        logger.error(
                            "model %s was trained on different features %s than featureset %s states: %s",
                            model,
                            model.feature_names_in_,
                            featureset,
                            features,
                        )
                        continue

                models[splitext(model_name)[0]] = model

    return models, X


# ----------------------------------------------------------------------------------------------------------------------------
# Real inference
# ----------------------------------------------------------------------------------------------------------------------------


def get_models_raw_predictions(trained_models: dict, X, Y):
    """X should already contain only right features in right order."""
    predictions = {}
    for model_name, model in tqdmu(trained_models.items(), desc="Getting raw predictions"):
        prediction = model.predict(X)

        # assert len(prediction) == cDAY_SIZE
        predictions[model_name] = prediction

    return predictions  # np.mean(np.array(predictions), axis=0)
