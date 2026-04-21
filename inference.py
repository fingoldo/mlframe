"""Everything related to gaining predictions from already trained models."""

# pylint: disable=wrong-import-order,wrong-import-position,unidiomatic-typecheck,pointless-string-statement

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import hashlib
import joblib
import os
import pandas as pd, numpy as np
from os.path import join, isfile, isdir, splitext

from pyutilz.system import tqdmu

# Allow-listed extensions for joblib model deserialization. Anything outside
# this set is skipped to make "drop a planted .pkl in the dir" attacks harder.
_ALLOWED_MODEL_EXTENSIONS = frozenset({".dump", ".joblib", ".pkl", ".pickle"})


def _sha256_of_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _verify_sidecar(path: str) -> bool:
    """If a ``<path>.sha256`` sidecar exists, verify it matches; otherwise return True.

    Returns False only when a sidecar exists and its digest does NOT match.
    """
    sidecar = path + ".sha256"
    if not isfile(sidecar):
        return True
    with open(sidecar, "r", encoding="utf-8") as f:
        expected = f.read().strip().split()[0].lower()
    actual = _sha256_of_file(path).lower()
    return expected == actual


def _load_features_file(features_file: str):
    """Prefer orjson-based sidecar (``features_file + '.json'`` or a ``.json`` twin),
    fall back to joblib. Returns a list[str] or None on failure.
    """
    json_candidates = []
    base, ext = splitext(features_file)
    json_candidates.append(features_file + ".json")
    if ext.lower() != ".json":
        json_candidates.append(base + ".json")

    for candidate in json_candidates:
        if isfile(candidate):
            try:
                import orjson
                with open(candidate, "rb") as f:
                    data = orjson.loads(f.read())
            except ImportError:
                import json
                with open(candidate, "r", encoding="utf-8") as f:
                    data = json.load(f)
            if isinstance(data, list):
                return [str(c) for c in data]
            logger.warning("JSON features file %s did not contain a list", candidate)
            return None

    if not _verify_sidecar(features_file):
        logger.error("sha256 mismatch for features file %s; refusing to load", features_file)
        return None

    features = joblib.load(features_file)
    if isinstance(features, pd.core.indexes.base.Index):
        features = features.values.tolist()
    return features


# ----------------------------------------------------------------------------------------------------------------------------
# Disk ops
# ----------------------------------------------------------------------------------------------------------------------------


def read_trained_models(
    featureset: str,
    X: pd.DataFrame,
    features_file_name: str = "features.dump",
    inference_folder: str = "infer",
    trusted_root: Optional[str] = None,
    allowed_extensions: Optional[Iterable[str]] = None,
):
    """Read trained models from a folder, along with required features names.
    Ensure that the models conform passed dataset.

    If ``trusted_root`` is provided, the resolved model directory MUST be inside it
    (absolute-path commonpath check). Otherwise a ValueError is raised.

    Model files are only loaded when their extension is in ``allowed_extensions``
    (default: ``.dump``, ``.joblib``, ``.pkl``, ``.pickle``) and when an optional
    ``<model>.sha256`` sidecar, if present, matches. A features sidecar JSON
    (``features.dump.json`` or ``features.json``) is preferred over the joblib
    dump when available.
    """

    models = {}
    allowed = frozenset(e.lower() for e in allowed_extensions) if allowed_extensions is not None else _ALLOWED_MODEL_EXTENSIONS

    fpath = join(inference_folder, featureset)
    if trusted_root is not None:
        abs_root = os.path.abspath(trusted_root)
        abs_fpath = os.path.abspath(fpath)
        try:
            common = os.path.commonpath([abs_root, abs_fpath])
        except ValueError:
            raise ValueError(f"Path {abs_fpath} is not inside trusted_root {abs_root}")
        if common != abs_root:
            raise ValueError(f"Path {abs_fpath} is not inside trusted_root {abs_root}")
    if not isdir(fpath):
        return models, X

    features = None
    features_file = join(fpath, features_file_name)
    if isfile(features_file) or isfile(features_file + ".json") or isfile(splitext(features_file)[0] + ".json"):
        try:
            features = _load_features_file(features_file)
        except Exception:
            logger.warning("Could not read features file %s of featureset %s", features_file_name, featureset, exc_info=True)
    else:
        logger.warning("Did not find features file for %s", featureset)

    if features is None:
        features = X.columns.values.tolist()
    else:
        X = X[features]

    sidecar_suffixes = (".sha256",)
    feature_sidecars = {
        features_file_name,
        features_file_name + ".json",
        splitext(features_file_name)[0] + ".json",
    }
    for model_name in tqdmu(sorted(os.listdir(fpath)), desc="Reading trained models"):
        if model_name in feature_sidecars or model_name.endswith(sidecar_suffixes):
            continue
        ext = splitext(model_name)[1].lower()
        if ext not in allowed:
            continue
        model_file = join(fpath, model_name)
        if not isfile(model_file):
            continue
        if not _verify_sidecar(model_file):
            logger.error("sha256 mismatch for model %s; skipping", model_file)
            continue
        try:
            model = joblib.load(model_file)
        except Exception as e:
            logger.warning("Could not read model file %s of featureset %s: %s", model_file, featureset, e)
            continue

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
