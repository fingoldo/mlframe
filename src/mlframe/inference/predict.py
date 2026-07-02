"""Everything related to gaining predictions from already trained models."""

from __future__ import annotations


# pylint: disable=wrong-import-order,wrong-import-position,unidiomatic-typecheck,pointless-string-statement

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Iterable, Optional

import hashlib
import joblib
import os
import pandas as pd, numpy as np
from os.path import join, isfile, isdir, splitext

from pyutilz.system import tqdmu

from mlframe.utils.safe_pickle import (
    _sha256_of_file as _safe_pickle_sha256_of_file,
    verify_sidecar as _safe_pickle_verify_sidecar,
)

# Allow-listed extensions for joblib model deserialization. Anything outside
# this set is skipped to make "drop a planted .pkl in the dir" attacks harder.
_ALLOWED_MODEL_EXTENSIONS = frozenset({".dump", ".joblib", ".pkl", ".pickle"})


def _sha256_of_file(path: str, chunk: int = 1 << 20) -> str:
    return _safe_pickle_sha256_of_file(path, chunk=chunk)


def _verify_sidecar(path: str) -> bool:
    """Back-compat shim delegating to :func:`mlframe.utils.safe_pickle.verify_sidecar`."""
    return _safe_pickle_verify_sidecar(path)


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
                with open(candidate, encoding="utf-8") as f:
                    data = json.load(f)
            if isinstance(data, list):
                return [str(c) for c in data]
            logger.warning("JSON features file %s did not contain a list", candidate)
            return None

    if not _verify_sidecar(features_file):
        logger.error("sha256 mismatch for features file %s; refusing to load", features_file)
        return None

    # Trusts the sha256 sidecar verified just above: an integrity/corruption gate, NOT authenticity (an attacker with dir write access rewrites both).
    features = joblib.load(features_file)
    if isinstance(features, pd.core.indexes.base.Index):
        # ``.to_list()`` is the pandas-modern path (handles nullable dtypes); ``.values.tolist()`` round-trips through ndarray needlessly.
        features = features.to_list()
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
    # Containment check runs ALWAYS: when the caller passes no trusted_root, default it to
    # inference_folder so a malicious featureset ("../.." or an absolute path) cannot escape and make
    # read_trained_models joblib.load an arbitrary pickle. The intended model dir is always inside
    # inference_folder, so this never rejects a legitimate call.
    abs_root = os.path.abspath(trusted_root if trusted_root is not None else inference_folder)
    abs_fpath = os.path.abspath(fpath)
    try:
        common = os.path.commonpath([abs_root, abs_fpath])
    except ValueError as e:
        # preserve the original ValueError ("Paths don't have the same drive" on Windows) via `from e`
        # so cross-drive root mismatches don't masquerade as path-traversal.
        raise ValueError(f"Path {abs_fpath} is not inside trusted_root {abs_root}") from e
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
        features = X.columns.to_list()
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
        # Wave 19 P1: validate the .meta.json sidecar (written since
        # 2026-05-20) before unpickling so library-version drift gets a
        # WARN log instead of producing a cryptic AttributeError deep
        # inside predict(). Non-fatal: legacy bundles (no sidecar) keep
        # loading silently per back-compat contract.
        try:
            from mlframe.training.io import validate_load_meta_sidecar as _vlms
            _vlms(model_file, strict=False)
        except Exception as _meta_e:
            logger.debug(
                "inference.read_trained_models: sidecar validation raised "
                "for %s: %s; proceeding with load.", model_file, _meta_e,
            )
        try:
            # Trusts the sha256 sidecar verified above: integrity/corruption gate, NOT authenticity (dir-write attacker rewrites payload+sidecar).
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
    """X should already contain only right features in right order.

    For classifiers this returns PROBABILITIES (not hard labels): the positive-class column for binary,
    the full (n, n_classes) matrix for multiclass -- consistent with ``explainability.py`` which scores on
    ``predict_proba``. A model without ``predict_proba`` (a regressor) falls back to ``predict``.
    """
    predictions = {}
    for model_name, model in tqdmu(trained_models.items(), desc="Getting raw predictions"):
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(X))
            # Binary -> positive-class column; multiclass -> full matrix. Shape guard so a degenerate
            # single-column proba does not crash on the [:, 1] index.
            prediction = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba
        else:
            prediction = model.predict(X)

        predictions[model_name] = prediction

    return predictions  # np.mean(np.array(predictions), axis=0)
