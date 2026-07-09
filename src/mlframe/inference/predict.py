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

import os
import pandas as pd, numpy as np
from os.path import join, isfile, isdir, splitext

from pyutilz.system import tqdmu

from mlframe.utils.safe_pickle import (
    _sha256_of_file as _safe_pickle_sha256_of_file,
    verify_sidecar as _safe_pickle_verify_sidecar,
)
from mlframe.training.io import safe_joblib_load

# Allow-listed extensions for joblib model deserialization. Anything outside
# this set is skipped to make "drop a planted .pkl in the dir" attacks harder.
_ALLOWED_MODEL_EXTENSIONS = frozenset({".dump", ".joblib", ".pkl", ".pickle"})


def _sha256_of_file(path: str, chunk: int = 1 << 20) -> str:
    """Back-compat shim delegating to :func:`mlframe.utils.safe_pickle._sha256_of_file`."""
    return str(_safe_pickle_sha256_of_file(path, chunk=chunk))


def _verify_sidecar(path: str) -> bool:
    """Back-compat shim delegating to :func:`mlframe.utils.safe_pickle.verify_sidecar`."""
    return _safe_pickle_verify_sidecar(path)


def _model_feature_names(model) -> Optional[list]:
    """Return the fitted feature-name list a model exposes (sklearn-API ``feature_names_in_``, or
    CatBoost's own ``feature_names_``), normalised to a plain ``list[str]``. Returns None when the model
    exposes neither -- callers must treat that as "cannot validate", not "matches"."""
    if hasattr(model, "feature_names_in_"):
        names = model.feature_names_in_
    elif hasattr(model, "feature_names_"):
        names = model.feature_names_
    else:
        return None
    if isinstance(names, np.ndarray):
        names = names.tolist()
    return list(names)


def _check_model_feature_order(model, expected_features: list, context: str) -> bool:
    """Validate that ``model``'s own fitted feature-name attribute matches ``expected_features`` (order and
    names). Returns False (with an ERROR log) on a genuine mismatch, True when it matches or the model
    exposes no name attribute at all (logged as a WARN so an escaped model type is at least visible, per
    the "no name attribute => WARN, not silent pass" fix direction)."""
    names = _model_feature_names(model)
    if names is None:
        logger.warning(
            "model %s of type %s exposes neither feature_names_in_ nor feature_names_; " "column-order/name mismatch cannot be validated (%s)",
            model,
            type(model).__name__,
            context,
        )
        return True
    if names != expected_features:
        logger.error(
            "model %s was trained on different features %s than expected: %s (%s)",
            model,
            names,
            expected_features,
            context,
        )
        return False
    return True


def _load_features_file(features_file: str):
    """Prefer orjson-based sidecar (``features_file + '.json'`` or a ``.json`` twin),
    fall back to joblib. Both paths require a matching ``<candidate>.sha256`` sidecar (via
    ``_verify_sidecar``) before the content is trusted. Returns a list[str], or None when no features
    file / sidecar is present or the sha256 sidecar mismatches. Malformed sidecar JSON propagates the
    decode error -- the sole caller, ``read_trained_models``, wraps this call in try/except and logs a
    warning.
    """
    json_candidates = []
    base, ext = splitext(features_file)
    json_candidates.append(features_file + ".json")
    if ext.lower() != ".json":
        json_candidates.append(base + ".json")

    for candidate in json_candidates:
        if isfile(candidate):
            if not _verify_sidecar(candidate):
                logger.error("sha256 mismatch for features JSON sidecar %s; refusing to load", candidate)
                return None
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
    features = safe_joblib_load(features_file)
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

    models: dict = {}
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
        # audit5: a features file listing a column absent from X otherwise raised a cryptic KeyError. Surface a
        # clear, actionable error naming the missing columns.
        _missing = [c for c in features if c not in X.columns]
        if _missing:
            raise ValueError(
                f"read_trained_models: the features file for '{featureset}' lists columns absent from X: "
                f"{_missing[:10]}{' ...' if len(_missing) > 10 else ''}"
            )
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
        # Validate the .meta.json sidecar before unpickling so library-version drift gets a WARN log instead of
        # producing a cryptic AttributeError deep inside predict(). Non-fatal: legacy bundles (no sidecar) keep
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
            # Routed through the same _SafeUnpickler-derived module/class allowlist training/io.py uses for
            # dill bundles, so infer/ models get the RCE-surface restriction, not just the integrity gate.
            model = safe_joblib_load(model_file)
        except Exception as e:
            logger.warning("Could not read model file %s of featureset %s: %s", model_file, featureset, e)
            continue

        if not _check_model_feature_order(model, features, f"featureset {featureset}, file {model_file}"):
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

    ``read_trained_models`` already validates column order for models it loads, but this function is a public,
    directly-importable entry point with no caller-independent guard of its own -- it re-checks each model's own
    fitted feature-name attribute against ``X``'s columns so a caller that skips ``read_trained_models`` (builds
    ``trained_models``/``X`` itself) still gets the same silent-wrong-prediction protection.
    """
    predictions = {}
    expected_features = list(X.columns) if hasattr(X, "columns") else None
    for model_name, model in tqdmu(trained_models.items(), desc="Getting raw predictions"):
        if expected_features is not None:
            if not _check_model_feature_order(model, expected_features, f"model {model_name!r} in get_models_raw_predictions"):
                raise ValueError(f"get_models_raw_predictions: model {model_name!r} was trained on different features than X provides; refusing to predict")
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(X))
            # Binary -> positive-class column; multiclass -> full matrix. Shape guard so a degenerate
            # single-column proba does not crash on the [:, 1] index.
            prediction = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba
        else:
            prediction = model.predict(X)

        predictions[model_name] = prediction

    return predictions  # np.mean(np.array(predictions), axis=0)
