"""Prediction + load entry points for mlframe trained suites."""
from __future__ import annotations

import glob
import logging
import os
import pickle as _pickle
from collections import defaultdict
from copy import deepcopy
from os.path import exists, join

from scipy import stats
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import polars as pl
from pyutilz.strings import slugify

from ..configs import TargetTypes
from ..extractors import FeaturesAndTargetsExtractor
from ..io import load_mlframe_model
from ..pipeline import prepare_df_for_catboost
from ..utils import drop_columns_from_dataframe, get_pandas_view_of_polars_df
from .utils import (
    DEFAULT_PROBABILITY_THRESHOLD,
    _drop_cols_df,
    _setup_model_directories,
    _validate_input_columns_against_metadata,
    _validate_trusted_path,
)

logger = logging.getLogger(__name__)


def predict_mlframe_models_suite(
    df: pl.DataFrame | pd.DataFrame,
    models_path: str,
    features_and_targets_extractor: FeaturesAndTargetsExtractor | None = None,
    model_names: list[str] | None = None,
    return_probabilities: bool = True,
    verbose: int = 1,
    trusted_root: str | None = None,
) -> dict[str, Any]:
    """
    Generate predictions using a trained mlframe models suite.

    Loads the trained suite from disk and applies all required transformations
    to raw input data before generating predictions.

    Args:
        df: Input DataFrame (raw data, same format as training input)
        models_path: Path to the models directory (e.g., "data/models/target_name/model_name")
        features_and_targets_extractor: Optional extractor to preprocess input (same as training)
        model_names: Optional list of specific model names to use (None = all models)
        return_probabilities: If True, return probabilities; if False, return class predictions
        verbose: Verbosity level

    Returns:
        Dict with:
            - "predictions": Dict[model_name, predictions array]
            - "probabilities": Dict[model_name, probabilities array] (if return_probabilities)
            - "ensemble_predictions": Combined ensemble predictions (if multiple models)
            - "metadata": Loaded metadata dict
    """
    # Validate inputs
    if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
        raise TypeError(f"df must be pandas or polars DataFrame, got {type(df).__name__}")
    if len(df) == 0:
        raise ValueError("df cannot be empty")
    if not isinstance(models_path, str) or not os.path.isdir(models_path):
        raise ValueError(f"models_path must be a valid directory, got: {models_path}")

    results = {
        "predictions": {},
        "probabilities": {},
        "ensemble_predictions": None,
        "ensemble_probabilities": None,
        "metadata": None,
        "input_df": None,
    }

    # Prefer pickle-proto5 + zstd; fall back to legacy metadata.joblib for older saves.
    metadata_file_new = join(models_path, "metadata.pkl.zst")
    metadata_file_uncompressed = join(models_path, "metadata.pkl")
    metadata_file_legacy = join(models_path, "metadata.joblib")
    if exists(metadata_file_new):
        metadata_file = metadata_file_new
        loader_kind = "pkl.zst"
    elif exists(metadata_file_uncompressed):
        metadata_file = metadata_file_uncompressed
        loader_kind = "pkl"
    elif exists(metadata_file_legacy):
        metadata_file = metadata_file_legacy
        loader_kind = "joblib"
    else:
        raise FileNotFoundError(
            f"Metadata file not found in {models_path}; expected one of "
            f"metadata.pkl.zst, metadata.pkl, metadata.joblib"
        )

    if verbose:
        logger.info("Loading metadata from %s...", metadata_file)
    _root = trusted_root if trusted_root is not None else os.path.abspath(models_path)
    _validate_trusted_path(metadata_file, _root)
    if loader_kind == "pkl.zst":
        import pickle as _pickle
        import zstandard as _zstd
        _dctx = _zstd.ZstdDecompressor()
        with open(metadata_file, "rb") as _f:
            metadata = _pickle.loads(_dctx.decompress(_f.read()))
    elif loader_kind == "pkl":
        import pickle as _pickle
        with open(metadata_file, "rb") as _f:
            metadata = _pickle.load(_f)
    else:
        metadata = joblib.load(metadata_file)
    results["metadata"] = metadata

    pipeline = metadata.get("pipeline")
    metadata.get("columns", [])

    if verbose:
        logger.info("Preprocessing input data...")

    if features_and_targets_extractor is not None:
        df, _, _, _, _, _, columns_to_drop, _ = features_and_targets_extractor.transform(df)
        df = _drop_cols_df(df, columns_to_drop)

    if isinstance(df, pl.DataFrame):
        df = get_pandas_view_of_polars_df(df)

    df = _validate_input_columns_against_metadata(df, metadata, verbose=verbose)

    if pipeline is not None:
        if verbose:
            logger.info("Applying pipeline transformation...")
        df = pipeline.transform(df)

    results["input_df"] = df

    if verbose:
        logger.info("Loading and running models...")

    model_files = glob.glob(join(models_path, "**", "*.dump"), recursive=True)

    if not model_files:
        logger.warning(f"No model files found in {models_path}")
        return results

    all_probs = []
    all_preds = []

    for model_file in model_files:
        model_name = os.path.basename(model_file).replace(".dump", "")

        if model_names and model_name not in model_names:
            continue

        if verbose:
            logger.info("Loading model: %s", model_name)

        try:
            model_obj = load_mlframe_model(model_file)
            if model_obj is None:
                logger.warning(f"Failed to load model: {model_file}")
                continue

            model = model_obj.model if hasattr(model_obj, "model") else model_obj

            input_for_model = df
            if hasattr(model_obj, "pre_pipeline") and model_obj.pre_pipeline is not None:
                if model_obj.pre_pipeline != pipeline:
                    input_for_model = model_obj.pre_pipeline.transform(df)

            if return_probabilities and hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_for_model)
                results["probabilities"][model_name] = probs
                all_probs.append(probs)

                # Shape-aware decision rule: 1-D = sigmoid threshold; (N,2) = threshold class-1;
                # (N,K>2) = argmax. Multilabel cannot be inferred from shape; caller must hold that contract.
                if probs.ndim == 2:
                    if probs.shape[1] == 2:
                        preds = (probs[:, 1] > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
                    else:
                        preds = np.argmax(probs, axis=1)
                else:
                    preds = (probs > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
                results["predictions"][model_name] = preds
                all_preds.append(preds)
            else:
                preds = model.predict(input_for_model)
                results["predictions"][model_name] = preds
                all_preds.append(preds)

        except KeyboardInterrupt:
            raise
        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Error loading/predicting with model {model_file}: {e}")
            continue

    if len(all_probs) > 1:
        if verbose:
            logger.info("Computing ensemble predictions...")

        avg_probs = np.mean(np.stack(all_probs), axis=0)
        results["ensemble_probabilities"] = avg_probs
        # Expose the ensemble inside the per-model probabilities dict under the
        # canonical "ensemble" key so consumers can iterate one dict.
        if isinstance(results.get("probabilities"), dict):
            results["probabilities"]["ensemble"] = avg_probs

        if avg_probs.ndim == 2:
            if avg_probs.shape[1] == 2:
                ensemble_preds = (avg_probs[:, 1] > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
            else:
                ensemble_preds = np.argmax(avg_probs, axis=1)
        else:
            ensemble_preds = (avg_probs > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
        results["ensemble_predictions"] = ensemble_preds
        if isinstance(results.get("predictions"), dict):
            results["predictions"]["ensemble"] = ensemble_preds

    elif len(all_preds) > 1:
        # Majority voting when probabilities are unavailable.
        ensemble_preds, _ = stats.mode(np.stack(all_preds), axis=0)
        results["ensemble_predictions"] = ensemble_preds.flatten()

    elif len(all_preds) == 1:
        results["ensemble_predictions"] = all_preds[0]
        if all_probs:
            results["ensemble_probabilities"] = all_probs[0]

    if verbose:
        logger.info("Generated predictions for %d models", len(results['predictions']))

    return results


def predict_from_models(
    df: pl.DataFrame | pd.DataFrame,
    models: dict,
    metadata: dict,
    features_and_targets_extractor: FeaturesAndTargetsExtractor | None = None,
    return_probabilities: bool = True,
    verbose: int = 1,
) -> dict[str, Any]:
    """
    Generate predictions using in-memory models from train_mlframe_models_suite.

    This function works with models already in memory, avoiding disk I/O.
    Use this when you have the models dict returned by train_mlframe_models_suite.

    Args:
        df: Input DataFrame (raw data, same format as training input)
        models: Models dict returned by train_mlframe_models_suite.
            Structure: models[target_type][target_name] = [model_obj, ...]
        metadata: Metadata dict returned by train_mlframe_models_suite
        features_and_targets_extractor: Optional extractor to preprocess input (same as training)
        return_probabilities: If True, return probabilities; if False, return class predictions
        verbose: Verbosity level

    Returns:
        Dict with:
            - "predictions": Dict[model_name, predictions array]
            - "probabilities": Dict[model_name, probabilities array] (if return_probabilities)
            - "ensemble_predictions": Combined ensemble predictions (if multiple models)
            - "ensemble_probabilities": Averaged probabilities (if multiple models)
            - "models_used": List of model names that were used

    Example:
        ```python
        models, metadata = train_mlframe_models_suite(...)

        # Later, predict on new data
        results = predict_from_models(
            df=new_data,
            models=models,
            metadata=metadata,
            features_and_targets_extractor=ft_extractor,
        )
        print(results["ensemble_probabilities"])
        ```
    """
    # Validate inputs
    if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
        raise TypeError(f"df must be pandas or polars DataFrame, got {type(df).__name__}")
    if len(df) == 0:
        raise ValueError("df cannot be empty")

    results = {
        "predictions": {},
        "probabilities": {},
        "ensemble_predictions": None,
        "ensemble_probabilities": None,
        "models_used": [],
    }

    if verbose:
        logger.info("Preprocessing input data...")

    if features_and_targets_extractor is not None:
        df, _, _, _, _, _, columns_to_drop, _ = features_and_targets_extractor.transform(df)
        df = _drop_cols_df(df, columns_to_drop)

    if isinstance(df, pl.DataFrame):
        df = get_pandas_view_of_polars_df(df)

    metadata.get("columns", [])

    df = _validate_input_columns_against_metadata(df, metadata, verbose=verbose)

    pipeline = metadata.get("pipeline")
    if pipeline is not None:
        if verbose:
            logger.info("Applying pipeline transformation...")
        df = pipeline.transform(df)

    if verbose:
        logger.info("Running predictions on in-memory models...")

    all_probs = []
    all_preds = []

    for target_type, targets in models.items():
        for target_name, model_list in targets.items():
            for model_obj in model_list:
                if model_obj is None or not hasattr(model_obj, "model") or model_obj.model is None:
                    continue

                model_name = f"{target_type}_{target_name}"
                if hasattr(model_obj, "pre_pipeline") and model_obj.pre_pipeline is not None:
                    pipeline_name = type(model_obj.pre_pipeline).__name__
                    model_name = f"{model_name}_{pipeline_name}"

                # Disambiguate when multiple models in this target share a name.
                base_name = model_name
                counter = 1
                while model_name in results["predictions"]:
                    model_name = f"{base_name}_{counter}"
                    counter += 1

                if verbose:
                    logger.info("Predicting with model: %s", model_name)

                try:
                    model = model_obj.model

                    input_for_model = df
                    if hasattr(model_obj, "pre_pipeline") and model_obj.pre_pipeline is not None:
                        if model_obj.pre_pipeline != pipeline:
                            input_for_model = model_obj.pre_pipeline.transform(df)

                    if return_probabilities and hasattr(model, "predict_proba"):
                        probs = model.predict_proba(input_for_model)
                        results["probabilities"][model_name] = probs
                        all_probs.append(probs)

                        if probs.ndim == 2:
                            if probs.shape[1] == 2:
                                preds = (probs[:, 1] > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
                            else:
                                preds = np.argmax(probs, axis=1)
                        else:
                            preds = (probs > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
                        results["predictions"][model_name] = preds
                        all_preds.append(preds)
                    else:
                        preds = model.predict(input_for_model)
                        results["predictions"][model_name] = preds
                        all_preds.append(preds)

                    results["models_used"].append(model_name)

                except Exception as e:
                    logger.error(f"Error predicting with model {model_name}: {e}")
                    continue

    if len(all_probs) > 1:
        if verbose:
            logger.info("Computing ensemble predictions...")

        avg_probs = np.mean(np.stack(all_probs), axis=0)
        results["ensemble_probabilities"] = avg_probs
        if isinstance(results.get("probabilities"), dict):
            results["probabilities"]["ensemble"] = avg_probs

        if avg_probs.ndim == 2:
            if avg_probs.shape[1] == 2:
                ensemble_preds = (avg_probs[:, 1] > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
            else:
                ensemble_preds = np.argmax(avg_probs, axis=1)
        else:
            ensemble_preds = (avg_probs > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
        results["ensemble_predictions"] = ensemble_preds
        if isinstance(results.get("predictions"), dict):
            results["predictions"]["ensemble"] = ensemble_preds

    elif len(all_preds) > 1:
        # Majority voting when probabilities are unavailable.
        ensemble_preds, _ = stats.mode(np.stack(all_preds), axis=0)
        results["ensemble_predictions"] = ensemble_preds.flatten()

    elif len(all_preds) == 1:
        results["ensemble_predictions"] = all_preds[0]
        if all_probs:
            results["ensemble_probabilities"] = all_probs[0]

    if verbose:
        logger.info("Generated predictions for %d models", len(results['predictions']))

    return results


def load_mlframe_suite(models_path: str, trusted_root: str | None = None) -> tuple[dict, dict]:
    """
    Load a trained mlframe models suite from disk.

    Args:
        models_path: Path to the models directory (e.g., "data/models/target_name/model_name")

    Returns:
        Tuple of (models dict, metadata dict) in the same format as train_mlframe_models_suite:
        - models: Dict[target_type][target_name] = [model_obj, ...]
        - metadata: Dict with training configuration and artifacts
    """
    if not isinstance(models_path, str):
        raise TypeError(f"models_path must be string, got {type(models_path).__name__}")
    if not os.path.isdir(models_path):
        raise ValueError(f"models_path must be a valid directory, got: {models_path}")

    # Prefer pickle-proto5 + zstd; fall back to legacy metadata.joblib.
    metadata_file_new = join(models_path, "metadata.pkl.zst")
    metadata_file_uncompressed = join(models_path, "metadata.pkl")
    metadata_file_legacy = join(models_path, "metadata.joblib")
    if exists(metadata_file_new):
        metadata_file = metadata_file_new
        _kind = "pkl.zst"
    elif exists(metadata_file_uncompressed):
        metadata_file = metadata_file_uncompressed
        _kind = "pkl"
    elif exists(metadata_file_legacy):
        metadata_file = metadata_file_legacy
        _kind = "joblib"
    else:
        raise FileNotFoundError(
            f"Metadata file not found in {models_path}; expected one of "
            f"metadata.pkl.zst, metadata.pkl, metadata.joblib"
        )

    _root = trusted_root if trusted_root is not None else os.path.abspath(models_path)
    _validate_trusted_path(metadata_file, _root)
    if _kind == "pkl.zst":
        import pickle as _pickle
        import zstandard as _zstd
        _dctx = _zstd.ZstdDecompressor()
        with open(metadata_file, "rb") as _f:
            metadata = _pickle.loads(_dctx.decompress(_f.read()))
    elif _kind == "pkl":
        import pickle as _pickle
        with open(metadata_file, "rb") as _f:
            metadata = _pickle.load(_f)
    else:
        metadata = joblib.load(metadata_file)

    slug_to_original_target_type = metadata.get("slug_to_original_target_type", {})
    slug_to_original_target_name = metadata.get("slug_to_original_target_name", {})

    # Structure: models[target_type][target_name] = [model_obj, ...]
    # On-disk layout from _setup_model_directories: models_path/target_type/target_name/model.dump
    models = defaultdict(lambda: defaultdict(list))
    model_files = glob.glob(join(models_path, "**", "*.dump"), recursive=True)

    for model_file in model_files:
        rel_path = os.path.relpath(model_file, models_path)
        path_parts = rel_path.split(os.sep)

        if len(path_parts) >= 3:
            slugified_target_type = path_parts[0]
            slugified_target_name = path_parts[1]
            target_type = slug_to_original_target_type.get(slugified_target_type, slugified_target_type)
            target_name = slug_to_original_target_name.get(slugified_target_name, slugified_target_name)
        else:
            target_type = "unknown"
            target_name = "unknown"

        model_obj = load_mlframe_model(model_file)
        if model_obj is not None:
            models[target_type][target_name].append(model_obj)

    return dict(models), metadata


__all__ = [
    "predict_mlframe_models_suite",
    "predict_from_models",
    "load_mlframe_suite",
]
