"""Prediction + load entry points for mlframe trained suites.

Pulled out of ``core.py`` (which still hosts the giant
``train_mlframe_models_suite`` orchestrator) so the predict / load surface
lives in its own ~550-LOC module. ``core.py`` re-exports each public
symbol below at its bottom for full back-compat.

Functions
---------
- :func:`predict_mlframe_models_suite` -- top-level predict entry that
  loads a trained suite from disk and runs predictions on raw data.
- :func:`predict_from_models` -- predict from an already-loaded
  ``(models, metadata)`` pair (no I/O).
- :func:`load_mlframe_suite` -- load a saved ``(models, metadata)`` pair
  from disk.
"""
from __future__ import annotations

import glob
import logging
import os
import pickle as _pickle
from copy import deepcopy
from os.path import exists, join
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
from ..utils import drop_columns_from_dataframe
from .utils import (
    DEFAULT_PROBABILITY_THRESHOLD,
    _drop_cols_df,
    _setup_model_directories,
    _validate_input_columns_against_metadata,
    _validate_trusted_path,
)

logger = logging.getLogger(__name__)


def predict_mlframe_models_suite(
    df: Union[pl.DataFrame, pd.DataFrame],
    models_path: str,
    features_and_targets_extractor: Optional[FeaturesAndTargetsExtractor] = None,
    model_names: Optional[List[str]] = None,
    return_probabilities: bool = True,
    verbose: int = 1,
    trusted_root: Optional[str] = None,
) -> Dict[str, Any]:
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
        "input_df": None,  # Transformed input DataFrame
    }

    # ==================================================================================
    # 1. LOAD METADATA
    # ==================================================================================

    # 2026-04-29: prefer the new pickle-proto5 + zstd format (8-47x
    # faster than ``joblib.dump`` on representative mlframe metadata)
    # but fall back to legacy ``metadata.joblib`` so saves from older
    # mlframe versions keep loading without manual migration.
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
    # Default trusted_root to the models directory if not provided -- matches the
    # in-process "we just wrote this file" flow while still refusing path escape.
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

    # Extract key components from metadata
    pipeline = metadata.get("pipeline")
    columns = metadata.get("columns", [])
    # Future enhancement: apply outlier_detector during inference to filter anomalous inputs
    # outlier_detector = metadata.get("outlier_detector")

    # ==================================================================================
    # 2. PREPROCESS INPUT DATA
    # ==================================================================================

    if verbose:
        logger.info("Preprocessing input data...")

    # Apply features extractor if provided (same transformation as training)
    if features_and_targets_extractor is not None:
        df, _, _, _, _, _, columns_to_drop, _ = features_and_targets_extractor.transform(df)
        # Drop extra columns (target, etc.) -- unified via helper.
        df = _drop_cols_df(df, columns_to_drop)

    # Convert to pandas if needed
    if isinstance(df, pl.DataFrame):
        df = get_pandas_view_of_polars_df(df)

    df = _validate_input_columns_against_metadata(df, metadata, verbose=verbose)

    # Apply pipeline transformation if available
    if pipeline is not None:
        if verbose:
            logger.info("Applying pipeline transformation...")
        df = pipeline.transform(df)

    results["input_df"] = df

    # ==================================================================================
    # 3. LOAD AND RUN MODELS
    # ==================================================================================

    if verbose:
        logger.info("Loading and running models...")

    # Find all model files
    model_files = glob.glob(join(models_path, "**", "*.dump"), recursive=True)

    if not model_files:
        logger.warning(f"No model files found in {models_path}")
        return results

    all_probs = []
    all_preds = []

    for model_file in model_files:
        model_name = os.path.basename(model_file).replace(".dump", "")

        # Filter by model_names if specified
        if model_names and model_name not in model_names:
            continue

        if verbose:
            logger.info("Loading model: %s", model_name)

        try:
            model_obj = load_mlframe_model(model_file)
            if model_obj is None:
                logger.warning(f"Failed to load model: {model_file}")
                continue

            # Get the underlying model
            model = model_obj.model if hasattr(model_obj, "model") else model_obj

            # Apply any model-specific pre_pipeline if different from main pipeline
            input_for_model = df
            if hasattr(model_obj, "pre_pipeline") and model_obj.pre_pipeline is not None:
                if model_obj.pre_pipeline != pipeline:
                    input_for_model = model_obj.pre_pipeline.transform(df)

            # Generate predictions
            if return_probabilities and hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_for_model)
                results["probabilities"][model_name] = probs
                all_probs.append(probs)

                # Shape-aware decision rule:
                # - 1-D probs: threshold (binary sigmoid output)
                # - (N, 2) binary: threshold on class-1 column
                # - (N, K) multiclass (K>2): argmax
                # Multilabel cannot be inferred from shape alone -- caller
                # must hold that contract; for now this path treats K>2 as
                # multiclass (the dominant case for ndim==2 K>2).
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
            raise  # Always allow user interruption
        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Error loading/predicting with model {model_file}: {e}")
            continue

    # ==================================================================================
    # 4. ENSEMBLE PREDICTIONS
    # ==================================================================================

    if len(all_probs) > 1:
        if verbose:
            logger.info("Computing ensemble predictions...")

        # Average probabilities
        avg_probs = np.mean(np.stack(all_probs), axis=0)
        results["ensemble_probabilities"] = avg_probs
        # Also expose the ensemble inside the per-model probabilities dict under
        # the canonical "ensemble" key so downstream consumers can iterate a
        # single dict and see both per-model and ensemble streams (2026-04-15).
        if isinstance(results.get("probabilities"), dict):
            results["probabilities"]["ensemble"] = avg_probs

        # Ensemble predictions from averaged probabilities (shape-aware).
        if avg_probs.ndim == 2:
            if avg_probs.shape[1] == 2:
                ensemble_preds = (avg_probs[:, 1] > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
            else:
                # Multiclass -- argmax across K classes
                ensemble_preds = np.argmax(avg_probs, axis=1)
        else:
            ensemble_preds = (avg_probs > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
        results["ensemble_predictions"] = ensemble_preds
        if isinstance(results.get("predictions"), dict):
            results["predictions"]["ensemble"] = ensemble_preds

    elif len(all_preds) > 1:
        # Majority voting for predictions without probabilities
        ensemble_preds, _ = stats.mode(np.stack(all_preds), axis=0)
        results["ensemble_predictions"] = ensemble_preds.flatten()

    elif len(all_preds) == 1:
        # Single model - use its predictions as ensemble
        results["ensemble_predictions"] = all_preds[0]
        if all_probs:
            results["ensemble_probabilities"] = all_probs[0]

    if verbose:
        logger.info("Generated predictions for %d models", len(results['predictions']))

    return results


def predict_from_models(
    df: Union[pl.DataFrame, pd.DataFrame],
    models: Dict,
    metadata: Dict,
    features_and_targets_extractor: Optional[FeaturesAndTargetsExtractor] = None,
    return_probabilities: bool = True,
    verbose: int = 1,
) -> Dict[str, Any]:
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

    # ==================================================================================
    # 1. PREPROCESS INPUT DATA
    # ==================================================================================

    if verbose:
        logger.info("Preprocessing input data...")

    # Apply features extractor if provided (same transformation as training)
    if features_and_targets_extractor is not None:
        df, _, _, _, _, _, columns_to_drop, _ = features_and_targets_extractor.transform(df)
        # Drop extra columns (target, etc.) -- unified via helper.
        df = _drop_cols_df(df, columns_to_drop)

    # Convert to pandas if needed
    if isinstance(df, pl.DataFrame):
        df = get_pandas_view_of_polars_df(df)

    # Get expected columns from metadata
    columns = metadata.get("columns", [])

    df = _validate_input_columns_against_metadata(df, metadata, verbose=verbose)

    # Apply main pipeline transformation if available
    pipeline = metadata.get("pipeline")
    if pipeline is not None:
        if verbose:
            logger.info("Applying pipeline transformation...")
        df = pipeline.transform(df)

    # ==================================================================================
    # 2. RUN PREDICTIONS
    # ==================================================================================

    if verbose:
        logger.info("Running predictions on in-memory models...")

    all_probs = []
    all_preds = []

    for target_type, targets in models.items():
        for target_name, model_list in targets.items():
            for model_obj in model_list:
                if model_obj is None or not hasattr(model_obj, "model") or model_obj.model is None:
                    continue

                # Generate a unique name for this model
                model_name = f"{target_type}_{target_name}"
                if hasattr(model_obj, "pre_pipeline") and model_obj.pre_pipeline is not None:
                    # Add pipeline info to name if present
                    pipeline_name = type(model_obj.pre_pipeline).__name__
                    model_name = f"{model_name}_{pipeline_name}"

                # Avoid duplicate names
                base_name = model_name
                counter = 1
                while model_name in results["predictions"]:
                    model_name = f"{base_name}_{counter}"
                    counter += 1

                if verbose:
                    logger.info("Predicting with model: %s", model_name)

                try:
                    model = model_obj.model

                    # Apply model-specific pre_pipeline if present and different from main pipeline
                    input_for_model = df
                    if hasattr(model_obj, "pre_pipeline") and model_obj.pre_pipeline is not None:
                        if model_obj.pre_pipeline != pipeline:
                            input_for_model = model_obj.pre_pipeline.transform(df)

                    # Generate predictions
                    if return_probabilities and hasattr(model, "predict_proba"):
                        probs = model.predict_proba(input_for_model)
                        results["probabilities"][model_name] = probs
                        all_probs.append(probs)

                        # Shape-aware decision rule (see predict_mlframe_models_suite).
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

    # ==================================================================================
    # 3. ENSEMBLE PREDICTIONS
    # ==================================================================================

    if len(all_probs) > 1:
        if verbose:
            logger.info("Computing ensemble predictions...")

        # Average probabilities
        avg_probs = np.mean(np.stack(all_probs), axis=0)
        results["ensemble_probabilities"] = avg_probs
        # Also expose the ensemble inside the per-model probabilities dict under
        # the canonical "ensemble" key so downstream consumers can iterate a
        # single dict and see both per-model and ensemble streams (2026-04-15).
        if isinstance(results.get("probabilities"), dict):
            results["probabilities"]["ensemble"] = avg_probs

        # Ensemble predictions from averaged probabilities (shape-aware).
        if avg_probs.ndim == 2:
            if avg_probs.shape[1] == 2:
                ensemble_preds = (avg_probs[:, 1] > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
            else:
                # Multiclass -- argmax across K classes
                ensemble_preds = np.argmax(avg_probs, axis=1)
        else:
            ensemble_preds = (avg_probs > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
        results["ensemble_predictions"] = ensemble_preds
        if isinstance(results.get("predictions"), dict):
            results["predictions"]["ensemble"] = ensemble_preds

    elif len(all_preds) > 1:
        # Majority voting for predictions without probabilities
        ensemble_preds, _ = stats.mode(np.stack(all_preds), axis=0)
        results["ensemble_predictions"] = ensemble_preds.flatten()

    elif len(all_preds) == 1:
        # Single model - use its predictions as ensemble
        results["ensemble_predictions"] = all_preds[0]
        if all_probs:
            results["ensemble_probabilities"] = all_probs[0]

    if verbose:
        logger.info("Generated predictions for %d models", len(results['predictions']))

    return results


def load_mlframe_suite(models_path: str, trusted_root: Optional[str] = None) -> Tuple[Dict, Dict]:
    """
    Load a trained mlframe models suite from disk.

    Args:
        models_path: Path to the models directory (e.g., "data/models/target_name/model_name")

    Returns:
        Tuple of (models dict, metadata dict) in the same format as train_mlframe_models_suite:
        - models: Dict[target_type][target_name] = [model_obj, ...]
        - metadata: Dict with training configuration and artifacts
    """
    # Validate inputs
    if not isinstance(models_path, str):
        raise TypeError(f"models_path must be string, got {type(models_path).__name__}")
    if not os.path.isdir(models_path):
        raise ValueError(f"models_path must be a valid directory, got: {models_path}")

    # 2026-04-29: prefer the new pickle-proto5 + zstd format, fall
    # back to legacy ``metadata.joblib`` for backward compat.
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

    # Get slug-to-original name mappings from metadata (if available)
    slug_to_original_target_type = metadata.get("slug_to_original_target_type", {})
    slug_to_original_target_name = metadata.get("slug_to_original_target_name", {})

    # Load all models into nested structure matching train_mlframe_models_suite output
    # Structure: models[target_type][target_name] = [model_obj, ...]
    # Path structure from _setup_model_directories: models_path/target_type/target_name/model.dump
    models = defaultdict(lambda: defaultdict(list))
    model_files = glob.glob(join(models_path, "**", "*.dump"), recursive=True)

    for model_file in model_files:
        # Extract target_type and target_name from path
        rel_path = os.path.relpath(model_file, models_path)
        path_parts = rel_path.split(os.sep)

        if len(path_parts) >= 3:
            # path_parts = [slugified_target_type, slugified_target_name, model_file.dump]
            slugified_target_type = path_parts[0]
            slugified_target_name = path_parts[1]

            # Restore original names from metadata mappings
            target_type = slug_to_original_target_type.get(slugified_target_type, slugified_target_type)
            target_name = slug_to_original_target_name.get(slugified_target_name, slugified_target_name)
        else:
            # Fallback for flat structure or unexpected layout
            target_type = "unknown"
            target_name = "unknown"

        model_obj = load_mlframe_model(model_file)
        if model_obj is not None:
            models[target_type][target_name].append(model_obj)

    return dict(models), metadata


__all__ = [
    "train_mlframe_models_suite",
    "predict_mlframe_models_suite",
    "predict_from_models",
    "load_mlframe_suite",
]
