"""
Core training functions for mlframe.

Contains the refactored train_mlframe_models_suite function.
"""

# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------


import os
import psutil
import joblib
import pandas as pd
import polars as pl
from typing import Union, Optional, List, Dict, Any, Tuple, TypeVar
from copy import deepcopy
import numpy as np
import scipy.stats as stats
from collections import defaultdict
from os.path import join, exists
import glob
from pyutilz.system import clean_ram, tqdmu
from pyutilz.strings import slugify
from sklearn.pipeline import Pipeline
import category_encoders as ce

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from pyutilz.system import ensure_dir_exists

from .configs import (
    PreprocessingConfig,
    TrainingSplitConfig,
    PolarsPipelineConfig,
    TrainingConfig,
    TargetTypes,
    LinearModelConfig,
)
from .preprocessing import (
    load_and_prepare_dataframe,
    preprocess_dataframe,
    save_split_artifacts,
    create_split_dataframes,
)
from .pipeline import fit_and_transform_pipeline
from mlframe.feature_selection.filters import MRMR
from .utils import log_ram_usage, log_phase, drop_columns_from_dataframe, get_pandas_view_of_polars_df
from .helpers import get_trainset_features_stats_polars
from .models import is_linear_model, LINEAR_MODEL_TYPES
from .strategies import get_strategy, PipelineCache
from .io import load_mlframe_model
from .splitting import make_train_test_split

# Extractors from new module
from .extractors import FeaturesAndTargetsExtractor

# score_ensemble is in ensembling module
from ..ensembling import score_ensemble

# Training execution functions from train_eval module
from .train_eval import process_model, select_target

# Fairness subgroups creation
from mlframe.metrics import create_fairness_subgroups

# Constants
DEFAULT_PROBABILITY_THRESHOLD = 0.5


# Type variable for config classes
ConfigT = TypeVar("ConfigT")


def _ensure_config(
    config: Union[ConfigT, Dict[str, Any], None],
    config_class: type,
    kwargs: Dict[str, Any],
) -> ConfigT:
    """
    Convert dict/None to Pydantic config object.

    Args:
        config: Config object, dict, or None
        config_class: Pydantic config class to instantiate
        kwargs: Keyword arguments to extract config fields from

    Returns:
        Pydantic config object of type config_class
    """
    if isinstance(config, dict):
        return config_class(**config)
    elif config is None:
        # Extract only fields that belong to this config class
        return config_class(**{k: v for k, v in kwargs.items() if k in config_class.model_fields})
    return config


def _apply_outlier_detection_global(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    train_idx: np.ndarray,
    val_idx: Optional[np.ndarray],
    outlier_detector: Any,
    od_val_set: bool,
    verbose: bool,
) -> Tuple[
    pd.DataFrame,
    Optional[pd.DataFrame],
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """
    Apply outlier detection ONCE globally (unsupervised - no target needed).

    This function fits the outlier detector on training features only and applies
    it to both training and validation sets. Since standard sklearn outlier detectors
    (IsolationForest, LOF, OneClassSVM) are unsupervised, no target is needed.

    Args:
        train_df: Training DataFrame (features only)
        val_df: Validation DataFrame (can be None)
        train_idx: Training indices
        val_idx: Validation indices (can be None)
        outlier_detector: Outlier detector object (e.g., IsolationForest pipeline)
        od_val_set: Whether to apply outlier detection to validation set
        verbose: Whether to log information

    Returns:
        Tuple of:
        - filtered_train_df: Filtered training DataFrame (inliers only)
        - filtered_val_df: Filtered validation DataFrame (or original if no OD on val)
        - filtered_train_idx: Filtered training indices
        - filtered_val_idx: Filtered validation indices
        - train_od_idx: Boolean mask of inliers in training set
        - val_od_idx: Boolean mask of inliers in validation set
    """
    if outlier_detector is None:
        return train_df, val_df, train_idx, val_idx, None, None

    if verbose:
        logger.info("Fitting outlier detector (once for all targets)...")

    # Fit on training features only (unsupervised - no target needed)
    outlier_detector.fit(train_df)

    # Predict on training set
    is_inlier = outlier_detector.predict(train_df)
    train_od_idx = is_inlier == 1

    filtered_train_df = train_df
    filtered_train_idx = train_idx

    train_kept = train_od_idx.sum()
    if train_kept < len(train_df):
        logger.info(f"Outlier rejection: {len(train_df):_} train samples -> {train_kept:_} kept.")
        filtered_train_df = train_df.loc[train_od_idx]
        filtered_train_idx = train_idx[train_od_idx]

    # Predict on validation set if requested
    filtered_val_df = val_df
    filtered_val_idx = val_idx
    val_od_idx = None

    if val_df is not None and od_val_set:
        is_inlier = outlier_detector.predict(val_df)
        val_od_idx = is_inlier == 1
        val_kept = val_od_idx.sum()
        if val_kept < len(val_df):
            logger.info(f"Outlier rejection: {len(val_df):_} val samples -> {val_kept:_} kept.")
            filtered_val_df = val_df.loc[val_od_idx]
            filtered_val_idx = val_idx[val_od_idx]

    clean_ram()
    if verbose:
        log_ram_usage()

    return (filtered_train_df, filtered_val_df, filtered_train_idx, filtered_val_idx, train_od_idx, val_od_idx)


def _setup_model_directories(
    target_name: str,
    model_name: str,
    target_type: str,
    cur_target_name: str,
    data_dir: Optional[str],
    models_dir: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Set up directories for model artifacts and charts.

    Args:
        target_name: Name of the target being trained
        model_name: Name of the model configuration
        target_type: Type of target (regression/classification)
        cur_target_name: Current target name within the target type
        data_dir: Base data directory
        models_dir: Models subdirectory

    Returns:
        Tuple of (plot_file, model_file) paths, either can be None
    """
    parts = slugify(target_name), slugify(model_name), slugify(target_type.lower()), slugify(cur_target_name)

    if data_dir is not None:
        plot_file = join(data_dir, "charts", *parts) + os.path.sep
        ensure_dir_exists(plot_file)
    else:
        plot_file = None

    if data_dir is not None and models_dir is not None:
        model_file = join(data_dir, models_dir, *parts) + os.path.sep
        ensure_dir_exists(model_file)
    else:
        model_file = None

    return plot_file, model_file


def _build_common_params_for_target(
    init_common_params: Dict[str, Any],
    trainset_features_stats: Optional[Dict],
    plot_file: Optional[str],
    train_od_idx: Optional[np.ndarray],
    val_od_idx: Optional[np.ndarray],
    current_train_target: Optional[Any],
    current_val_target: Optional[Any],
    outlier_detector: Optional[Any],
    control_params_override: Optional[Dict[str, Any]],
    fairness_subgroups: Optional[Dict],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build common_params and control_params_override for select_target call.

    Args:
        init_common_params: Initial common parameters from caller
        trainset_features_stats: Computed feature statistics
        plot_file: Path for saving plots
        train_od_idx: Outlier detection indices for training set
        val_od_idx: Outlier detection indices for validation set
        current_train_target: Training targets (filtered if OD applied)
        current_val_target: Validation targets (filtered if OD applied)
        outlier_detector: Outlier detector object (or None)
        control_params_override: Control parameters override dict
        fairness_subgroups: Pre-computed fairness subgroups

    Returns:
        Tuple containing:
            - od_common_params: Dict with common parameters for model training, including
              trainset_features_stats, plot_file, OD indices, and optionally train/val targets.
            - current_control_override: Dict with control overrides, including any
              pre-computed fairness subgroups under "_precomputed_fairness_subgroups" key.
    """
    # Add pre-computed fairness subgroups to control_params_override
    current_control_override = control_params_override.copy() if control_params_override else {}
    if fairness_subgroups is not None:
        current_control_override["_precomputed_fairness_subgroups"] = fairness_subgroups

    # Build common_params dict
    # Filter out train_target/val_target from init_common_params to avoid conflict when OD is applied
    filtered_init_params = {k: v for k, v in init_common_params.items() if k not in ("train_target", "val_target")}
    od_common_params = dict(
        trainset_features_stats=trainset_features_stats,
        plot_file=plot_file,
        train_od_idx=train_od_idx,  # Pass for metadata
        val_od_idx=val_od_idx,  # Pass for metadata
        **filtered_init_params,
    )

    # When outlier detection is applied, pass targets directly to avoid re-subsetting
    if outlier_detector is not None:
        od_common_params["train_target"] = current_train_target
        if current_val_target is not None:
            od_common_params["val_target"] = current_val_target

    return od_common_params, current_control_override


def _build_pre_pipelines(
    use_ordinary_models: bool,
    rfecv_models: List[str],
    rfecv_models_params: Dict[str, Any],
    use_mrmr_fs: bool,
    mrmr_kwargs: Dict[str, Any],
) -> Tuple[List[Any], List[str]]:
    """
    Build lists of pre-pipelines and their names for feature selection.

    Args:
        use_ordinary_models: Whether to include no-pipeline (ordinary) models
        rfecv_models: List of RFECV model names to use
        rfecv_models_params: Dict mapping RFECV model names to their pipeline configurations
        use_mrmr_fs: Whether to include MRMR feature selection
        mrmr_kwargs: Keyword arguments for MRMR

    Returns:
        Tuple of (pre_pipelines list, pre_pipeline_names list)
    """
    pre_pipelines = []
    pre_pipeline_names = []

    # Add ordinary models (no pre-pipeline)
    if use_ordinary_models:
        pre_pipelines.append(None)
        pre_pipeline_names.append("")

    # Add RFECV-based feature selection pipelines
    # Validate all RFECV models exist before processing
    unknown_rfecv_models = [m for m in rfecv_models if m not in rfecv_models_params]
    if unknown_rfecv_models:
        raise ValueError(f"Unknown RFECV model(s): {unknown_rfecv_models}. " f"Available: {list(rfecv_models_params.keys())}")
    for rfecv_model_name in rfecv_models:
        pre_pipelines.append(rfecv_models_params[rfecv_model_name])
        pre_pipeline_names.append(f"{rfecv_model_name} ")

    # Add MRMR feature selection pipeline
    if use_mrmr_fs:
        pre_pipelines.append(MRMR(**mrmr_kwargs))
        pre_pipeline_names.append("MRMR ")

    return pre_pipelines, pre_pipeline_names


def _build_process_model_kwargs(
    model_file: str,
    model_name_with_weight: str,
    target_type: TargetTypes,
    pre_pipeline: Any,
    pre_pipeline_name: str,
    cur_target_name: str,
    models: Dict,
    model_params: Dict[str, Any],
    common_params: Dict[str, Any],
    ens_models: Optional[List],
    trainset_features_stats: Optional[Dict],
    verbose: int,
    cached_dfs: Optional[Tuple],
    polars_pipeline_applied: bool = False,
    mlframe_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build kwargs dictionary for process_model call.

    Args:
        model_file: Path to save the model.
        model_name_with_weight: Model name with weight suffix.
        target_type: Type of target (regression or classification).
        pre_pipeline: Pre-processing pipeline.
        pre_pipeline_name: Name of the pre-pipeline.
        cur_target_name: Current target name.
        models: Dict to store trained models.
        model_params: Model-specific parameters.
        common_params: Common parameters for training.
        ens_models: List for ensemble models.
        trainset_features_stats: Feature statistics from training set.
        verbose: Verbosity level.
        cached_dfs: Tuple of cached (train_df, val_df, test_df) or None.
        polars_pipeline_applied: Whether Polars-ds pipeline was already applied globally.
        mlframe_model_name: Short model type name (cb, xgb, lgb, etc.) for early stopping setup.

    Returns:
        Dict of kwargs for process_model call.
    """
    # Add model_category to common_params for early stopping callback setup
    if mlframe_model_name:
        common_params = common_params.copy()
        common_params["model_category"] = mlframe_model_name

    kwargs = {
        "model_file": model_file,
        "model_name": model_name_with_weight,
        "target_type": target_type,
        "pre_pipeline": pre_pipeline,
        "pre_pipeline_name": pre_pipeline_name,
        "cur_target_name": cur_target_name,
        "models": models,
        "model_params": model_params,
        "common_params": common_params,
        "ens_models": ens_models,
        "trainset_features_stats": trainset_features_stats,
        "verbose": verbose,
    }

    # Skip pre_pipeline transform if Polars-ds pipeline was already applied globally
    # (data is already preprocessed - applying SimpleImputer/StandardScaler again would be redundant)
    if polars_pipeline_applied:
        kwargs["skip_pre_pipeline_transform"] = True

    # Add cached DataFrames if available (also skips pre_pipeline transform)
    if cached_dfs is not None:
        kwargs.update(
            {
                "skip_pre_pipeline_transform": True,
                "cached_train_df": cached_dfs[0],
                "cached_val_df": cached_dfs[1],
                "cached_test_df": cached_dfs[2],
            }
        )

    return kwargs


def _convert_dfs_to_pandas(
    train_df: Union[pd.DataFrame, pl.DataFrame],
    val_df: Optional[Union[pd.DataFrame, pl.DataFrame]],
    test_df: Optional[Union[pd.DataFrame, pl.DataFrame]],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Convert DataFrames to pandas format (zero-copy for Polars).

    Args:
        train_df: Training DataFrame (pandas or polars).
        val_df: Validation DataFrame (pandas or polars) or None.
        test_df: Test DataFrame (pandas or polars) or None.

    Returns:
        Tuple of (train_df_pd, val_df_pd, test_df_pd)

    Raises:
        TypeError: If any DataFrame is not pandas, polars, or None.
    """
    # Validate input types
    for name, df in [("train_df", train_df), ("val_df", val_df), ("test_df", test_df)]:
        if df is not None and not isinstance(df, (pd.DataFrame, pl.DataFrame)):
            raise TypeError(f"{name} must be pandas DataFrame, polars DataFrame, or None, got {type(df).__name__}")

    train_df_pd = train_df if isinstance(train_df, pd.DataFrame) else get_pandas_view_of_polars_df(train_df)
    val_df_pd = val_df if val_df is None or isinstance(val_df, pd.DataFrame) else get_pandas_view_of_polars_df(val_df)
    test_df_pd = test_df if test_df is None or isinstance(test_df, pd.DataFrame) else get_pandas_view_of_polars_df(test_df)

    return train_df_pd, val_df_pd, test_df_pd


def _get_pipeline_components(
    init_common_params: Optional[Dict[str, Any]],
    cat_features: List[str],
) -> Tuple[Optional[Any], SimpleImputer, StandardScaler]:
    """
    Get pipeline components (category_encoder, imputer, scaler) from params or defaults.

    Args:
        init_common_params: Initial common parameters that may contain pipeline components.
            Recognized keys: "category_encoder", "imputer", "scaler".
        cat_features: List of categorical feature names.

    Returns:
        Tuple containing:
            - category_encoder: Encoder for categorical features (e.g., CatBoostEncoder),
              or None if no categorical features exist.
            - imputer: SimpleImputer instance for handling missing values.
            - scaler: StandardScaler instance for feature normalization.
    """
    if init_common_params is None:
        init_common_params = {}

    category_encoder = init_common_params.get("category_encoder", None)
    imputer = init_common_params.get("imputer", None)
    scaler = init_common_params.get("scaler", None)

    # Initialize defaults if not provided
    if category_encoder is None and cat_features:
        category_encoder = ce.CatBoostEncoder()

    if imputer is None:
        imputer = SimpleImputer()

    if scaler is None:
        scaler = StandardScaler()

    return category_encoder, imputer, scaler


def _compute_fairness_subgroups(
    df: Union[pd.DataFrame, pl.DataFrame],
    control_params_override: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict], List[str]]:
    """
    Compute fairness subgroups from DataFrame if fairness_features are specified.

    Args:
        df: Full DataFrame (before splitting).
        control_params_override: Control params that may contain fairness_features.

    Returns:
        Tuple of (fairness subgroups dict or None, list of fairness feature names).
    """
    fairness_features = (control_params_override or {}).get("fairness_features", [])
    if not fairness_features:
        return None, fairness_features

    # Only select columns that are actually needed - massive memory savings for large DataFrames
    # (create_fairness_subgroups only accesses columns listed in features parameter)
    cols_to_select = [f for f in fairness_features if f not in ("**ORDER**", "**RANDOM**") and f in df.columns]

    if cols_to_select:
        if isinstance(df, pl.DataFrame):
            df_subset = df.select(cols_to_select).to_pandas()
        else:
            df_subset = df[cols_to_select]
    else:
        # Only special markers like **ORDER**, **RANDOM** - no actual columns needed
        df_subset = pd.DataFrame(index=range(len(df)))

    subgroups = create_fairness_subgroups(
        df_subset,
        features=fairness_features,
        cont_nbins=control_params_override.get("cont_nbins", 6),
        min_pop_cat_thresh=control_params_override.get("fairness_min_pop_cat_thresh", 1000),
    )
    return subgroups, fairness_features


def _should_skip_catboost_metamodel(
    model_or_pipeline_name: str,
    target_type: TargetTypes,
    control_params_override: Dict[str, Any],
) -> bool:
    """
    Check if CatBoost model should be skipped due to metamodel_func incompatibility.

    CatBoost regression with metamodel_func causes sklearn clone issues:
    RuntimeError: Cannot clone object <catboost.core.CatBoostRegressor...>,
    as the constructor either does not set or modifies parameter custom_metric.

    Args:
        model_or_pipeline_name: Model name or pre-pipeline name to check.
        target_type: Type of target (regression or classification).
        control_params_override: Control params that may contain metamodel_func.

    Returns:
        True if this combination should be skipped.
    """
    if target_type != TargetTypes.REGRESSION:
        return False
    if control_params_override.get("metamodel_func") is None:
        return False
    # Check if name contains 'cb' (for model names like 'cb') or 'cb_rfecv' (for pipelines)
    return model_or_pipeline_name in ("cb", "cb_rfecv")


def _create_initial_metadata(
    model_name: str,
    target_name: str,
    mlframe_models: List[str],
    preprocessing_config: PreprocessingConfig,
    pipeline_config: PolarsPipelineConfig,
    split_config: TrainingSplitConfig,
) -> Dict[str, Any]:
    """
    Create the initial metadata dictionary for tracking training.

    Args:
        model_name: Name of the model configuration.
        target_name: Name of the target being trained.
        mlframe_models: List of model types to train.
        preprocessing_config: Preprocessing configuration.
        pipeline_config: Pipeline configuration.
        split_config: Split configuration.

    Returns:
        Initial metadata dictionary.
    """
    return {
        "model_name": model_name,
        "target_name": target_name,
        "mlframe_models": mlframe_models,
        "configs": {
            "preprocessing": preprocessing_config,
            "pipeline": pipeline_config,
            "split": split_config,
        },
    }


def _initialize_training_defaults(
    init_common_params: Optional[Dict[str, Any]],
    rfecv_models: Optional[List[str]],
    mrmr_kwargs: Optional[Dict[str, Any]],
    config_params: Optional[Dict[str, Any]],
    control_params: Optional[Dict[str, Any]],
    config_params_override: Optional[Dict[str, Any]],
    control_params_override: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[str], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Initialize default values for training parameters.

    Args:
        init_common_params: Initial common parameters (can be None).
        rfecv_models: List of RFECV models (can be None).
        mrmr_kwargs: MRMR keyword arguments (can be None).
        config_params: Config parameters (can be None).
        control_params: Control parameters (can be None).
        config_params_override: Config params override (can be None).
        control_params_override: Control params override (can be None).

    Returns:
        Tuple of initialized values:
        - init_common_params: Dict (never None)
        - rfecv_models: List (never None)
        - mrmr_kwargs: Dict (never None)
        - config_params: Dict (never None)
        - control_params: Dict (never None)
        - config_params_override: Dict (never None)
        - control_params_override: Dict (never None)
    """
    if init_common_params is None:
        init_common_params = {}

    if rfecv_models is None:
        rfecv_models = []

    if mrmr_kwargs is None:
        mrmr_kwargs = dict(
            n_workers=max(1, psutil.cpu_count(logical=False)),
            verbose=2,
            fe_max_steps=0,
        )

    if config_params is None:
        config_params = {}

    if control_params is None:
        control_params = {}

    if config_params_override is None:
        config_params_override = {}

    if control_params_override is None:
        control_params_override = {}

    return (
        init_common_params,
        rfecv_models,
        mrmr_kwargs,
        config_params,
        control_params,
        config_params_override,
        control_params_override,
    )


def _finalize_and_save_metadata(
    metadata: Dict[str, Any],
    outlier_detector: Optional[Any],
    outlier_detection_result: Dict,
    trainset_features_stats: Optional[Dict],
    data_dir: str,
    models_dir: str,
    target_name: str,
    model_name: str,
    verbose: int,
) -> None:
    """
    Finalize and save metadata to disk.

    Args:
        metadata: Dict to update with final values.
        outlier_detector: Outlier detector object.
        outlier_detection_result: Global outlier detection result (train_od_idx, val_od_idx).
        trainset_features_stats: Feature statistics from training set.
        data_dir: Base data directory.
        models_dir: Models subdirectory.
        target_name: Target name for path construction.
        model_name: Model name for path construction.
        verbose: Verbosity level.
    """
    # Add shared objects to metadata
    metadata.update(
        {
            "outlier_detector": outlier_detector,
            "outlier_detection_result": outlier_detection_result,
            "trainset_features_stats": trainset_features_stats,
        }
    )

    # Save metadata
    if data_dir and models_dir:
        metadata_file = join(data_dir, models_dir, slugify(target_name), slugify(model_name), "metadata.joblib")
        try:
            joblib.dump(metadata, metadata_file)
            if verbose:
                logger.info(f"Saved metadata to {metadata_file}")
        except (OSError, IOError) as e:
            logger.error(f"Failed to save metadata to {metadata_file}: {e}")
            raise


def train_mlframe_models_suite(
    df: Union[pl.DataFrame, pd.DataFrame, str],
    target_name: str,
    model_name: str,
    features_and_targets_extractor: FeaturesAndTargetsExtractor,
    # Model selection
    mlframe_models: Optional[List[str]] = None,
    use_ordinary_models: bool = True,
    use_mlframe_ensembles: bool = True,
    # Configurations (can be dicts or Pydantic objects)
    preprocessing_config: Optional[Union[PreprocessingConfig, Dict]] = None,
    split_config: Optional[Union[TrainingSplitConfig, Dict]] = None,
    pipeline_config: Optional[Union[PolarsPipelineConfig, Dict]] = None,
    # Model-specific configurations
    linear_model_config: Optional[LinearModelConfig] = None,
    # Feature selection
    use_mrmr_fs: bool = False,
    mrmr_kwargs: Optional[Dict] = None,
    rfecv_models: Optional[List[str]] = None,
    # Override parameters (for backward compatibility)
    config_params: Optional[Dict] = None,
    control_params: Optional[Dict] = None,
    config_params_override: Optional[Dict] = None,
    control_params_override: Optional[Dict] = None,
    init_common_params: Optional[Dict] = None,
    # Paths
    data_dir: str = "",
    models_dir: str = "models",
    # Misc
    verbose: int = 1,
    # Outlier detection (run once for all models)
    outlier_detector: Optional[Any] = None,
    od_val_set: bool = True,
    # Backward compatibility parameters
    **kwargs,
) -> Tuple[Dict, Dict]:
    """
    Train a suite of ML models on a dataset.

    This is the refactored main training function with cleaner, more modular code.

    Args:
        df: DataFrame or path to parquet file
        target_name: Name of the target to predict
        model_name: Base name for the models
        features_and_targets_extractor: FeaturesAndTargetsExtractor instance for computing targets

        mlframe_models: List of model types to train (cb, lgb, xgb, mlp, hgb, linear, ridge, etc.)
        use_ordinary_models: Whether to train regular models
        use_mlframe_ensembles: Whether to create ensembles

        preprocessing_config: Preprocessing configuration
        split_config: Train/val/test split configuration
        pipeline_config: Pipeline configuration

        use_mrmr_fs: Whether to use MRMR feature selection
        mrmr_kwargs: MRMR parameters
        rfecv_models: Models to use for RFECV

        config_params: Base model configuration parameters (legacy, prefer Pydantic configs)
        control_params: Base control parameters (legacy, prefer Pydantic configs)
        config_params_override: Override for model config parameters (highest priority)
        control_params_override: Override for control parameters (highest priority)
        init_common_params: Common initialization parameters (pipeline components, etc.)

        data_dir: Directory for saving artifacts
        models_dir: Directory for saving models

        verbose: Verbosity level (0=silent, 1=info, 2=debug)

    Returns:
        Tuple of (models_dict, metadata_dict)

    Note:
        Parameter precedence (highest to lowest):
        1. config_params_override / control_params_override (explicit overrides, always win)
        2. config_params / control_params (user-provided base configuration)
        3. Pydantic config objects (preprocessing_config, split_config, etc.)
        4. Built-in defaults

    Example:
        ```python
        models, metadata = train_mlframe_models_suite(
            df="data.parquet",
            target_name="target",
            model_name="experiment_1",
            features_and_targets_extractor=my_ft_extractor,
            mlframe_models=["linear", "ridge", "cb", "lgb"],
            preprocessing_config=PreprocessingConfig(fillna_value=0.0),
            split_config=TrainingSplitConfig(test_size=0.1, val_size=0.1),
        )
        ```
    """

    # ==================================================================================
    # 0. INPUT VALIDATION
    # ==================================================================================

    # Validate df parameter
    if not isinstance(df, (pd.DataFrame, pl.DataFrame, str)):
        raise TypeError(f"df must be pandas DataFrame, polars DataFrame, or path string, " f"got {type(df).__name__}")
    if isinstance(df, str) and not df.lower().endswith(".parquet"):
        raise ValueError(f"File path must be a .parquet file, got: {df}")

    # Validate required parameters
    if not target_name:
        raise ValueError("target_name cannot be empty")
    if not model_name:
        raise ValueError("model_name cannot be empty")
    if features_and_targets_extractor is None:
        raise ValueError("features_and_targets_extractor is required")

    # ==================================================================================
    # 1. CONFIGURATION SETUP
    # ==================================================================================

    if verbose:
        log_phase(f"Starting mlframe training suite: {model_name}")

    # Convert dict configs to Pydantic if needed
    preprocessing_config = _ensure_config(preprocessing_config, PreprocessingConfig, kwargs)
    pipeline_config = _ensure_config(pipeline_config, PolarsPipelineConfig, kwargs)
    split_config = _ensure_config(split_config, TrainingSplitConfig, kwargs)

    # Default models
    if mlframe_models is None:
        mlframe_models = ["cb", "lgb", "xgb", "mlp", "linear"]

    # Metadata for tracking
    metadata = _create_initial_metadata(
        model_name=model_name,
        target_name=target_name,
        mlframe_models=mlframe_models,
        preprocessing_config=preprocessing_config,
        pipeline_config=pipeline_config,
        split_config=split_config,
    )

    # ==================================================================================
    # 2. DATA LOADING & PREPROCESSING
    # ==================================================================================

    if verbose:
        log_phase("PHASE 1: Data Loading & Preprocessing")

    # Load and prepare dataframe
    df = load_and_prepare_dataframe(df, preprocessing_config, verbose=verbose)

    # Apply features_and_targets_extractor to extract targets
    if verbose:
        logger.info("Create additional features & extracting targets...")

    df, target_by_type, group_ids_raw, group_ids, timestamps, artifacts, additional_columns_to_drop, sample_weights = features_and_targets_extractor.transform(
        df
    )

    clean_ram()
    if verbose:
        log_ram_usage()

    # Drop columns AFTER features_and_targets_extractor (columns might be needed by features_and_targets_extractor or created by it)
    df = drop_columns_from_dataframe(
        df,
        additional_columns_to_drop=additional_columns_to_drop,
        config_drop_columns=preprocessing_config.drop_columns,
        verbose=verbose,
    )

    # Preprocess dataframe (handle nulls, infinities, constants, dtypes)
    df = preprocess_dataframe(df, preprocessing_config, verbose=verbose)

    # ==================================================================================
    # 3. TRAIN/VAL/TEST SPLITTING
    # ==================================================================================

    if verbose:
        log_phase("PHASE 2: Train/Val/Test Splitting")

    if verbose:
        logger.info(f"Making train_val_test split...")
    train_idx, val_idx, test_idx, train_details, val_details, test_details = make_train_test_split(
        df=df,
        timestamps=timestamps,
        **split_config.model_dump(),
    )
    if verbose:
        log_ram_usage()

    # Save artifacts
    if data_dir:
        save_split_artifacts(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            timestamps=timestamps,
            group_ids_raw=group_ids_raw,
            artifacts=artifacts,
            data_dir=data_dir,
            models_dir=models_dir,
            target_name=target_name,
            model_name=model_name,
        )

    metadata.update(
        {
            "train_details": train_details,
            "val_details": val_details,
            "test_details": test_details,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
        }
    )

    # Pre-compute fairness subgroups from full df BEFORE splitting
    # (bins must cover all rows for train/val/test evaluation)
    fairness_subgroups, fairness_features = _compute_fairness_subgroups(df, control_params_override)
    if verbose:
        if fairness_features and fairness_subgroups is None:
            logger.warning(f"Fairness features {fairness_features} specified but subgroups could not be computed")
        elif fairness_subgroups is not None:
            logger.info(f"Computed {len(fairness_subgroups)} fairness subgroups")

    # Create split dataframes
    train_df, val_df, test_df = create_split_dataframes(
        df=df,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    # Delete original df to free RAM
    if verbose:
        logger.info("Deleting original DataFrame to free RAM...")

    del df
    clean_ram()

    if verbose:
        log_ram_usage()

    # ==================================================================================
    # 4. PIPELINE FITTING & TRANSFORMATION
    # ==================================================================================

    if verbose:
        log_phase("PHASE 3: Pipeline Fitting & Transformation")

    # Track if input is Polars before pipeline transformation
    was_polars_input = isinstance(train_df, pl.DataFrame)

    train_df, val_df, test_df, pipeline, cat_features = fit_and_transform_pipeline(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config=pipeline_config,
        ensure_float32=preprocessing_config.ensure_float32_dtypes,
        verbose=verbose,
    )

    # Track if Polars-ds pipeline was applied (to skip redundant pre_pipeline transforms later)
    polars_pipeline_applied = was_polars_input and pipeline_config.use_polarsds_pipeline and pipeline is not None

    metadata["pipeline"] = pipeline
    metadata["cat_features"] = cat_features
    metadata["columns"] = train_df.columns.tolist() if isinstance(train_df, pd.DataFrame) else train_df.columns

    # ==================================================================================
    # 5. MODEL TRAINING
    # ==================================================================================

    if verbose:
        log_phase("PHASE 4: Model Training")

    # Initialize default values for training parameters
    (
        init_common_params,
        rfecv_models,
        mrmr_kwargs,
        config_params,
        control_params,
        config_params_override,
        control_params_override,
    ) = _initialize_training_defaults(
        init_common_params=init_common_params,
        rfecv_models=rfecv_models,
        mrmr_kwargs=mrmr_kwargs,
        config_params=config_params,
        control_params=control_params,
        config_params_override=config_params_override,
        control_params_override=control_params_override,
    )

    # Get pipeline components (category_encoder, imputer, scaler) from params or defaults
    category_encoder, imputer, scaler = _get_pipeline_components(init_common_params, cat_features)

    # Compute trainset stats while data is still in Polars format (more efficient)
    if isinstance(train_df, pl.DataFrame):
        if verbose:
            logger.info("Computing trainset_features_stats on Polars...")
        trainset_features_stats = get_trainset_features_stats_polars(train_df)
    else:
        trainset_features_stats = None  # Will be computed later in train_and_evaluate_model

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Actual training
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    if verbose:
        logger.info("Zero-copy conversion to pandas...")

    # Cache pandas versions for select_target (zero-copy Arrow-backed view for Polars)
    train_df_pd, val_df_pd, test_df_pd = _convert_dfs_to_pandas(train_df, val_df, test_df)

    if verbose:
        log_ram_usage()

    # ==================================================================================
    # 4.5 OUTLIER DETECTION (once, before model training loops)
    # ==================================================================================

    (filtered_train_df, filtered_val_df, filtered_train_idx, filtered_val_idx, train_od_idx, val_od_idx) = _apply_outlier_detection_global(
        train_df=train_df_pd,
        val_df=val_df_pd,
        train_idx=train_idx,
        val_idx=val_idx,
        outlier_detector=outlier_detector,
        od_val_set=od_val_set,
        verbose=verbose,
    )

    # Single global OD result (not per-target)
    outlier_detection_result = {
        "train_od_idx": train_od_idx,
        "val_od_idx": val_od_idx,
    }

    models = defaultdict(lambda: defaultdict(list))

    for target_type, targets in tqdmu(target_by_type.items(), desc="target type"):
        # !TODO ! optimize for creation of inner feature matrices of cb,lgb,xgb here. They should be created once per featureset, not once per target.
        for cur_target_name, cur_target_values in tqdmu(targets.items(), desc="target"):
            # Initialize rfecv_models_params before conditional to avoid NameError if mlframe_models is empty
            rfecv_models_params = {}
            if mlframe_models:
                # Set up directories for charts and models
                plot_file, model_file = _setup_model_directories(
                    target_name=target_name,
                    model_name=model_name,
                    target_type=target_type,
                    cur_target_name=cur_target_name,
                    data_dir=data_dir,
                    models_dir=models_dir,
                )

                # Subset targets using pre-filtered indices (OD already applied globally)
                current_train_target = (
                    cur_target_values[filtered_train_idx]
                    if isinstance(cur_target_values, (np.ndarray, pl.Series))
                    else cur_target_values.iloc[filtered_train_idx]
                )
                current_val_target = None
                if filtered_val_idx is not None:
                    current_val_target = (
                        cur_target_values[filtered_val_idx]
                        if isinstance(cur_target_values, (np.ndarray, pl.Series))
                        else cur_target_values.iloc[filtered_val_idx]
                    )

                if verbose:
                    logger.info(f"select_target...")

                # Build common_params and control_params_override for select_target
                od_common_params, current_control_override = _build_common_params_for_target(
                    init_common_params=init_common_params,
                    trainset_features_stats=trainset_features_stats,
                    plot_file=plot_file,
                    train_od_idx=train_od_idx,
                    val_od_idx=val_od_idx,
                    current_train_target=current_train_target,
                    current_val_target=current_val_target,
                    outlier_detector=outlier_detector,
                    control_params_override=control_params_override,
                    fairness_subgroups=fairness_subgroups,
                )

                common_params, models_params, rfecv_models_params, cpu_configs, gpu_configs = select_target(
                    model_name=f"{target_name} {model_name} {cur_target_name}",
                    target=cur_target_values,  # Full target (for test_target extraction)
                    target_type=target_type,
                    df=None,
                    train_df=filtered_train_df,  # Use pre-filtered DataFrame
                    val_df=filtered_val_df,  # Use pre-filtered DataFrame
                    test_df=test_df_pd,  # Test set is not filtered by outlier detector
                    train_idx=filtered_train_idx,  # Use pre-filtered indices
                    val_idx=filtered_val_idx,  # Use pre-filtered indices
                    test_idx=test_idx,
                    train_details=train_details,
                    val_details=val_details,
                    test_details=test_details,
                    group_ids=group_ids,
                    cat_features=cat_features,
                    config_params=config_params,
                    config_params_override=config_params_override,
                    control_params=control_params,
                    control_params_override=current_control_override,
                    common_params=od_common_params,
                    mlframe_models=mlframe_models,
                    linear_model_config=linear_model_config,
                )

            if verbose:
                log_ram_usage()

            # Build list of pre-pipelines (feature selection methods) to try
            pre_pipelines, pre_pipeline_names = _build_pre_pipelines(
                use_ordinary_models=use_ordinary_models,
                rfecv_models=rfecv_models,
                rfecv_models_params=rfecv_models_params,
                use_mrmr_fs=use_mrmr_fs,
                mrmr_kwargs=mrmr_kwargs,
            )

            for pre_pipeline, pre_pipeline_name in tqdmu(zip(pre_pipelines, pre_pipeline_names), desc="pre_pipeline"):
                # Skip CatBoost RFECV pipeline with metamodel_func due to sklearn clone issue
                if _should_skip_catboost_metamodel(pre_pipeline_name.strip(), target_type, control_params_override):
                    continue
                ens_models = [] if use_mlframe_ensembles else None
                orig_pre_pipeline = pre_pipeline

                # Initialize pipeline cache for transformed DataFrames (reset for each pre_pipeline)
                pipeline_cache = PipelineCache()

                # Build weight schemas: uniform (no weighting) plus any from extractor
                weight_schemas = {"uniform": None}
                weight_schemas.update(sample_weights)

                # -----------------------------------------------------------------------
                # MODEL LOOP: Train each model type with all weight variations
                # -----------------------------------------------------------------------
                for mlframe_model_name in tqdmu(mlframe_models, desc="mlframe model"):
                    # Skip CatBoost model with metamodel_func due to sklearn clone issue
                    if _should_skip_catboost_metamodel(mlframe_model_name, target_type, control_params_override):
                        continue

                    if mlframe_model_name not in models_params:
                        logger.warning(f"mlframe model {mlframe_model_name} not known, skipping...")
                        continue

                    # Use strategy pattern to determine pipeline and cache key
                    strategy = get_strategy(mlframe_model_name)
                    pre_pipeline = strategy.build_pipeline(
                        base_pipeline=orig_pre_pipeline,
                        cat_features=cat_features,
                        category_encoder=category_encoder if cat_features else None,
                        imputer=imputer,
                        scaler=scaler,
                    )
                    cache_key = strategy.cache_key

                    # --- WEIGHT SCHEMA LOOP: Train with each sample weighting ---
                    for weight_name, weight_values in tqdmu(weight_schemas.items(), desc="wighting schema"):
                        # Create model name with weight suffix
                        model_name_with_weight = mlframe_model_name
                        if weight_name != "uniform":
                            model_name_with_weight = f"{mlframe_model_name}_{weight_name}"

                        # Shallow copy common_params - only sample_weight changes per iteration
                        current_common_params = common_params.copy()
                        current_common_params["sample_weight"] = weight_values

                        # Check if we have cached transformed DataFrames for this pipeline type
                        cached_dfs = pipeline_cache.get(cache_key)

                        # Build process_model kwargs using helper
                        process_model_kwargs = _build_process_model_kwargs(
                            model_file=model_file,
                            model_name_with_weight=model_name_with_weight,
                            target_type=target_type,
                            pre_pipeline=pre_pipeline,
                            pre_pipeline_name=pre_pipeline_name,
                            cur_target_name=cur_target_name,
                            models=models,
                            model_params=models_params[mlframe_model_name],
                            common_params=current_common_params,
                            ens_models=ens_models,
                            trainset_features_stats=trainset_features_stats,
                            verbose=verbose,
                            cached_dfs=cached_dfs,
                            polars_pipeline_applied=polars_pipeline_applied,
                            mlframe_model_name=mlframe_model_name,
                        )

                        trainset_features_stats, pre_pipeline, train_df_transformed, val_df_transformed, test_df_transformed = process_model(
                            **process_model_kwargs
                        )

                        # Cache the transformed DataFrames if not already cached
                        if cached_dfs is None:
                            pipeline_cache.set(cache_key, train_df_transformed, val_df_transformed, test_df_transformed)

                    # Update orig_pre_pipeline for tree models only.
                    # Tree models return just the base_pipeline (feature selector) from build_pipeline(),
                    # so after process_model() fits it, we preserve the fitted version for subsequent models.
                    # Non-tree models wrap base_pipeline in a full Pipeline (with encoder/imputer/scaler),
                    # which we don't want to use as the base for other model types.
                    # For optimal performance, list tree models first in mlframe_models.
                    if cache_key == "tree":
                        orig_pre_pipeline = pre_pipeline

                if ens_models and len(ens_models) > 1:
                    if verbose:
                        logger.info(f"evaluating simple ensembles...")
                    _ensembles = score_ensemble(  # Result used for side effects (logging/metrics)
                        models_and_predictions=ens_models,
                        ensemble_name=f"{pre_pipeline_name}{len(ens_models)}models ",
                        **common_params,
                    )

    # ==================================================================================
    # 6. FINALIZATION
    # ==================================================================================

    # Finalize and save metadata
    _finalize_and_save_metadata(
        metadata=metadata,
        outlier_detector=outlier_detector,
        outlier_detection_result=outlier_detection_result,
        trainset_features_stats=trainset_features_stats,
        data_dir=data_dir,
        models_dir=models_dir,
        target_name=target_name,
        model_name=model_name,
        verbose=verbose,
    )

    if verbose:
        log_phase(f"Training suite completed for {model_name}, {sum(len(v) for targets in models.values() for v in targets.values())} models.")
        log_ram_usage()

    return dict(models), metadata


def predict_mlframe_models_suite(
    df: Union[pl.DataFrame, pd.DataFrame],
    models_path: str,
    features_and_targets_extractor: Optional[FeaturesAndTargetsExtractor] = None,
    model_names: Optional[List[str]] = None,
    return_probabilities: bool = True,
    verbose: int = 1,
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

    metadata_file = join(models_path, "metadata.joblib")
    if not exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    if verbose:
        logger.info(f"Loading metadata from {metadata_file}...")
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
        # Drop extra columns (target, etc.)
        if columns_to_drop:
            if isinstance(df, pd.DataFrame):
                df = df.drop(columns=[c for c in columns_to_drop if c in df.columns], errors="ignore")
            else:  # Polars
                df = df.drop([c for c in columns_to_drop if c in df.columns])

    # Convert to pandas if needed
    if isinstance(df, pl.DataFrame):
        df = get_pandas_view_of_polars_df(df)

    # Ensure columns match training columns
    if columns:
        missing_cols = set(columns) - set(df.columns)
        extra_cols = set(df.columns) - set(columns)
        if missing_cols:
            logger.warning(f"Missing columns in input: {missing_cols}")
        if extra_cols:
            if verbose:
                logger.info(f"Dropping extra columns: {extra_cols}")
            df = df[[c for c in columns if c in df.columns]]

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
            logger.info(f"Loading model: {model_name}")

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

                # For binary classification, get class 1 probability for averaging
                if probs.ndim == 2:
                    preds = (probs[:, 1] > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
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

        # Ensemble predictions from averaged probabilities
        if avg_probs.ndim == 2:
            ensemble_preds = (avg_probs[:, 1] > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
        else:
            ensemble_preds = (avg_probs > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
        results["ensemble_predictions"] = ensemble_preds

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
        logger.info(f"Generated predictions for {len(results['predictions'])} models")

    return results


def load_mlframe_suite(models_path: str) -> Tuple[Dict, Dict]:
    """
    Load a trained mlframe models suite from disk.

    Args:
        models_path: Path to the models directory

    Returns:
        Tuple of (models dict, metadata dict)
    """
    # Validate inputs
    if not isinstance(models_path, str):
        raise TypeError(f"models_path must be string, got {type(models_path).__name__}")
    if not os.path.isdir(models_path):
        raise ValueError(f"models_path must be a valid directory, got: {models_path}")

    metadata_file = join(models_path, "metadata.joblib")
    if not exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    metadata = joblib.load(metadata_file)

    # Load all models
    models = {}
    model_files = glob.glob(join(models_path, "**", "*.dump"), recursive=True)

    for model_file in model_files:
        model_name = os.path.basename(model_file).replace(".dump", "")
        model_obj = load_mlframe_model(model_file)
        if model_obj is not None:
            models[model_name] = model_obj

    return models, metadata


__all__ = [
    "train_mlframe_models_suite",
    "predict_mlframe_models_suite",
    "load_mlframe_suite",
]
