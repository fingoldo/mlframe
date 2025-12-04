"""
Model training and evaluation functions.

This module contains the core functions for training and evaluating models,
including select_target, process_model, and related helpers.

Functions
---------
select_target
    Configure model parameters for a specific target variable.
process_model
    Process a single model: load from cache or train from scratch.
"""

import logging
from timeit import default_timer as timer
from os.path import join, exists
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

from pyutilz.system import clean_ram
from mlframe.helpers import get_own_ram_usage

from .configs import TargetTypes, DEFAULT_CALIBRATION_BINS, LinearModelConfig
from .io import load_mlframe_model, save_mlframe_model


# =============================================================================
# Constants
# =============================================================================

DEFAULT_FAIRNESS_MIN_POP_CAT_THRESH = 1000
"""Minimum population threshold for fairness categorical features."""

DEFAULT_RFECV_MAX_RUNTIME_MINS = 180  # 60 * 3
"""Maximum runtime in minutes for RFECV optimization."""

DEFAULT_RFECV_CV_SPLITS = 4
"""Number of cross-validation splits for RFECV."""

DEFAULT_RFECV_MAX_NOIMPROVING_ITERS = 15
"""Maximum iterations without improvement before stopping RFECV."""

# Import from trainer module (migrated from training_old.py)
from .trainer import (
    configure_training_params,
    _build_configs_from_params,
    train_and_evaluate_model,
)

logger = logging.getLogger(__name__)


def select_target(
    model_name: str,
    target: Union[np.ndarray, pd.Series, pl.Series],
    target_type: TargetTypes,
    df: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
    train_idx: Optional[np.ndarray] = None,
    val_idx: Optional[np.ndarray] = None,
    test_idx: Optional[np.ndarray] = None,
    train_details: str = "",
    val_details: str = "",
    test_details: str = "",
    group_ids: Optional[np.ndarray] = None,
    cat_features: Optional[List[str]] = None,
    control_params: Optional[Dict[str, Any]] = None,
    control_params_override: Optional[Dict[str, Any]] = None,
    config_params: Optional[Dict[str, Any]] = None,
    config_params_override: Optional[Dict[str, Any]] = None,
    common_params: Optional[Dict[str, Any]] = None,
    sample_weight: Optional[np.ndarray] = None,
    mlframe_models: Optional[List[str]] = None,
    linear_model_config: Optional[LinearModelConfig] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Configure model parameters for a specific target variable.

    From multiple possible targets in a dataframe, selects the required one
    and adjusts parameters for respective level 0 models.

    Parameters
    ----------
    model_name : str
        Base name for the model, will be augmented with target statistics.
    target : np.ndarray, pd.Series, or pl.Series
        Target values for training.
    target_type : TargetTypes
        Type of ML task (REGRESSION or BINARY_CLASSIFICATION).
    df : pd.DataFrame
        Full dataset containing features.
    train_df : pd.DataFrame, optional
        Pre-split training features DataFrame.
    test_df : pd.DataFrame, optional
        Pre-split test features DataFrame.
    val_df : pd.DataFrame, optional
        Pre-split validation features DataFrame.
    train_idx : np.ndarray, optional
        Indices for training rows in df.
    val_idx : np.ndarray, optional
        Indices for validation rows in df.
    test_idx : np.ndarray, optional
        Indices for test rows in df.
    train_details : str, default=""
        Additional details for training set report.
    val_details : str, default=""
        Additional details for validation set report.
    test_details : str, default=""
        Additional details for test set report.
    group_ids : np.ndarray, optional
        Group identifiers for per-group AUC computation.
    cat_features : list of str, optional
        Names of categorical feature columns.
    control_params : dict, optional
        Control parameters for training (e.g., prefer_gpu_configs, nbins).
    control_params_override : dict, optional
        Override values for control_params.
    config_params : dict, optional
        Configuration parameters (e.g., learning_rate, iterations).
    config_params_override : dict, optional
        Override values for config_params.
    common_params : dict, optional
        Common parameters passed to all models.
    sample_weight : np.ndarray, optional
        Sample weights for weighted training.
    mlframe_models : list of str, optional
        List of model types to create (e.g., ["cb", "lgb", "linear"]).
        If None, all models are created. Used for lazy model creation.
    linear_model_config : LinearModelConfig, optional
        Configuration for linear models. If provided, applies to all linear
        model types (linear, ridge, lasso, etc.).

    Returns
    -------
    tuple
        (common_params, models_params, rfecv_models_params, cpu_configs, gpu_configs)
        - common_params: Updated common parameters dict
        - models_params: Per-model parameter dicts
        - rfecv_models_params: RFECV configuration dicts
        - cpu_configs: CPU-specific model configurations
        - gpu_configs: GPU-specific model configurations

    Raises
    ------
    TypeError
        If target is not a supported type (np.ndarray, pd.Series, pl.Series).
    """
    if target_type == TargetTypes.REGRESSION:
        model_name += f" MT={target.mean():.4f}"
    else:
        # Compute value counts for classification target
        if isinstance(target, (pl.Series, pd.Series)):
            vlcnts = target.value_counts(normalize=True)
        elif isinstance(target, np.ndarray):
            vlcnts = pd.Series(target).value_counts(normalize=True)
        else:
            raise TypeError(f"target must be np.ndarray, pd.Series, or pl.Series, got {type(target).__name__}")

        # Extract positive class percentage
        if isinstance(target, pl.Series):
            vlcnts = vlcnts.filter(pl.col(target.name) == 1)
            perc = vlcnts["proportion"][0] if len(vlcnts) > 0 else 0
        else:
            perc = vlcnts.loc[1] if 1 in vlcnts.index else 0
        model_name += f" BT={perc*100:.0f}%"
    logger.debug(f"select_target: model_name={model_name}")

    # Build effective control params with defaults
    effective_control_params = control_params or {
        "prefer_gpu_configs": True,
        "fairness_features": None,
        "fairness_min_pop_cat_thresh": DEFAULT_FAIRNESS_MIN_POP_CAT_THRESH,
        "use_robust_eval_metric": True,
        "nbins": DEFAULT_CALIBRATION_BINS,
        "xgboost_verbose": 0,
        "rfecv_model_verbose": 0,
        "prefer_cpu_for_lightgbm": True,
        "prefer_calibrated_classifiers": True,
    }
    if control_params_override:
        effective_control_params = {**effective_control_params, **control_params_override}

    # Build effective config params with defaults
    effective_config_params = (
        config_params.copy()
        if config_params
        else {
            "has_time": False,
            "learning_rate": 0.2,
            "iterations": 700,
            "early_stopping_rounds": 100,
            "catboost_custom_classif_metrics": None,
            "rfecv_kwargs": {
                "max_runtime_mins": DEFAULT_RFECV_MAX_RUNTIME_MINS,
                "cv_n_splits": DEFAULT_RFECV_CV_SPLITS,
                "max_noimproving_iters": DEFAULT_RFECV_MAX_NOIMPROVING_ITERS,
            },
        }
    )
    if config_params_override:
        effective_config_params.update(config_params_override)

    (
        common_params,
        models_params,
        cb_rfecv,
        lgb_rfecv,
        xgb_rfecv,
        cpu_configs,
        gpu_configs,
    ) = configure_training_params(
        df=df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target=target,
        target_label_encoder=None,
        cat_features=cat_features,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        sample_weight=sample_weight,
        train_details=train_details,
        val_details=val_details,
        test_details=test_details,
        group_ids=group_ids,
        model_name=model_name,
        common_params=common_params,
        config_params=effective_config_params,
        use_regression=target_type == TargetTypes.REGRESSION,
        mlframe_models=mlframe_models,
        linear_model_config=linear_model_config,
        **effective_control_params,
    )

    rfecv_models_params = dict(
        cb_rfecv=cb_rfecv,
        lgb_rfecv=lgb_rfecv,
        xgb_rfecv=xgb_rfecv,
    )
    return common_params, models_params, rfecv_models_params, cpu_configs, gpu_configs


def _call_train_evaluate_with_configs(
    model_obj: Optional[Any],
    model_params: Dict[str, Any],
    common_params: Dict[str, Any],
    pre_pipeline: Optional[Any],
    skip_pre_pipeline_transform: bool,
    skip_preprocessing: bool,
    model_name_prefix: str,
    just_evaluate: bool = False,
    verbose: bool = False,
    trainset_features_stats: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Call train_and_evaluate_model with config objects.

    Helper function that merges model_params and common_params, builds
    configuration objects, and calls the config-based API.

    Parameters
    ----------
    model_obj : Any, optional
        The model object to train (sklearn estimator, Pipeline, etc.).
    model_params : dict
        Model-specific parameters (e.g., fit_params).
    common_params : dict
        Common parameters shared across models (e.g., data, targets, indices).
    pre_pipeline : Any, optional
        Preprocessing pipeline (sklearn TransformerMixin).
    skip_pre_pipeline_transform : bool
        Whether to skip the preprocessing pipeline transform.
    skip_preprocessing : bool
        Whether to skip only preprocessing (scaler/imputer/encoder) while still
        running feature selectors.
    model_name_prefix : str
        Prefix to add to the model name for reports.
    just_evaluate : bool, default=False
        If True, skip training and only evaluate cached predictions.
    verbose : bool, default=False
        Whether to print verbose output.
    trainset_features_stats : dict, optional
        Pre-computed feature statistics from training set.

    Returns
    -------
    tuple
        (model, train_df, val_df, test_df) where:
        - model: Trained model with results attached
        - train_df: Transformed training DataFrame (or None)
        - val_df: Transformed validation DataFrame (or None)
        - test_df: Transformed test DataFrame (or None)
    """
    # Merge all params into flat dict for _build_configs_from_params
    all_params = {**common_params, **model_params}

    # Extract params that go directly to v2 (not through _build_configs_from_params)
    all_params.pop("model", None)  # passed separately
    train_od_idx = all_params.pop("train_od_idx", None)
    val_od_idx = all_params.pop("val_od_idx", None)
    all_params.pop("trainset_features_stats", None)  # use function arg

    # Add control params
    all_params["pre_pipeline"] = pre_pipeline
    all_params["skip_pre_pipeline_transform"] = skip_pre_pipeline_transform
    all_params["skip_preprocessing"] = skip_preprocessing
    all_params["model_name_prefix"] = model_name_prefix
    all_params["just_evaluate"] = just_evaluate
    all_params["verbose"] = verbose

    # Build config objects
    data, control, metrics, display, naming, confidence, predictions = _build_configs_from_params(**all_params)

    # Call train_and_evaluate_model with config objects
    return train_and_evaluate_model(
        model=model_obj,
        data=data,
        control=control,
        metrics=metrics,
        display=display,
        naming=naming,
        confidence=confidence,
        predictions=predictions,
        train_od_idx=train_od_idx,
        val_od_idx=val_od_idx,
        trainset_features_stats=trainset_features_stats,
    )


def process_model(
    model_file: str,
    model_name: str,
    model_file_name: str,
    target_type: TargetTypes,
    pre_pipeline: Optional[Any],
    pre_pipeline_name: str,
    cur_target_name: str,
    trainset_features_stats: Optional[Dict[str, Any]],
    models: Dict[str, Dict[str, List[Any]]],
    model_params: Dict[str, Any],
    common_params: Dict[str, Any],
    ens_models: Optional[List[Any]],
    verbose: int,
    skip_pre_pipeline_transform: bool = False,
    skip_preprocessing: bool = False,
    cached_train_df: Optional[pd.DataFrame] = None,
    cached_val_df: Optional[pd.DataFrame] = None,
    cached_test_df: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Any], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Process a single model: load from cache or train from scratch.

    Handles model loading from cache if available, otherwise trains the model
    and optionally saves it. Updates the models dict and ensemble list.

    Parameters
    ----------
    model_file : str
        Directory path for saving/loading model files.
    model_name : str
        Name identifier for the model.
    target_type : TargetTypes
        Type of ML task (REGRESSION or BINARY_CLASSIFICATION).
    pre_pipeline : Any, optional
        Preprocessing pipeline (sklearn TransformerMixin).
    pre_pipeline_name : str
        Name of the preprocessing pipeline for file naming.
    cur_target_name : str
        Current target column name being processed.
    trainset_features_stats : dict, optional
        Statistics computed on training features (e.g., means, stds).
    models : dict
        Dictionary to store trained models, keyed by target name and type.
    model_params : dict
        Model-specific parameters including the 'model' key.
    common_params : dict
        Common parameters for training (data, indices, etc.).
    ens_models : list, optional
        List to collect models for ensemble. Can be None if not using ensembles.
    verbose : int
        Verbosity level (0=silent, 1=info, 2=debug).
    skip_pre_pipeline_transform : bool, default=False
        Whether to skip the preprocessing pipeline transform.
    skip_preprocessing : bool, default=False
        Whether to skip only preprocessing (scaler/imputer/encoder) while still
        running feature selectors.
    cached_train_df : pd.DataFrame, optional
        Pre-transformed training DataFrame to reuse.
    cached_val_df : pd.DataFrame, optional
        Pre-transformed validation DataFrame to reuse.
    cached_test_df : pd.DataFrame, optional
        Pre-transformed test DataFrame to reuse.

    Returns
    -------
    tuple
        (trainset_features_stats, pre_pipeline, train_df, val_df, test_df)
        - trainset_features_stats: Updated feature statistics
        - pre_pipeline: The preprocessing pipeline (may be updated from cache)
        - train_df: Transformed training DataFrame
        - val_df: Transformed validation DataFrame
        - test_df: Transformed test DataFrame

    Raises
    ------
    KeyError
        If 'model' key is missing in model_params when not loading from cache.
    """
    # Build model file path
    fname = f"{model_file_name}.dump"
    if pre_pipeline_name:
        fname = pre_pipeline_name + " " + fname
    fpath = join(model_file, fname) if model_file else None

    # Prepare common_params with cached DataFrames if provided
    effective_common_params = common_params.copy()
    effective_common_params["model_name"] = model_name
    if cached_train_df is not None:
        effective_common_params["train_df"] = cached_train_df
    if cached_val_df is not None:
        effective_common_params["val_df"] = cached_val_df
    if cached_test_df is not None:
        effective_common_params["test_df"] = cached_test_df

    # Remove parameters not accepted by train_and_evaluate_model
    for key in ["scaler", "imputer", "category_encoder", "rfecv_params", "model_path", "artifact_dir"]:
        effective_common_params.pop(key, None)

    # Check if model exists in cache
    use_cached_model = bool(fpath and exists(fpath))
    if use_cached_model:
        if verbose:
            logger.info(f"Loading model from file {fpath}")
        loaded_model = load_mlframe_model(fpath)
        if verbose:
            logger.info(f"Loaded.")        
        model_obj = loaded_model.model
        pre_pipeline = loaded_model.pre_pipeline
    else:
        if "model" not in model_params:
            raise KeyError(f"'model' key missing in model_params. Available keys: {list(model_params.keys())}")
        model_obj = model_params["model"]

    # Train or evaluate the model
    start = timer()
    if verbose and not use_cached_model:
        pipeline_label = pre_pipeline_name.strip() if pre_pipeline_name else ""
        model_type_name = type(model_obj).__name__
        logger.info(
            f"Starting train_and_evaluate {model_type_name} on {target_type} {pipeline_label} {model_name.strip()}" f", RAM usage {get_own_ram_usage():.1f}GBs...".replace("  ", " ")
        )

    model, train_df_transformed, val_df_transformed, test_df_transformed = _call_train_evaluate_with_configs(
        model_obj=model_obj,
        model_params=model_params,
        common_params=effective_common_params,
        pre_pipeline=pre_pipeline,
        skip_pre_pipeline_transform=skip_pre_pipeline_transform,
        skip_preprocessing=skip_preprocessing,
        model_name_prefix=pre_pipeline_name,
        just_evaluate=use_cached_model,
        verbose=verbose,
        trainset_features_stats=trainset_features_stats,
    )

    # Handle failed model - don't save or add to lists
    if model.model is None:
        logger.warning(f"Skipping failed model {model_name}")
        return trainset_features_stats, pre_pipeline, train_df_transformed, val_df_transformed, test_df_transformed

    if not use_cached_model:
        end = timer()
        if verbose:
            logger.info(f"Finished training, took {(end-start)/60:.1f} min. RAM usage {get_own_ram_usage():.1f}GBs...")
        if fpath:
            save_mlframe_model(model, fpath)

    models[cur_target_name][target_type].append(model)

    # ens_models can be None when not building ensembles
    if ens_models is not None:
        ens_models.append(model)

    if trainset_features_stats is None:
        trainset_features_stats = model.trainset_features_stats
        common_params["trainset_features_stats"] = trainset_features_stats

    clean_ram()

    return trainset_features_stats, pre_pipeline, train_df_transformed, val_df_transformed, test_df_transformed


__all__ = [
    # Constants
    "DEFAULT_FAIRNESS_MIN_POP_CAT_THRESH",
    "DEFAULT_RFECV_MAX_RUNTIME_MINS",
    "DEFAULT_RFECV_CV_SPLITS",
    "DEFAULT_RFECV_MAX_NOIMPROVING_ITERS",
    # Functions
    "select_target",
    "process_model",
    "_call_train_evaluate_with_configs",
]
