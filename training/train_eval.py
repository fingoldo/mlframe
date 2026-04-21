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

from pyutilz.system import clean_ram, get_own_memory_usage
from mlframe.training.utils import maybe_clean_ram_adaptive

from .configs import (
    TargetTypes,
    DEFAULT_CALIBRATION_BINS,
    DEFAULT_FAIRNESS_MIN_POP_CAT_THRESH,
    DEFAULT_RFECV_MAX_RUNTIME_MINS,
    DEFAULT_RFECV_CV_SPLITS,
    DEFAULT_RFECV_MAX_NOIMPROVING_ITERS,
    LinearModelConfig,
    ModelHyperparamsConfig,
    TrainingBehaviorConfig,
)
from .io import load_mlframe_model, save_mlframe_model


logger = logging.getLogger(__name__)


def _extract_polars_cat_columns(df) -> List[str]:
    """Return column names whose dtype is `pl.Categorical` or `pl.Enum`.

    Used by cache-schema validation: if a column is Polars Categorical at
    predict time but was not registered as a cat_feature at fit time,
    CatBoost's native-Polars fastpath raises
    ``CatBoostError: Unsupported data type Categorical for a numerical feature column``.
    """
    if df is None or not isinstance(df, pl.DataFrame):
        return []
    out: List[str] = []
    for name, dtype in df.schema.items():
        if dtype == pl.Categorical or isinstance(dtype, pl.Enum):
            out.append(name)
    return out


def _validate_cached_model_schema(
    loaded_model,
    current_df,
) -> Optional[str]:
    """Return a reason string if the cached model's schema is incompatible
    with the current DataFrame, else None.

    Catches the common stale-cache scenarios where preprocessing or the
    feature set changed between the saved model and the current run:
        * the column list or order differs;
        * a column is Polars Categorical in `current_df` but was not
          registered as a cat_feature in the saved CatBoost model.

    The latter produces a cryptic
    ``CatBoostError: Unsupported data type Categorical for a numerical feature column``
    deep inside CatBoost's pyx layer — this pre-flight check catches it.
    """
    m = getattr(loaded_model, "model", None)
    if m is None:
        return None

    # 1. Feature-name sequence check (CB / XGB / LGB / sklearn linear).
    saved_names: Optional[List[str]] = None
    for attr in ("feature_names_", "feature_names_in_"):
        if hasattr(m, attr):
            try:
                raw = getattr(m, attr)
                if raw is not None:
                    saved_names = list(raw)
                    break
            except Exception:
                continue
    if saved_names is None and hasattr(m, "get_booster"):
        try:
            booster_names = m.get_booster().feature_names
            if booster_names:
                saved_names = list(booster_names)
        except Exception:
            saved_names = None

    if saved_names:
        current_names = list(current_df.columns) if current_df is not None else []
        if saved_names != current_names:
            diff = set(current_names) ^ set(saved_names)
            if diff:
                sample = sorted(diff)[:8]
                return (
                    f"feature-name mismatch (saved={len(saved_names)}, current={len(current_names)}); "
                    f"symmetric diff sample={sample}"
                )
            return "feature-name order differs between saved model and current df"

    # 2. CatBoost-specific cat_features check. Only meaningful if we have
    # saved feature names (to resolve indices back to names).
    if saved_names and hasattr(m, "_get_cat_feature_indices"):
        try:
            saved_cat_names = {saved_names[i] for i in m._get_cat_feature_indices() if 0 <= i < len(saved_names)}
        except Exception:
            saved_cat_names = None
        if saved_cat_names is not None:
            current_pl_cats = set(_extract_polars_cat_columns(current_df))
            missing = current_pl_cats - saved_cat_names
            if missing:
                return (
                    f"CatBoost cache mismatch: columns {sorted(missing)} are Polars Categorical in "
                    f"current df but were not trained as cat_features in the saved model"
                )
    return None


# =============================================================================
# Constants
# =============================================================================

# Constants now imported from configs.py (DEFAULT_FAIRNESS_MIN_POP_CAT_THRESH, etc.)

# Import from trainer module
from .trainer import (
    configure_training_params,
    _build_configs_from_params,
    train_and_evaluate_model,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Storage Optimization
# =============================================================================


def optimize_model_for_storage(
    model,
    target_type: TargetTypes,
    metadata_columns: Optional[List[str]] = None,
) -> None:
    """Optimize a model object for storage by removing redundant data.

    For classification models:
    - Removes train_preds, val_preds, test_preds (can be recreated from *_probs >= 0.5)

    For binary classification:
    - Keeps *_probs arrays in original (n, 2) shape for compatibility with
      load_mlframe_suite and ensembling code

    If metadata_columns is provided:
    - Removes model.columns if identical to metadata_columns

    Parameters
    ----------
    model : SimpleNamespace
        Model object with predictions and metadata attributes.
    target_type : TargetTypes
        Type of ML task (REGRESSION or BINARY_CLASSIFICATION).
    metadata_columns : list of str, optional
        Columns stored in metadata. If provided and identical to model.columns,
        model.columns will be set to None to save storage.

    Notes
    -----
    This function modifies the model object in-place.
    """
    is_classification = target_type == TargetTypes.BINARY_CLASSIFICATION

    if is_classification:
        # Remove *_preds for classification (can be recreated from *_probs >= 0.5)
        model.train_preds = None
        model.val_preds = None
        model.test_preds = None

        # NOTE: Do NOT squeeze probs from (n, 2) to (n,) here.
        # The in-memory models must keep 2D probs to match load_mlframe_suite output
        # and to remain compatible with ensembling code (ensemble_probabilistic_predictions
        # expects 2D arrays via pred.shape[1]).

    # Remove columns if identical to metadata columns
    if metadata_columns is not None and hasattr(model, "columns") and model.columns is not None:
        model_columns = list(model.columns) if not isinstance(model.columns, list) else model.columns
        if model_columns == metadata_columns:
            model.columns = None


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
    text_features: Optional[List[str]] = None,
    embedding_features: Optional[List[str]] = None,
    hyperparams_config: Optional[ModelHyperparamsConfig] = None,
    behavior_config: Optional[TrainingBehaviorConfig] = None,
    common_params: Optional[Dict[str, Any]] = None,
    sample_weight: Optional[np.ndarray] = None,
    mlframe_models: Optional[List[str]] = None,
    linear_model_config: Optional[LinearModelConfig] = None,
    train_df_size_bytes: Optional[float] = None,
    val_df_size_bytes: Optional[float] = None,
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
    hyperparams_config : ModelHyperparamsConfig, optional
        Model hyperparameters (iterations, learning_rate, per-model kwargs).
    behavior_config : TrainingBehaviorConfig, optional
        Training behavior flags (GPU preference, calibration, fairness).
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

        # Degenerate-target guard: classification with a single class
        # (all 0s or all 1s after the target-building threshold) makes
        # ROC AUC / PR AUC undefined, the scorer returns NaN, and
        # early-stopping / ICE / RFECV all silently break in downstream
        # layers (some of which we already added NaN observability for).
        # Better to WARN loud right here so the operator can fix the
        # threshold or data selection. We do NOT abort — in rare cases
        # (sanity runs, debugging) training a model on a single-class
        # target is intentional, and a hard raise would regress
        # legitimate callers.
        if 0.0 < perc < 1.0:
            # Also flag extreme imbalance (<0.1% or >99.9%) because
            # even with both classes present the signal is near-zero.
            if perc < 1e-3 or perc > (1.0 - 1e-3):
                logger.warning(
                    "select_target: extreme class imbalance for '%s' "
                    "(positive rate %.4f%%). Training may converge on "
                    "the majority class; AUC metrics will be noisy.",
                    model_name, perc * 100,
                )
        else:
            logger.warning(
                "select_target: degenerate classification target '%s' "
                "has only one class (positive rate=%.0f%%). ROC AUC / "
                "PR AUC are undefined; scorer will return NaN and "
                "early-stopping will stall. Fix the target threshold or "
                "pre-filter the data upstream.",
                model_name, perc * 100,
            )
    logger.debug(f"select_target: model_name={model_name}")

    # Ensure configs have defaults
    if hyperparams_config is None:
        hyperparams_config = ModelHyperparamsConfig()
    if behavior_config is None:
        behavior_config = TrainingBehaviorConfig()

    # Convert Pydantic configs to dicts for configure_training_params
    # exclude_none=True: downstream functions handle missing keys with their own defaults
    effective_config_params = hyperparams_config.model_dump(exclude_none=True)
    # Only include defined fields — exclude any extra fields (e.g. _precomputed_fairness_subgroups).
    # Also exclude suite-level meta-flags that have no meaning to
    # configure_training_params (crash reporting + per-model
    # continue-on-failure are consumed directly by
    # train_mlframe_models_suite, not passed down the stack).
    defined_behavior_fields = set(TrainingBehaviorConfig.model_fields.keys())
    _SUITE_LEVEL_FLAGS = {
        "enable_crash_reporting",
        "continue_on_model_failure",
        "align_polars_categorical_dicts",
        # 2026-04-21 Fix 8: save-time filename policy; not consumed by
        # configure_training_params, so mask from the downstream kwarg set.
        "model_file_hash_suffix",
    }
    effective_behavior_params = {
        k: v for k, v in behavior_config.model_dump(exclude_none=True).items()
        if k in defined_behavior_fields and k not in _SUITE_LEVEL_FLAGS
    }
    # Pass _precomputed_fairness_subgroups explicitly if present (set by _build_common_params_for_target)
    precomputed_fairness = (behavior_config.model_extra or {}).get("_precomputed_fairness_subgroups")
    if precomputed_fairness is not None:
        effective_behavior_params["_precomputed_fairness_subgroups"] = precomputed_fairness

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
        text_features=text_features,
        embedding_features=embedding_features,
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
        train_df_size_bytes=train_df_size_bytes,
        val_df_size_bytes=val_df_size_bytes,
        **effective_behavior_params,
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
    optimize_storage: bool = True,
    metadata_columns: Optional[List[str]] = None,
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

    # Check if model exists in cache.
    #
    # Gating: historically this path loaded blindly whenever the .dump
    # existed. Preserve that as the default (suite-level cache is expected
    # to "just work"), but now callers can opt out explicitly with
    # ``init_common_params={"use_cache": False}`` for forced retrain.
    #
    # Schema validation: when the cache does load, validate the saved
    # model's feature list + cat_features against the current preprocessed
    # DataFrame. A mismatch usually means preprocessing or the feature set
    # changed between runs — the classic symptom being a cryptic
    # ``Unsupported data type Categorical for a numerical feature column``
    # crash deep in CatBoost's Polars fastpath. Invalidate the stale cache
    # and retrain rather than bubble the opaque backend error.
    use_cache_flag = bool(common_params.get("use_cache", True))
    use_cached_model = use_cache_flag and bool(fpath and exists(fpath))
    if use_cached_model:
        if verbose:
            logger.info(f"Loading model from file {fpath}")
        loaded_model = load_mlframe_model(fpath)
        if verbose:
            logger.info(f"Loaded.")
        mismatch = _validate_cached_model_schema(loaded_model, common_params.get("train_df"))
        if mismatch:
            logger.warning(f"Invalidating stale cached model at {fpath}: {mismatch}. Retraining.")
            use_cached_model = False
        else:
            model_obj = loaded_model.model
            pre_pipeline = loaded_model.pre_pipeline
    if not use_cached_model:
        if "model" not in model_params:
            raise KeyError(f"'model' key missing in model_params. Available keys: {list(model_params.keys())}")
        model_obj = model_params["model"]

    maybe_clean_ram_adaptive()

    # Train or evaluate the model
    start = timer()
    if verbose and not use_cached_model:
        pipeline_label = pre_pipeline_name.strip() if pre_pipeline_name else ""
        model_type_name = type(model_obj).__name__
        logger.info(
            f"Starting train_and_evaluate {model_type_name} on {target_type} {pipeline_label} {model_name.strip()}"
            f", RAM usage {get_own_memory_usage():.1f}GBs...".replace("  ", " ")
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
            logger.info(f"Finished training, took {(end-start)/60:.1f} min. RAM usage {get_own_memory_usage():.1f}GBs...")
        if fpath:
            save_mlframe_model(model, fpath)

    # Optimize model for in-memory storage (after saving to disk to preserve full data in files)
    if optimize_storage:
        optimize_model_for_storage(model, target_type, metadata_columns)

    models[target_type][cur_target_name].append(model)

    # ens_models can be None when not building ensembles
    if ens_models is not None:
        ens_models.append(model)

    if trainset_features_stats is None:
        trainset_features_stats = model.trainset_features_stats
        common_params["trainset_features_stats"] = trainset_features_stats

    maybe_clean_ram_adaptive()

    return trainset_features_stats, pre_pipeline, train_df_transformed, val_df_transformed, test_df_transformed


__all__ = [
    # Constants
    "DEFAULT_FAIRNESS_MIN_POP_CAT_THRESH",
    "DEFAULT_RFECV_MAX_RUNTIME_MINS",
    "DEFAULT_RFECV_CV_SPLITS",
    "DEFAULT_RFECV_MAX_NOIMPROVING_ITERS",
    # Functions
    "optimize_model_for_storage",
    "select_target",
    "process_model",
    "_call_train_evaluate_with_configs",
]
