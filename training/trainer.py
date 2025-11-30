"""
Core training and evaluation functions.

This module contains:
- train_and_evaluate_model: Main training function
- train_and_evaluate_model_v2: Config-based wrapper
- configure_training_params: Training parameter configuration
- _build_configs_from_params: Config object builder
- Helper functions for training
"""

import copy
import inspect
import logging
import pickle
from functools import partial
from os import sep as os_sep
from os.path import join, exists
from types import SimpleNamespace
from typing import Optional, Tuple, Union, Callable, Sequence, List, Any, Dict

import numpy as np
import pandas as pd
import polars as pl
import joblib
import matplotlib.pyplot as plt

from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin, is_classifier
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    max_error,
    r2_score,
    root_mean_squared_error,
    make_scorer,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from ngboost import NGBClassifier, NGBRegressor
import flaml.default as flaml_zeroshot
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlframe.lightninglib import MLPNeuronsByLayerArchitecture
from mlframe.lightninglib import PytorchLightningRegressor, PytorchLightningClassifier

from pyutilz.system import clean_ram, ensure_dir_exists, compute_total_gpus_ram, get_gpuinfo_gpu_info
from pyutilz.strings import slugify
from pyutilz.pandaslib import get_df_memory_consumption
from pyutilz.pythonlib import prefix_dict_elems, get_human_readable_set_size

from mlframe.helpers import get_model_best_iter, ensure_no_infinity
from mlframe.config import (
    TABNET_MODEL_TYPES,
    XGBOOST_MODEL_TYPES,
    CATBOOST_MODEL_TYPES,
    LGBM_MODEL_TYPES,
)

# Constants (originally from training_old.py)
from numba.cuda import is_available as is_cuda_available

CUDA_IS_AVAILABLE = is_cuda_available()
MODELS_SUBDIR = "models"
GPU_VRAM_SAFE_SATURATION_LIMIT: float = 0.9
GPU_VRAM_SAFE_FREE_LIMIT_GB: float = 0.1
from mlframe.metrics import (
    compute_probabilistic_multiclass_error,
    fast_calibration_report,
    fast_roc_auc,
)


# Import helper functions from helpers module (migrated from training_old.py)
from .helpers import (
    get_trainset_features_stats,
    get_trainset_features_stats_polars,
    get_training_configs,
    parse_catboost_devices,
    LightGBMCallback,
    CatBoostCallback,
    XGBoostCallback,
)

# Fairness and feature importance functions from their respective modules
from mlframe.metrics import create_fairness_subgroups, create_fairness_subgroups_indices, compute_fairness_metrics
from mlframe.feature_importance import plot_feature_importance
from mlframe.metrics import ICE
from mlframe.feature_selection.wrappers import RFECV

from .configs import (
    DataConfig,
    TrainingControlConfig,
    MetricsConfig,
    DisplayConfig,
    NamingConfig,
    ConfidenceAnalysisConfig,
    PredictionsContainer,
)
from .utils import log_ram_usage, get_categorical_columns, get_numeric_columns
from .models import create_linear_model, LINEAR_MODEL_TYPES

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def get_function_param_names(func: Callable) -> List[str]:
    """Get parameter names from a function signature.

    Parameters
    ----------
    func : Callable
        Function to inspect.

    Returns
    -------
    list of str
        List of parameter names.
    """
    signature = inspect.signature(func)
    return list(signature.parameters.keys())


def _extract_target_subset(
    target: Optional[Union[pd.Series, pl.Series, np.ndarray]],
    idx: Optional[np.ndarray],
) -> Optional[Union[pd.Series, pl.Series, np.ndarray]]:
    """Extract target subset handling pandas Series, polars Series, and numpy arrays.

    Parameters
    ----------
    target : pd.Series, pl.Series, np.ndarray, or None
        Target values to subset.
    idx : np.ndarray or None
        Indices to select. If None, returns full target.

    Returns
    -------
    pd.Series, pl.Series, np.ndarray, or None
        Subsetted target values.
    """
    if idx is None:
        return target
    if isinstance(target, pd.Series):
        return target.iloc[idx]
    elif isinstance(target, pl.Series):
        return target.gather(idx)
    return target[idx]


def _subset_dataframe(
    df: Optional[Union[pd.DataFrame, pl.DataFrame]],
    idx: Optional[np.ndarray],
    drop_columns: Optional[List[str]] = None,
) -> Optional[Union[pd.DataFrame, pl.DataFrame]]:
    """Subset DataFrame with optional column dropping, handling pandas and polars.

    Parameters
    ----------
    df : pd.DataFrame, pl.DataFrame, or None
        Input DataFrame to subset.
    idx : np.ndarray or None
        Indices to select. If None, returns full DataFrame.
    drop_columns : list of str, optional
        Columns to drop from the result.

    Returns
    -------
    pd.DataFrame, pl.DataFrame, or None
        Subsetted DataFrame with specified columns dropped.
    """
    if df is None:
        return df
    if idx is None:
        result = df
    elif isinstance(df, pd.DataFrame):
        result = df.iloc[idx]
    elif isinstance(df, pl.DataFrame):
        result = df[idx]
    else:
        return df

    if drop_columns:
        # Validate drop_columns is a list-like, not a string
        if isinstance(drop_columns, str):
            logger.warning(f"drop_columns should be a list, got string '{drop_columns}'. Converting to list.")
            drop_columns = [drop_columns]
        if isinstance(result, pd.DataFrame):
            return result.drop(columns=[c for c in drop_columns if c in result.columns])
        elif isinstance(result, pl.DataFrame):
            cols_to_drop = [c for c in drop_columns if c in result.columns]
            return result.drop(cols_to_drop) if cols_to_drop else result
    return result


def _prepare_df_for_model(df, model_type_name):
    """Convert DataFrame to numpy if required by model type (e.g., TabNet)."""
    if df is None:
        return None
    if model_type_name in TABNET_MODEL_TYPES and hasattr(df, "values"):
        return df.values
    return df


def _setup_sample_weight(sample_weight, train_idx, model_obj, fit_params):
    """Configure sample weights in fit_params if supported by model."""
    if sample_weight is None:
        return
    if "sample_weight" not in get_function_param_names(model_obj.fit):
        return

    if isinstance(sample_weight, (pd.Series, pd.DataFrame)):
        if train_idx is not None:
            fit_params["sample_weight"] = sample_weight.iloc[train_idx].values
        else:
            fit_params["sample_weight"] = sample_weight.values
    else:
        if train_idx is not None:
            fit_params["sample_weight"] = sample_weight[train_idx]
        else:
            fit_params["sample_weight"] = sample_weight


def _initialize_mutable_defaults(drop_columns, default_drop_columns, fi_kwargs, confidence_model_kwargs):
    """Initialize mutable default arguments."""
    if drop_columns is None:
        drop_columns = []
    if default_drop_columns is None:
        default_drop_columns = []
    if fi_kwargs is None:
        fi_kwargs = {}
    if confidence_model_kwargs is None:
        confidence_model_kwargs = {}
    return drop_columns, default_drop_columns, fi_kwargs, confidence_model_kwargs


def _validate_infinity_and_columns(df, train_df, skip_infinity_checks, drop_columns, default_drop_columns):
    """Validate DataFrames for infinity values and compute real drop columns."""
    if not skip_infinity_checks:
        if df is not None:
            ensure_no_infinity(df)
        else:
            if train_df is not None:
                ensure_no_infinity(train_df)

    if df is not None:
        real_drop_columns = [col for col in drop_columns + default_drop_columns if col in df.columns]
    elif train_df is not None:
        real_drop_columns = [col for col in drop_columns + default_drop_columns if col in train_df.columns]
    else:
        real_drop_columns = []

    return real_drop_columns


def _setup_model_info_and_paths(model, model_name, model_name_prefix, plot_file, data_dir, models_subdir):
    """Extract model object info and construct naming/path information."""
    if type(model).__name__ == "Pipeline":
        model_obj = model.named_steps["est"]
    else:
        model_obj = model

    if model_obj is not None:
        if isinstance(model_obj, TransformedTargetRegressor):
            model_obj = model_obj.regressor
    model_type_name = type(model_obj).__name__ if model_obj is not None else ""

    if plot_file:
        if not plot_file.endswith(os_sep):
            plot_file = plot_file + "_"
        if model_name_prefix:
            plot_file = plot_file + slugify(model_name_prefix) + " "
        if model_type_name:
            plot_file = plot_file + slugify(model_type_name) + " "
        plot_file = plot_file.strip()

    if model_name_prefix:
        model_name = model_name_prefix + model_name
    if model_type_name not in model_name:
        model_name = model_type_name + " " + model_name

    ensure_dir_exists(join(data_dir, models_subdir))
    model_file_name = join(data_dir, models_subdir, f"{model_name}.dump")

    return model_obj, model_type_name, model_name, plot_file, model_file_name


def _compute_trainset_stats(train_df, trainset_features_stats, verbose):
    """Compute trainset feature statistics if not already provided."""
    if not trainset_features_stats:
        if verbose:
            logger.info("Computing trainset_features_stats...")
        if isinstance(train_df, pl.DataFrame):
            trainset_features_stats = get_trainset_features_stats_polars(train_df)
        elif isinstance(train_df, pd.DataFrame):
            trainset_features_stats = get_trainset_features_stats(train_df)
    return trainset_features_stats


def _disable_xgboost_early_stopping_if_needed(model_type_name, model_obj):
    """Disable XGBoost early stopping when no validation data is available."""
    if model_type_name in XGBOOST_MODEL_TYPES and model_obj is not None:
        es_rounds = getattr(model_obj, "early_stopping_rounds", None)
        if es_rounds is not None:
            logger.warning(f"No validation data available - disabling early stopping for {model_type_name}")
            model_obj.set_params(early_stopping_rounds=None)


def _extract_targets_from_indices(target, train_idx, val_idx, test_idx, train_target, val_target, test_target):
    """Extract train/val/test targets from main target using indices."""
    if target is not None:
        if train_target is None and (train_idx is not None):
            train_target = _extract_target_subset(target, train_idx)
        if val_target is None and (val_idx is not None):
            val_target = _extract_target_subset(target, val_idx)
        if test_target is None and (test_idx is not None):
            test_target = _extract_target_subset(target, test_idx)
    return train_target, val_target, test_target


def _prepare_train_df_for_fitting(train_df, model, model_type_name, fit_params):
    """Prepare training DataFrame and fit_params for model fitting."""
    if model_type_name in TABNET_MODEL_TYPES:
        train_df = train_df.values

    if fit_params and type(model).__name__ == "Pipeline":
        fit_params = prefix_dict_elems(fit_params, "est__")

    return train_df, fit_params


def _update_model_name_after_training(model_name, train_df_len, train_details, best_iter):
    """Update model name with training details and early stopping info."""
    model_name = model_name + "\n" + " ".join([f" trained on {get_human_readable_set_size(train_df_len)} rows", train_details])

    if best_iter:
        print(f"es_best_iter: {best_iter:_}")
        model_name = model_name + f" @iter={best_iter:_}"

    return model_name


def _prepare_test_split(df, test_df, test_idx, test_target, target, real_drop_columns, model, pre_pipeline, skip_pre_pipeline_transform):
    """Prepare test DataFrame and target for evaluation."""
    if (df is not None) or (test_df is not None):
        if test_df is None:
            test_df = _subset_dataframe(df, test_idx, real_drop_columns)

        if test_target is None:
            test_target = _extract_target_subset(target, test_idx)

        if model is not None and pre_pipeline and not skip_pre_pipeline_transform:
            test_df = pre_pipeline.transform(test_df)
        columns = list(test_df.columns) if hasattr(test_df, "columns") else []
    else:
        columns = []
        test_df = None

    return test_df, test_target, columns


def _apply_pre_pipeline_transforms(model, pre_pipeline, train_df, val_df, train_target, skip_pre_pipeline_transform, use_cache, model_file_name, verbose):
    """Apply pre-pipeline transformations to train and validation DataFrames."""
    if model is not None and pre_pipeline:
        if skip_pre_pipeline_transform:
            if verbose:
                logger.info(f"Skipping pre_pipeline fit/transform (already transformed)")
        elif use_cache and exists(model_file_name):
            if verbose:
                logger.info(f"Transforming train_df via pre_pipeline...")
            train_df = pre_pipeline.transform(train_df, train_target)
            if verbose:
                log_ram_usage()
        else:
            if verbose:
                logger.info(f"Fitting & Transforming train_df via pre_pipeline...")
            train_df = pre_pipeline.fit_transform(train_df, train_target)
            if verbose:
                log_ram_usage()

        if not skip_pre_pipeline_transform and val_df is not None:
            if verbose:
                logger.info(f"Transforming val_df via pre_pipeline...")
            val_df = pre_pipeline.transform(val_df)
            if verbose:
                log_ram_usage()
        clean_ram()

    return train_df, val_df


def _setup_eval_set(
    model_type_name: str,
    fit_params: Dict[str, Any],
    val_df: Union[pd.DataFrame, np.ndarray],
    val_target: Union[pd.Series, np.ndarray],
    callback_params: Optional[Dict[str, Any]] = None,
    model_obj: Optional[Any] = None,
) -> None:
    """Configure eval_set/validation data for different model types.

    Modifies fit_params in-place to add the appropriate eval_set configuration
    based on the model type (LightGBM, CatBoost, XGBoost, etc.).

    Parameters
    ----------
    model_type_name : str
        Name of the model type (e.g., 'LGBMClassifier').
    fit_params : dict
        Dictionary to populate with eval_set configuration.
    val_df : pd.DataFrame or np.ndarray
        Validation features.
    val_target : pd.Series or np.ndarray
        Validation target values.
    callback_params : dict, optional
        Parameters for early stopping callback.
    model_obj : Any, optional
        Model object for XGBoost callback setup.
    """
    eval_set_configs = {
        "lgbm": ("eval_set", "tuple"),
        "hgboost": ("X_val", "separate"),
        "ngboost": ("X_val", "separate_Y"),
        "catboost": ("eval_set", "list_of_tuples"),
        "xgb": ("eval_set", "list_of_tuples"),
        "tabnet": ("eval_set", "list_of_tuples_values"),
        "pytorch": ("eval_set", "tuple"),
    }

    model_category = None
    model_type_lower = model_type_name.lower()
    for key in eval_set_configs:
        if key in model_type_lower:
            model_category = key
            break

    if model_category is None:
        return

    param_name, value_format = eval_set_configs[model_category]

    if value_format == "tuple":
        fit_params[param_name] = (val_df, val_target)
    elif value_format == "list_of_tuples":
        fit_params[param_name] = [(val_df, val_target)]
    elif value_format == "list_of_tuples_values":
        fit_params[param_name] = [(val_df.values, val_target.values if hasattr(val_target, "values") else val_target)]
    elif value_format == "separate":
        fit_params["X_val"] = val_df
        fit_params["y_val"] = val_target
    elif value_format == "separate_Y":
        fit_params["X_val"] = val_df
        fit_params["Y_val"] = val_target

    if callback_params:
        _setup_early_stopping_callback(model_category, fit_params, callback_params, model_obj)


def _setup_early_stopping_callback(model_category, fit_params, callback_params, model_obj=None):
    """Set up early stopping callback for the given model category."""
    no_callback_list_models = {"xgboost", "hgboost", "ngboost"}

    if model_category not in no_callback_list_models:
        if "callbacks" not in fit_params:
            fit_params["callbacks"] = []

    if model_category == "lgbm":
        es_callback = LightGBMCallback(**callback_params)
        fit_params["callbacks"].append(es_callback)
    elif model_category == "catboost":
        es_callback = CatBoostCallback(**callback_params)
        fit_params["callbacks"].append(es_callback)
    elif model_category == "xgboost" and model_obj is not None:
        es_callback = XGBoostCallback(**callback_params)
        callbacks = model_obj.get_params().get("callbacks", [])
        if callbacks is None:
            callbacks = []
        if es_callback not in callbacks:
            callbacks.append(es_callback)
        model_obj.set_params(callbacks=callbacks)


def _handle_oom_error(model_obj, model_type_name):
    """Handle GPU out-of-memory error by switching model to CPU."""
    if model_type_name in XGBOOST_MODEL_TYPES:
        if model_obj.get_params().get("device") in ("gpu", "cuda"):
            model_obj.set_params(device="cpu")
            logger.warning(f"{model_type_name} experienced OOM on gpu, switching to cpu...")
            return True
    elif model_type_name in CATBOOST_MODEL_TYPES:
        if model_obj.get_params().get("task_type") == "GPU":
            model_obj.set_params(task_type="CPU")
            logger.warning(f"{model_type_name} experienced OOM on gpu, switching to cpu...")
            return True
    elif model_type_name in LGBM_MODEL_TYPES:
        if model_obj.get_params().get("device_type") in ("gpu", "cuda"):
            model_obj.set_params(device_type="cpu")
            logger.warning(f"{model_type_name} experienced OOM on gpu, switching to cpu...")
            return True
    return False


def _train_model_with_fallback(
    model: Any,
    model_obj: Any,
    model_type_name: str,
    train_df: Union[pd.DataFrame, np.ndarray],
    train_target: Union[pd.Series, np.ndarray],
    fit_params: Dict[str, Any],
    verbose: bool = False,
) -> Tuple[Any, Optional[int]]:
    """Train model with automatic GPU->CPU fallback on OOM errors.

    Parameters
    ----------
    model : Any
        Model to train (may be a Pipeline).
    model_obj : Any
        The actual estimator object (extracted from Pipeline if needed).
    model_type_name : str
        Name of the model type (e.g., 'CatBoostClassifier').
    train_df : pd.DataFrame or np.ndarray
        Training features.
    train_target : pd.Series or np.ndarray
        Training target values.
    fit_params : dict
        Additional parameters for model.fit().
    verbose : bool, default=False
        Whether to log verbose output.

    Returns
    -------
    tuple
        (trained_model, best_iteration) where best_iteration may be None.
    """
    try:
        model.fit(train_df, train_target, **fit_params)
    except (RuntimeError, ValueError, OSError, MemoryError) as e:
        try_again = False
        error_str = str(e)

        if "out of memory" in error_str:
            try_again = _handle_oom_error(model_obj, model_type_name)

        elif "User defined callbacks are not supported for GPU" in error_str:
            if "callbacks" in fit_params:
                logger.warning(e)
                try_again = True
                del fit_params["callbacks"]

        elif "CUDA Tree Learner" in error_str:
            logger.warning("CUDA is not enabled in this LightGBM build. Falling back to CPU.")
            model.set_params(device_type="cpu")
            try_again = True

        if try_again:
            clean_ram()
            model.fit(train_df, train_target, **fit_params)
        else:
            raise e

    clean_ram()

    best_iter = None
    if model is not None:
        try:
            best_iter = get_model_best_iter(model_obj)
            if best_iter and verbose:
                logger.info(f"es_best_iter: {best_iter:_}")
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Could not get best iteration: {e}")

    return model, best_iter


def _filter_categorical_features(fit_params, train_df):
    """Filter cat_features to only include actual categorical columns in the DataFrame."""
    if "cat_features" not in fit_params:
        return

    if isinstance(train_df, pd.DataFrame):
        cat_columns = set(train_df.select_dtypes(["category", "object"]).columns)
        fit_params["cat_features"] = [col for col in fit_params["cat_features"] if col in cat_columns]
    elif isinstance(train_df, pl.DataFrame):
        cat_columns = set(train_df.select(pl.col(pl.Categorical)).columns)
        fit_params["cat_features"] = [col for col in fit_params["cat_features"] if col in cat_columns]


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Metrics and Reporting (imported from evaluation module)
# -----------------------------------------------------------------------------------------------------------------------------------------------------

# Import report functions - these are already in evaluation.py
from .evaluation import (
    report_model_perf,
    report_regression_model_perf,
    report_probabilistic_model_perf,
    get_model_feature_importances,
    plot_model_feature_importances,
)


def _compute_split_metrics(
    split_name: str,
    df,
    target,
    idx,
    model,
    model_type_name: str,
    model_name: str,
    metrics_dict: dict,
    group_ids=None,
    target_label_encoder=None,
    preds=None,
    probs=None,
    figsize=(15, 5),
    nbins: int = 10,
    print_report: bool = True,
    plot_file: str = "",
    show_perf_chart: bool = True,
    show_fi: bool = False,
    fi_kwargs: dict = None,
    subgroups: dict = None,
    custom_ice_metric=None,
    custom_rice_metric=None,
    details: str = "",
    has_other_splits: bool = False,
):
    """Unified metrics computation for train/val/test splits."""
    if df is None:
        return preds, probs, []

    columns = list(df.columns) if hasattr(df, "columns") else []
    df_prepared = _prepare_df_for_model(df, model_type_name)

    effective_show_fi = show_fi and not has_other_splits
    split_plot_file = f"{plot_file}_{split_name}" if plot_file else ""

    preds, probs = report_model_perf(
        targets=target,
        columns=columns,
        df=df_prepared,
        model_name=model_name,
        model=model,
        target_label_encoder=target_label_encoder,
        preds=preds,
        probs=probs,
        figsize=figsize,
        report_title=" ".join([split_name.upper(), details]).strip(),
        nbins=nbins,
        print_report=print_report,
        plot_file=split_plot_file,
        show_perf_chart=show_perf_chart,
        show_fi=effective_show_fi,
        fi_kwargs=fi_kwargs if fi_kwargs else {},
        subgroups=subgroups,
        subset_index=idx,
        custom_ice_metric=custom_ice_metric,
        custom_rice_metric=custom_rice_metric,
        metrics=metrics_dict,
        group_ids=group_ids[idx] if group_ids is not None and idx is not None else None,
    )
    return preds, probs, columns


def run_confidence_analysis(
    test_df: pd.DataFrame,
    test_target: np.ndarray,
    test_probs: np.ndarray,
    cat_features: List[str] = None,
    confidence_model_kwargs: dict = None,
    fit_params: dict = None,
    use_shap: bool = True,
    max_features: int = 20,
    cmap: str = "coolwarm",
    alpha: float = 0.5,
    title: str = "Confidence Analysis",
    ylabel: str = "Prediction Confidence",
    figsize: Tuple[float, float] = (10, 6),
    verbose: bool = False,
) -> Any:
    """Analyze which features most affect prediction confidence."""
    if test_df is None:
        return None

    if verbose:
        logger.info("Running confidence analysis...")

    if confidence_model_kwargs is None:
        confidence_model_kwargs = {}

    confidence_model = CatBoostRegressor(verbose=0, eval_fraction=0.1, task_type=("GPU" if CUDA_IS_AVAILABLE else "CPU"), **confidence_model_kwargs)

    fit_params_copy = {}
    if fit_params:
        fit_params_copy = copy.copy(fit_params)
        if "eval_set" in fit_params_copy:
            del fit_params_copy["eval_set"]

    if cat_features is not None:
        fit_params_copy["cat_features"] = cat_features
    elif "cat_features" not in fit_params_copy:
        fit_params_copy["cat_features"] = get_categorical_columns(test_df, include_string=False)

    fit_params_copy["plot"] = False

    confidence_targets = test_probs[np.arange(test_probs.shape[0]), test_target]

    clean_ram()
    confidence_model.fit(test_df, confidence_targets, **fit_params_copy)
    clean_ram()

    if use_shap:
        try:
            import shap
            import shap.utils.transformers

            shap.utils.transformers.is_transformers_lm = lambda model: False
        except (ImportError, AttributeError):
            pass
        explainer = shap.TreeExplainer(confidence_model)
        shap_values = explainer(test_df)
        shap.plots.beeswarm(
            shap_values,
            max_display=max_features,
            color=plt.get_cmap(cmap),
            alpha=alpha,
            color_bar_label=ylabel,
            show=False,
        )
        plt.xlabel(title)
        plt.show()
    else:
        plot_model_feature_importances(
            model=confidence_model,
            columns=list(test_df.columns),
            model_name=title,
            num_factors=max_features,
            figsize=(figsize[0] * 0.7, figsize[1] / 2),
        )

    return confidence_model


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Config Building Functions
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def _build_configs_from_params(
    # Data params
    df=None,
    train_df=None,
    val_df=None,
    test_df=None,
    target=None,
    train_target=None,
    val_target=None,
    test_target=None,
    train_idx=None,
    val_idx=None,
    test_idx=None,
    group_ids=None,
    sample_weight=None,
    drop_columns=None,
    default_drop_columns=None,
    target_label_encoder=None,
    skip_infinity_checks=False,
    # Control params
    verbose=False,
    use_cache=False,
    just_evaluate=False,
    compute_trainset_metrics=False,
    compute_valset_metrics=True,
    compute_testset_metrics=True,
    pre_pipeline=None,
    skip_pre_pipeline_transform=False,
    fit_params=None,
    callback_params=None,
    # Metrics params
    nbins=10,
    custom_ice_metric=None,
    custom_rice_metric=None,
    subgroups=None,
    train_details="",
    val_details="",
    test_details="",
    # Display params
    figsize=(15, 5),
    print_report=True,
    show_perf_chart=True,
    show_fi=True,
    fi_kwargs=None,
    plot_file="",
    data_dir="",
    models_subdir=MODELS_SUBDIR,
    display_sample_size=0,
    show_feature_names=False,
    # Naming params
    model_name="",
    model_name_prefix="",
    # Confidence params
    include_confidence_analysis=False,
    confidence_analysis_use_shap=True,
    confidence_analysis_max_features=6,
    confidence_analysis_cmap="bwr",
    confidence_analysis_alpha=0.9,
    confidence_analysis_ylabel="Feature value",
    confidence_analysis_title="Confidence of correct Test set predictions",
    confidence_model_kwargs=None,
    # Predictions params
    train_preds=None,
    train_probs=None,
    val_preds=None,
    val_probs=None,
    test_preds=None,
    test_probs=None,
):
    """Build config objects from old-style parameters."""
    merged_drop_columns = list(drop_columns or []) + list(default_drop_columns or [])

    data_config = DataConfig(
        df=df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target=target,
        train_target=train_target,
        val_target=val_target,
        test_target=test_target,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        group_ids=group_ids,
        sample_weight=sample_weight,
        drop_columns=merged_drop_columns,
        target_label_encoder=target_label_encoder,
        skip_infinity_checks=skip_infinity_checks,
    )

    control_config = TrainingControlConfig(
        verbose=verbose,
        use_cache=use_cache,
        just_evaluate=just_evaluate,
        compute_trainset_metrics=compute_trainset_metrics,
        compute_valset_metrics=compute_valset_metrics,
        compute_testset_metrics=compute_testset_metrics,
        pre_pipeline=pre_pipeline,
        skip_pre_pipeline_transform=skip_pre_pipeline_transform,
        fit_params=fit_params,
        callback_params=callback_params,
    )

    metrics_config = MetricsConfig(
        nbins=nbins,
        custom_ice_metric=custom_ice_metric,
        custom_rice_metric=custom_rice_metric,
        subgroups=subgroups,
        train_details=train_details,
        val_details=val_details,
        test_details=test_details,
    )

    display_config = DisplayConfig(
        figsize=figsize,
        print_report=print_report,
        show_perf_chart=show_perf_chart,
        show_fi=show_fi,
        fi_kwargs=fi_kwargs or {},
        plot_file=plot_file or "",
        data_dir=data_dir or "",
        models_subdir=models_subdir or "models",
        display_sample_size=display_sample_size,
        show_feature_names=show_feature_names,
    )

    naming_config = NamingConfig(
        model_name=model_name,
        model_name_prefix=model_name_prefix,
    )

    confidence_config = ConfidenceAnalysisConfig(
        include=include_confidence_analysis,
        use_shap=confidence_analysis_use_shap,
        max_features=confidence_analysis_max_features,
        cmap=confidence_analysis_cmap,
        alpha=confidence_analysis_alpha,
        ylabel=confidence_analysis_ylabel,
        title=confidence_analysis_title,
        model_kwargs=confidence_model_kwargs or {},
    )

    predictions_container = PredictionsContainer(
        train_preds=train_preds,
        train_probs=train_probs,
        val_preds=val_preds,
        val_probs=val_probs,
        test_preds=test_preds,
        test_probs=test_probs,
    )

    return data_config, control_config, metrics_config, display_config, naming_config, confidence_config, predictions_container


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Main Training Functions
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def train_and_evaluate_model(
    model: object,
    data: DataConfig,
    control: TrainingControlConfig,
    metrics: MetricsConfig,
    display: DisplayConfig,
    naming: NamingConfig,
    confidence: Optional[ConfidenceAnalysisConfig] = None,
    predictions: Optional[PredictionsContainer] = None,
    train_od_idx: Optional[np.ndarray] = None,
    val_od_idx: Optional[np.ndarray] = None,
    trainset_features_stats: Optional[dict] = None,
):
    """Train and evaluate a machine learning model with comprehensive metrics and optional caching.

    Parameters
    ----------
    model : object
        The model to train (sklearn estimator, Pipeline, etc.).
    data : DataConfig
        Input data configuration (DataFrames, targets, indices).
    control : TrainingControlConfig
        Training control flags (verbose, cache, metrics computation).
    metrics : MetricsConfig
        Metrics configuration (nbins, custom metrics, subgroups).
    display : DisplayConfig
        Display configuration (figsize, plot settings, paths).
    naming : NamingConfig
        Model naming configuration.
    confidence : ConfidenceAnalysisConfig, optional
        Confidence analysis configuration.
    predictions : PredictionsContainer, optional
        Pre-computed predictions (for just_evaluate mode).
    train_od_idx : np.ndarray, optional
        Training outlier detection indices.
    val_od_idx : np.ndarray, optional
        Validation outlier detection indices.
    trainset_features_stats : dict, optional
        Pre-computed feature statistics from training set.

    Returns
    -------
    tuple
        (result_namespace, train_df, val_df, test_df) where result_namespace contains
        model, predictions, metrics, and other training artifacts.
    """
    from IPython.display import display as ipython_display

    # Initialize optional configs with defaults
    if confidence is None:
        confidence = ConfidenceAnalysisConfig()
    if predictions is None:
        predictions = PredictionsContainer()

    # ---------------------------------------------------------------------------
    # Unpack data config
    # ---------------------------------------------------------------------------
    df = data.df
    train_df = data.train_df
    val_df = data.val_df
    test_df = data.test_df
    target = data.target
    train_target = data.train_target
    val_target = data.val_target
    test_target = data.test_target
    train_idx = data.train_idx
    val_idx = data.val_idx
    test_idx = data.test_idx
    group_ids = data.group_ids
    sample_weight = data.sample_weight
    drop_columns = list(data.drop_columns) if data.drop_columns else []
    target_label_encoder = data.target_label_encoder
    skip_infinity_checks = data.skip_infinity_checks

    # ---------------------------------------------------------------------------
    # Unpack control config
    # ---------------------------------------------------------------------------
    verbose = control.verbose
    use_cache = control.use_cache
    just_evaluate = control.just_evaluate
    compute_trainset_metrics = control.compute_trainset_metrics
    compute_valset_metrics = control.compute_valset_metrics
    compute_testset_metrics = control.compute_testset_metrics
    pre_pipeline = control.pre_pipeline
    skip_pre_pipeline_transform = control.skip_pre_pipeline_transform
    fit_params = control.fit_params
    callback_params = control.callback_params

    # ---------------------------------------------------------------------------
    # Unpack metrics config
    # ---------------------------------------------------------------------------
    nbins = metrics.nbins
    custom_ice_metric = metrics.custom_ice_metric
    custom_rice_metric = metrics.custom_rice_metric
    subgroups = metrics.subgroups
    train_details = metrics.train_details
    val_details = metrics.val_details
    test_details = metrics.test_details

    # ---------------------------------------------------------------------------
    # Unpack display config
    # ---------------------------------------------------------------------------
    figsize = display.figsize
    print_report = display.print_report
    show_perf_chart = display.show_perf_chart
    show_fi = display.show_fi
    fi_kwargs = dict(display.fi_kwargs) if display.fi_kwargs else {}
    plot_file = display.plot_file
    data_dir = display.data_dir
    models_subdir = display.models_subdir
    display_sample_size = display.display_sample_size
    show_feature_names = display.show_feature_names

    # ---------------------------------------------------------------------------
    # Unpack naming config
    # ---------------------------------------------------------------------------
    model_name = naming.model_name
    model_name_prefix = naming.model_name_prefix

    # ---------------------------------------------------------------------------
    # Unpack confidence config
    # ---------------------------------------------------------------------------
    include_confidence_analysis = confidence.include
    confidence_analysis_use_shap = confidence.use_shap
    confidence_analysis_max_features = confidence.max_features
    confidence_analysis_cmap = confidence.cmap
    confidence_analysis_alpha = confidence.alpha
    confidence_analysis_ylabel = confidence.ylabel
    confidence_analysis_title = confidence.title
    confidence_model_kwargs = dict(confidence.model_kwargs) if confidence.model_kwargs else {}

    # ---------------------------------------------------------------------------
    # Unpack predictions container
    # ---------------------------------------------------------------------------
    train_preds = predictions.train_preds
    train_probs = predictions.train_probs
    val_preds = predictions.val_preds
    val_probs = predictions.val_probs
    test_preds = predictions.test_preds
    test_probs = predictions.test_probs

    # ---------------------------------------------------------------------------
    # Begin original function logic
    # ---------------------------------------------------------------------------
    clean_ram()

    # default_drop_columns is no longer needed - merged into drop_columns
    default_drop_columns = []

    columns = []
    best_iter = None
    # Placeholder for future outlier detection functionality (currently unused but part of return API)
    test_is_inlier = None

    _orig_train_df = train_df
    _orig_val_df = val_df
    _orig_test_df = test_df

    real_drop_columns = _validate_infinity_and_columns(
        df=df,
        train_df=train_df,
        skip_infinity_checks=skip_infinity_checks,
        drop_columns=drop_columns,
        default_drop_columns=default_drop_columns,
    )

    if not custom_ice_metric:
        custom_ice_metric = partial(compute_probabilistic_multiclass_error, nbins=nbins)

    model_obj, model_type_name, model_name, plot_file, model_file_name = _setup_model_info_and_paths(
        model=model,
        model_name=model_name,
        model_name_prefix=model_name_prefix,
        plot_file=plot_file,
        data_dir=data_dir,
        models_subdir=models_subdir,
    )

    if use_cache and exists(model_file_name):
        logger.info(f"Loading model from file {model_file_name}")
        try:
            model, *_, pre_pipeline = joblib.load(model_file_name)
        except (EOFError, OSError, ModuleNotFoundError, pickle.UnpicklingError) as e:
            logger.warning(f"Failed to load cached model from {model_file_name}: {e}. Will retrain instead.")
            # Continue to training - model remains as originally passed

    if fit_params is None:
        fit_params = {}
    else:
        fit_params = copy.copy(fit_params)

    train_target, val_target, test_target = _extract_targets_from_indices(target, train_idx, val_idx, test_idx, train_target, val_target, test_target)

    if (df is not None) or (train_df is not None):
        if train_df is None:
            train_df = _subset_dataframe(df, train_idx, real_drop_columns)
        if val_df is None and val_idx is not None:
            val_df = _subset_dataframe(df, val_idx, real_drop_columns)

        trainset_features_stats = _compute_trainset_stats(train_df, trainset_features_stats, verbose)

    train_df, val_df = _apply_pre_pipeline_transforms(
        model=model,
        pre_pipeline=pre_pipeline,
        train_df=train_df,
        val_df=val_df,
        train_target=train_target,
        skip_pre_pipeline_transform=skip_pre_pipeline_transform,
        use_cache=use_cache,
        model_file_name=model_file_name,
        verbose=verbose,
    )

    if model is not None and pre_pipeline and not skip_pre_pipeline_transform:
        _orig_train_df = train_df
        if val_df is not None:
            _orig_val_df = val_df

    if val_df is not None:
        if isinstance(val_target, pl.Series):
            val_target = val_target.to_numpy()

        _setup_eval_set(model_type_name, fit_params, val_df, val_target, callback_params, model_obj)
        clean_ram()
    else:
        _disable_xgboost_early_stopping_if_needed(model_type_name, model_obj)

    if model is not None and fit_params:
        _filter_categorical_features(fit_params, train_df)

    if model is not None:
        if (not use_cache) or (not exists(model_file_name)):
            _setup_sample_weight(sample_weight, train_idx, model_obj, fit_params)
            if verbose:
                logger.info(f"{model_name} training dataset shape: {train_df.shape}")

            if display_sample_size:
                ipython_display(train_df.head(display_sample_size).style.set_caption(f"{model_name} features head"))
                ipython_display(train_df.tail(display_sample_size).style.set_caption(f"{model_name} features tail"))

            if train_df is not None:
                report_title = f"Training {model_name} model on {train_df.shape[1]} feature(s)"
                if show_feature_names:
                    report_title += ": " + ", ".join(train_df.columns.to_list())
                report_title += f", {len(train_df):_} records"

            train_df, fit_params = _prepare_train_df_for_fitting(train_df, model, model_type_name, fit_params)

            clean_ram()
            if verbose:
                logger.info("Training the model...")

            if isinstance(train_target, pl.Series):
                train_target = train_target.to_numpy()

            if not just_evaluate:
                model, best_iter = _train_model_with_fallback(
                    model=model,
                    model_obj=model_obj,
                    model_type_name=model_type_name,
                    train_df=train_df,
                    train_target=train_target,
                    fit_params=fit_params,
                    verbose=verbose,
                )

            model_name = _update_model_name_after_training(model_name, len(train_df), train_details, best_iter)

    metrics = {"train": {}, "val": {}, "test": {}, "best_iter": best_iter}

    if compute_trainset_metrics or compute_valset_metrics or compute_testset_metrics:
        if verbose:
            logger.info("Computing model's performance...")

        common_metrics_params = dict(
            model=model,
            model_type_name=model_type_name,
            model_name=model_name,
            group_ids=group_ids,
            target_label_encoder=target_label_encoder,
            figsize=figsize,
            nbins=nbins,
            print_report=print_report,
            plot_file=plot_file,
            show_perf_chart=show_perf_chart,
            show_fi=show_fi,
            fi_kwargs=fi_kwargs,
            subgroups=subgroups,
            custom_ice_metric=custom_ice_metric,
            custom_rice_metric=custom_rice_metric,
        )

        has_val = val_idx is not None or val_df is not None
        has_test = test_idx is not None or test_df is not None

        splits_config = [
            (
                "train",
                train_df,
                train_target,
                train_idx,
                train_preds,
                train_probs,
                train_details,
                compute_trainset_metrics and (train_idx is not None or train_df is not None),
            ),
            (
                "val",
                val_df,
                val_target,
                val_idx,
                val_preds,
                val_probs,
                val_details,
                compute_valset_metrics and ((val_idx is not None and len(val_idx) > 0) or val_df is not None),
            ),
        ]

        for split_name, split_df, split_target, split_idx, split_preds, split_probs, split_details, should_compute in splits_config:
            if should_compute:
                if split_name == "train":
                    has_other = has_val or has_test
                else:
                    has_other = has_test

                preds_result, probs_result, columns = _compute_split_metrics(
                    split_name=split_name,
                    df=split_df,
                    target=split_target,
                    idx=split_idx,
                    metrics_dict=metrics[split_name],
                    preds=split_preds,
                    probs=split_probs,
                    details=split_details,
                    has_other_splits=has_other,
                    **common_metrics_params,
                )
                if split_name == "train":
                    train_preds, train_probs = preds_result, probs_result
                else:
                    val_preds, val_probs = preds_result, probs_result

        if compute_testset_metrics and ((test_idx is not None and len(test_idx) > 0) or test_df is not None):
            if (df is not None) or (test_df is not None):
                # Free memory before test evaluation (train_df may not exist if just_evaluate=True)
                try:
                    if train_df is not None:
                        del train_df
                except NameError:
                    pass  # train_df was never assigned (e.g., just_evaluate=True with no training)
                clean_ram()

            test_df, test_target, columns = _prepare_test_split(
                df=df,
                test_df=test_df,
                test_idx=test_idx,
                test_target=test_target,
                target=target,
                real_drop_columns=real_drop_columns,
                model=model,
                pre_pipeline=pre_pipeline,
                skip_pre_pipeline_transform=skip_pre_pipeline_transform,
            )
            if test_df is not None:
                _orig_test_df = test_df

            test_preds, test_probs, columns = _compute_split_metrics(
                split_name="test",
                df=test_df,
                target=test_target,
                idx=test_idx,
                metrics_dict=metrics["test"],
                preds=test_preds,
                probs=test_probs,
                details=test_details,
                has_other_splits=False,
                **common_metrics_params,
            )

            if include_confidence_analysis and test_df is not None:
                run_confidence_analysis(
                    test_df=test_df,
                    test_target=test_target,
                    test_probs=test_probs,
                    cat_features=fit_params.get("cat_features") if fit_params else None,
                    confidence_model_kwargs=confidence_model_kwargs,
                    fit_params=fit_params if model_type_name == "CatBoostRegressor" else None,
                    use_shap=confidence_analysis_use_shap,
                    max_features=confidence_analysis_max_features,
                    cmap=confidence_analysis_cmap,
                    alpha=confidence_analysis_alpha,
                    title=confidence_analysis_title,
                    ylabel=confidence_analysis_ylabel,
                    figsize=figsize,
                    verbose=verbose,
                )

    clean_ram()

    return (
        SimpleNamespace(
            model=model,
            test_preds=test_preds,
            test_probs=test_probs,
            test_target=test_target,
            test_is_inlier=test_is_inlier,
            val_preds=val_preds,
            val_probs=val_probs,
            val_target=val_target,
            train_preds=train_preds,
            train_probs=train_probs,
            train_target=train_target,
            metrics=metrics,
            columns=columns,
            pre_pipeline=pre_pipeline,
            train_od_idx=train_od_idx,
            val_od_idx=val_od_idx,
            trainset_features_stats=trainset_features_stats,
        ),
        _orig_train_df,
        _orig_val_df,
        _orig_test_df,
    )


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Configure Training Params Helper Functions
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def _configure_xgboost_params(
    configs,
    cpu_configs,
    use_regression: bool,
    prefer_cpu_for_xgboost: bool,
    prefer_calibrated_classifiers: bool,
    use_flaml_zeroshot: bool,
    xgboost_verbose,
    metamodel_func,
):
    """Configure XGBoost model parameters."""
    xgb_configs = cpu_configs if prefer_cpu_for_xgboost else configs

    if use_regression:
        model_cls = flaml_zeroshot.XGBRegressor if use_flaml_zeroshot else XGBRegressor
        model = metamodel_func(model_cls(**xgb_configs.XGB_GENERAL_PARAMS))
    else:
        model_cls = flaml_zeroshot.XGBClassifier if use_flaml_zeroshot else XGBClassifier
        params = xgb_configs.XGB_CALIB_CLASSIF if prefer_calibrated_classifiers else xgb_configs.XGB_GENERAL_CLASSIF
        model = model_cls(**params)

    return dict(model=model, fit_params=dict(verbose=xgboost_verbose))


def _configure_lightgbm_params(
    configs,
    cpu_configs,
    use_regression: bool,
    prefer_cpu_for_lightgbm: bool,
    prefer_calibrated_classifiers: bool,
    use_flaml_zeroshot: bool,
    metamodel_func,
):
    """Configure LightGBM model parameters."""
    lgb_configs = cpu_configs if prefer_cpu_for_lightgbm else configs

    if use_regression:
        model_cls = flaml_zeroshot.LGBMRegressor if use_flaml_zeroshot else LGBMRegressor
        model = metamodel_func(model_cls(**lgb_configs.LGB_GENERAL_PARAMS))
    else:
        model_cls = flaml_zeroshot.LGBMClassifier if use_flaml_zeroshot else LGBMClassifier
        model = model_cls(**lgb_configs.LGB_GENERAL_PARAMS)

    fit_params = {}
    if prefer_calibrated_classifiers and not use_regression:
        fit_params["eval_metric"] = cpu_configs.lgbm_integral_calibration_error

    return dict(model=model, fit_params=fit_params)


def _configure_mlp_params(
    configs,
    config_params: dict,
    use_regression: bool,
    metamodel_func: callable,
) -> dict:
    """Configure MLP (PyTorch Lightning) model parameters."""
    mlp_kwargs = config_params.get("mlp_kwargs", {})

    mlp_network_params = dict(
        nlayers=20,
        first_layer_num_neurons=100,
        min_layer_neurons=1,
        neurons_by_layer_arch=MLPNeuronsByLayerArchitecture.Declining,
        consec_layers_neurons_ratio=1.5,
        activation_function=torch.nn.LeakyReLU,
        weights_init_fcn=partial(nn.init.kaiming_normal_, nonlinearity="leaky_relu", a=0.01),
        dropout_prob=0.15,
        inputs_dropout_prob=0.002,
        use_batchnorm=True,
    )
    if mlp_kwargs:
        mlp_network_params.update(mlp_kwargs.get("network_params", {}))

    mlp_general_params = configs.MLP_GENERAL_PARAMS.copy()
    if use_regression:
        mlp_general_params["model_params"] = mlp_general_params.get("model_params", {}).copy()
        mlp_general_params["model_params"]["loss_fn"] = F.mse_loss
        mlp_general_params["datamodule_params"] = mlp_general_params.get("datamodule_params", {}).copy()
        mlp_general_params["datamodule_params"]["labels_dtype"] = torch.float32
        mlp_model = PytorchLightningRegressor(network_params=mlp_network_params, **mlp_general_params)
    else:
        mlp_model = PytorchLightningClassifier(network_params=mlp_network_params, **mlp_general_params)

    return dict(model=metamodel_func(mlp_model))


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Configure Training Parameters
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def configure_training_params(
    df: pd.DataFrame = None,
    train_df: pd.DataFrame = None,
    test_df: pd.DataFrame = None,
    val_df: pd.DataFrame = None,
    target: pd.Series = None,
    target_label_encoder: object = None,
    train_target: pd.Series = None,
    test_target: pd.Series = None,
    val_target: pd.Series = None,
    train_idx: np.ndarray = None,
    val_idx: np.ndarray = None,
    test_idx: np.ndarray = None,
    cat_features: list = None,
    fairness_features: Sequence = None,
    cont_nbins: int = 6,
    fairness_min_pop_cat_thresh: Union[float, int] = 1000,
    use_robust_eval_metric: bool = False,
    sample_weight: np.ndarray = None,
    prefer_gpu_configs: bool = True,
    nbins: int = 10,
    use_regression: bool = False,
    verbose: bool = True,
    rfecv_model_verbose: bool = True,
    prefer_cpu_for_lightgbm: bool = True,
    prefer_cpu_for_xgboost: bool = False,
    xgboost_verbose: Union[int, bool] = False,
    cb_fit_params: dict = None,
    prefer_calibrated_classifiers: bool = True,
    default_regression_scoring: dict = None,
    default_classification_scoring: dict = None,
    train_details: str = "",
    val_details: str = "",
    test_details: str = "",
    group_ids: np.ndarray = None,
    model_name: str = "",
    common_params: dict = None,
    config_params: dict = None,
    metamodel_func: callable = None,
    use_flaml_zeroshot: bool = False,
    _precomputed_fairness_subgroups: dict = None,
):
    """Configure training parameters for all model types."""
    from .configs import LinearModelConfig

    def _identity(x):
        return x

    if metamodel_func is None:
        metamodel_func = _identity

    if default_regression_scoring is None:
        default_regression_scoring = dict(score_func=mean_absolute_error, response_method="predict", greater_is_better=False)

    if default_classification_scoring is None:
        default_classification_scoring = dict(score_func=fast_roc_auc, response_method="predict_proba", greater_is_better=True)

    if common_params is None:
        common_params = {}
    if config_params is None:
        config_params = {}
    if fairness_features is None:
        fairness_features = []
    if cb_fit_params is None:
        cb_fit_params = {}

    if not use_regression:
        if "catboost_custom_classif_metrics" not in config_params:
            nlabels = len(np.unique(target))
            if nlabels > 2:
                catboost_custom_classif_metrics = ["AUC", "PRAUC"]
            else:
                catboost_custom_classif_metrics = ["AUC", "PRAUC", "BrierScore"]
            config_params["catboost_custom_classif_metrics"] = catboost_custom_classif_metrics

    subgroups = _precomputed_fairness_subgroups
    if subgroups is None and fairness_features:
        for next_df in (df, train_df):
            if next_df is not None:
                subgroups = create_fairness_subgroups(
                    next_df,
                    features=fairness_features,
                    cont_nbins=cont_nbins,
                    min_pop_cat_thresh=fairness_min_pop_cat_thresh,
                )
                break

    if use_robust_eval_metric and subgroups is not None:
        indexed_subgroups = create_fairness_subgroups_indices(
            subgroups=subgroups, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, group_weights={}, cont_nbins=cont_nbins
        )
    else:
        indexed_subgroups = None

    cpu_configs = get_training_configs(has_gpu=False, subgroups=indexed_subgroups, **config_params)
    gpu_configs = get_training_configs(has_gpu=None, subgroups=indexed_subgroups, **config_params)

    train_df_size = get_df_memory_consumption(train_df)
    val_df_size = get_df_memory_consumption(val_df) if val_df is not None else 0
    data_size_gb = (train_df_size + val_df_size) / (1024**3)

    all_gpus = get_gpuinfo_gpu_info()
    single_gpu_limits = compute_total_gpus_ram(all_gpus)

    data_fits_gpu_ram = (GPU_VRAM_SAFE_SATURATION_LIMIT * data_size_gb + GPU_VRAM_SAFE_FREE_LIMIT_GB) < single_gpu_limits.get("gpu_max_ram_total", 0)

    cb_devices = config_params.get("cb_kwargs", {}).get("devices")
    if cb_devices:
        multi_gpu_limits = compute_total_gpus_ram(parse_catboost_devices(cb_devices, all_gpus=all_gpus))
        data_fits_cb_gpu_ram = (GPU_VRAM_SAFE_SATURATION_LIMIT * data_size_gb + GPU_VRAM_SAFE_FREE_LIMIT_GB) < multi_gpu_limits.get("gpus_ram_total", 0)
    else:
        data_fits_cb_gpu_ram = data_fits_gpu_ram

    logger.info(f"data_fits_gpu_ram={data_fits_gpu_ram}, data_fits_cb_gpu_ram={data_fits_cb_gpu_ram}, cb_devices={cb_devices}")

    configs = gpu_configs if (prefer_gpu_configs and data_fits_gpu_ram) else cpu_configs
    cb_configs = gpu_configs if (prefer_gpu_configs and data_fits_cb_gpu_ram) else cpu_configs

    common_params_result = dict(
        nbins=nbins,
        subgroups=subgroups,
        sample_weight=sample_weight,
        df=df,
        train_df=train_df,
        test_df=test_df,
        val_df=val_df,
        target=target,
        train_target=train_target,
        test_target=test_target,
        val_target=val_target,
        train_idx=train_idx,
        test_idx=test_idx,
        val_idx=val_idx,
        target_label_encoder=target_label_encoder,
        custom_ice_metric=configs.integral_calibration_error,
        custom_rice_metric=configs.final_integral_calibration_error,
        train_details=train_details,
        val_details=val_details,
        test_details=test_details,
        group_ids=group_ids,
        model_name=model_name,
    )
    if common_params:
        common_params_result.update(common_params)
    common_params = common_params_result

    cb_params = dict(
        model=(
            metamodel_func(CatBoostRegressor(**cb_configs.CB_REGR))
            if use_regression
            else CatBoostClassifier(**(cb_configs.CB_CALIB_CLASSIF if prefer_calibrated_classifiers else cb_configs.CB_CLASSIF))
        ),
        fit_params=dict(plot=verbose, cat_features=cat_features, **cb_fit_params),
    )

    hgb_params = dict(
        model=metamodel_func(
            (HistGradientBoostingRegressor(**configs.HGB_GENERAL_PARAMS) if use_regression else HistGradientBoostingClassifier(**configs.HGB_GENERAL_PARAMS)),
        )
    )

    xgb_params = _configure_xgboost_params(
        configs=configs,
        cpu_configs=cpu_configs,
        use_regression=use_regression,
        prefer_cpu_for_xgboost=prefer_cpu_for_xgboost,
        prefer_calibrated_classifiers=prefer_calibrated_classifiers,
        use_flaml_zeroshot=use_flaml_zeroshot,
        xgboost_verbose=xgboost_verbose,
        metamodel_func=metamodel_func,
    )

    lgb_params = _configure_lightgbm_params(
        configs=configs,
        cpu_configs=cpu_configs,
        use_regression=use_regression,
        prefer_cpu_for_lightgbm=prefer_cpu_for_lightgbm,
        prefer_calibrated_classifiers=prefer_calibrated_classifiers,
        use_flaml_zeroshot=use_flaml_zeroshot,
        metamodel_func=metamodel_func,
    )

    mlp_params = _configure_mlp_params(
        configs=configs,
        config_params=config_params,
        use_regression=use_regression,
        metamodel_func=metamodel_func,
    )

    ngb_params = dict(
        model=(
            metamodel_func(
                (NGBRegressor(**configs.NGB_GENERAL_PARAMS) if use_regression else NGBClassifier(**configs.NGB_GENERAL_PARAMS)),
            )
        ),
        fit_params=dict(early_stopping_rounds=config_params.get("early_stopping_rounds")),
    )

    # Linear models - create all variants in a loop
    linear_model_params = {}
    for model_type in LINEAR_MODEL_TYPES:
        linear_model_params[model_type] = dict(
            model=metamodel_func(create_linear_model(model_type, LinearModelConfig(model_type=model_type), use_regression=use_regression))
        )
    linear_params = linear_model_params["linear"]
    ridge_params = linear_model_params["ridge"]
    lasso_params = linear_model_params["lasso"]
    elasticnet_params = linear_model_params["elasticnet"]
    huber_params = linear_model_params["huber"]
    ransac_params = linear_model_params["ransac"]
    sgd_params = linear_model_params["sgd"]

    # RFECV setup
    rfecv_params = configs.COMMON_RFECV_PARAMS.copy()
    cb_rfecv_params = cb_configs.COMMON_RFECV_PARAMS.copy()

    if not common_params.get("show_perf_chart", True):
        rfecv_params["optimizer_plotting"] = "No"
        cb_rfecv_params["optimizer_plotting"] = "No"

    if "rfecv_params" in common_params:
        custom_rfecv_params = common_params.pop("rfecv_params")
        rfecv_params.update(custom_rfecv_params)
        cb_rfecv_params.update(custom_rfecv_params)

    if use_regression:
        rfecv_scoring = make_scorer(**default_regression_scoring)
    else:
        if prefer_calibrated_classifiers:

            def fs_and_hpt_integral_calibration_error(*args, **kwargs):
                return configs.fs_and_hpt_integral_calibration_error(*args, **kwargs, verbose=rfecv_model_verbose)

            rfecv_scoring = make_scorer(
                score_func=fs_and_hpt_integral_calibration_error,
                response_method="predict_proba",
                greater_is_better=False,
            )
        else:
            rfecv_scoring = make_scorer(**default_classification_scoring)

    params = (cb_configs.CB_REGR if use_regression else (cb_configs.CB_CALIB_CLASSIF if prefer_calibrated_classifiers else cb_configs.CB_CLASSIF)).copy()

    cb_rfecv = RFECV(
        estimator=(metamodel_func(CatBoostRegressor(**params)) if use_regression else CatBoostClassifier(**params)),
        fit_params=dict(plot=rfecv_model_verbose > 1),
        cat_features=cat_features,
        scoring=rfecv_scoring,
        **cb_rfecv_params,
    )

    lgb_fit_params = dict(eval_metric=cpu_configs.lgbm_integral_calibration_error) if prefer_calibrated_classifiers else {}

    lgb_rfecv = RFECV(
        estimator=(
            metamodel_func((flaml_zeroshot.LGBMRegressor if use_flaml_zeroshot else LGBMRegressor)(**configs.LGB_GENERAL_PARAMS))
            if use_regression
            else (flaml_zeroshot.LGBMClassifier if use_flaml_zeroshot else LGBMClassifier)(**configs.LGB_GENERAL_PARAMS)
        ),
        fit_params=lgb_fit_params,
        cat_features=cat_features,
        scoring=rfecv_scoring,
        **rfecv_params,
    )

    xgb_rfecv = RFECV(
        estimator=(
            metamodel_func((flaml_zeroshot.XGBRegressor if use_flaml_zeroshot else XGBRegressor)(**configs.XGB_GENERAL_PARAMS))
            if use_regression
            else (flaml_zeroshot.XGBClassifier if use_flaml_zeroshot else XGBClassifier)(
                **(configs.XGB_CALIB_CLASSIF if prefer_calibrated_classifiers else configs.XGB_GENERAL_CLASSIF)
            )
        ),
        fit_params=dict(verbose=False),
        cat_features=cat_features,
        scoring=rfecv_scoring,
        **rfecv_params,
    )

    models_params = dict(
        cb=cb_params,
        lgb=lgb_params,
        xgb=xgb_params,
        hgb=hgb_params,
        mlp=mlp_params,
        ngb=ngb_params,
        linear=linear_params,
        ridge=ridge_params,
        lasso=lasso_params,
        elasticnet=elasticnet_params,
        huber=huber_params,
        ransac=ransac_params,
        sgd=sgd_params,
    )

    return (
        common_params,
        models_params,
        cb_rfecv,
        lgb_rfecv,
        xgb_rfecv,
        cpu_configs,
        gpu_configs,
    )


__all__ = [
    "train_and_evaluate_model",
    "configure_training_params",
    "_build_configs_from_params",
    "run_confidence_analysis",
]
