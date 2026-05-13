"""Suite setup and configuration helpers extracted from ``core/utils.py``.

Config normalization, outlier detection, model directory setup,
common params building, pre-pipeline construction, Polars conversion,
fairness subgroup creation, and metadata lifecycle.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import category_encoders as ce

from ..configs import (
    FeatureSelectionConfig,
    TargetTypes,
)
from ..splitting import make_train_test_split

logger = logging.getLogger(__name__)


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
    baseline_rss_mb: float = 0.0,
    df_size_mb: float = 0.0,
    targets_for_classbalance: Optional[Dict[str, Any]] = None,
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

    # 2026-04-24: sklearn outlier detectors (IsolationForest / LOF /
    # EllipticEnvelope / OneClassSVM) all call ``validate_data`` /
    # ``check_array`` at fit time which attempts to coerce the input
    # to a numeric numpy array. Any non-numeric column (string, text,
    # categorical, embedding list, etc.) crashes with
    #   ValueError: could not convert string to float: 'A' / 'stream java java'
    # The detector is fit on FEATURES ONLY to find structural outliers --
    # dropping non-numeric columns before fit matches what sklearn would
    # expect the caller to pre-process upstream. Recompute numeric view
    # on each fit/predict call so polars and pandas paths are symmetric.
    def _numeric_only_view(df_):
        if isinstance(df_, pl.DataFrame):
            numeric_cols = [
                name for name, dt in df_.schema.items()
                if dt.is_numeric() or dt == pl.Boolean
            ]
            return df_.select(numeric_cols) if len(numeric_cols) != len(df_.columns) else df_
        if hasattr(df_, "select_dtypes"):
            return df_.select_dtypes(include=["number", "bool"])
        return df_

    _train_numeric = _numeric_only_view(train_df)
    # Fit on training features only (unsupervised - no target needed)
    outlier_detector.fit(_train_numeric)

    # Predict on training set
    is_inlier = outlier_detector.predict(_train_numeric)
    train_od_idx = is_inlier == 1

    filtered_train_df = train_df
    filtered_train_idx = train_idx

    def _filter_df_by_mask(_df, mask):
        """Boolean-mask filter that works for both pandas and Polars."""
        if isinstance(_df, pl.DataFrame):
            return _df.filter(pl.Series(mask))
        return _df.loc[mask]

    train_kept = train_od_idx.sum()
    if train_kept < len(train_df):
        # Class-balance pre-check 2026-04-27 (batch 3): when OD is fit on
        # features that include a label-correlated leak feature
        # (e.g. ``num_leak`` or any feature that's effectively the
        # target with noise), the unsupervised detector flags the
        # rare-class rows as outliers and removes them all. The
        # surviving train then has only one unique target value and
        # downstream CB/XGB classification crashes deep in C++ with
        # ``Target contains only one unique value``. Detect this
        # before propagating the filter; if any per-target check would
        # destroy class diversity, silently SKIP the OD filter for
        # train (val handled below). Fit stays intact for diagnostic
        # logging via ``train_od_idx``.
        _od_destroys_classes = False
        if targets_for_classbalance:
            for _tn, _tv in targets_for_classbalance.items():
                if _tv is None:
                    continue
                try:
                    _y_pre = (
                        _tv[train_idx]
                        if isinstance(_tv, (np.ndarray, pl.Series))
                        else _tv.iloc[train_idx]
                    )
                    _y_post = (
                        _tv[train_idx[train_od_idx]]
                        if isinstance(_tv, (np.ndarray, pl.Series))
                        else _tv.iloc[train_idx[train_od_idx]]
                    )
                    _arr_pre = np.asarray(_y_pre)
                    _arr_post = np.asarray(_y_post)
                    _flat_pre = _arr_pre.flatten() if _arr_pre.ndim > 1 else _arr_pre
                    _flat_post = _arr_post.flatten() if _arr_post.ndim > 1 else _arr_post
                    if len(np.unique(_flat_pre)) >= 2 and len(np.unique(_flat_post)) < 2:
                        _od_destroys_classes = True
                        logger.error(
                            "Outlier detection would eliminate the entire minority "
                            "class from train target '%s' (pre-OD unique=%d, post-OD "
                            "unique=%d). Typical cause: a feature highly correlated "
                            "with the target (e.g. label-leak feature) drives the "
                            "unsupervised OD to flag the rare class as outliers. "
                            "Skipping OD filter for train; original train_df retained.",
                            _tn,
                            len(np.unique(_flat_pre)),
                            len(np.unique(_flat_post)),
                        )
                        break
                except Exception as _exc:
                    logger.debug("Class-balance pre-check failed for target %s: %s", _tn, _exc)
        if not _od_destroys_classes:
            logger.info("Outlier rejection: %d train samples -> %d kept.", len(train_df), train_kept)
            filtered_train_df = _filter_df_by_mask(train_df, train_od_idx)
            filtered_train_idx = train_idx[train_od_idx]
        else:
            # Reset train_kept and train_od_idx so the min_keep guard below
            # sees the unfiltered count, and the downstream polars-fastpath
            # filter at core.py:~2758 (``train_df_polars.filter(...)``)
            # treats it as a no-op (all-True mask = keep all rows).
            train_kept = len(train_df)
            train_od_idx = np.ones(len(train_df), dtype=bool)

    # Guard against catastrophic outlier-detector misconfiguration where
    # ~every sample is flagged as an outlier. Previously this silently
    # produced a 0-row train set and failed 5+ minutes later deep inside
    # CatBoost/LightGBM with opaque "X is empty" / shape errors. Fail
    # fast and loud instead.
    min_keep = max(1, int(len(train_df) * 0.01))  # need >=1% AND >=1 row
    if train_kept < min_keep:
        raise ValueError(
            f"Outlier detector rejected {len(train_df) - train_kept:_} of {len(train_df):_} "
            f"train samples, leaving only {train_kept:_} rows (< {min_keep:_}, 1% of input). "
            f"The detector is likely misconfigured (e.g. contamination too high, trained on "
            f"unrepresentative data, or a sign convention bug). Training cannot proceed."
        )

    # Predict on validation set if requested
    filtered_val_df = val_df
    filtered_val_idx = val_idx
    val_od_idx = None

    if val_df is not None and od_val_set:
        is_inlier = outlier_detector.predict(_numeric_only_view(val_df))
        val_od_idx = is_inlier == 1
        val_kept = val_od_idx.sum()
        # Class-balance pre-check on val (mirror of the train-side check
        # above). If OD would eliminate the entire minority class from
        # val, skip the filter -- keep the unfiltered val_set so eval
        # has class-diverse data.
        if targets_for_classbalance and val_kept < len(val_df) and val_idx is not None:
            for _tn, _tv in targets_for_classbalance.items():
                if _tv is None:
                    continue
                try:
                    _y_pre = (
                        _tv[val_idx]
                        if isinstance(_tv, (np.ndarray, pl.Series))
                        else _tv.iloc[val_idx]
                    )
                    _y_post = (
                        _tv[val_idx[val_od_idx]]
                        if isinstance(_tv, (np.ndarray, pl.Series))
                        else _tv.iloc[val_idx[val_od_idx]]
                    )
                    _arr_pre = np.asarray(_y_pre)
                    _arr_post = np.asarray(_y_post)
                    _flat_pre = _arr_pre.flatten() if _arr_pre.ndim > 1 else _arr_pre
                    _flat_post = _arr_post.flatten() if _arr_post.ndim > 1 else _arr_post
                    if len(np.unique(_flat_pre)) >= 2 and len(np.unique(_flat_post)) < 2:
                        logger.error(
                            "Outlier detection would eliminate the entire minority "
                            "class from VAL target '%s' (pre-OD unique=%d, post-OD "
                            "unique=%d). Skipping OD filter for val; original "
                            "val_df retained for evaluation.",
                            _tn,
                            len(np.unique(_flat_pre)),
                            len(np.unique(_flat_post)),
                        )
                        # Reset OD effect on val. Set val_od_idx to all-True
                        # mask so the downstream polars filter at
                        # core.py:~2760 stays a no-op (keep all rows).
                        val_kept = len(val_df)
                        val_od_idx = np.ones(len(val_df), dtype=bool)
                        break
                except Exception as _exc:
                    logger.debug("Class-balance pre-check on val failed for target %s: %s", _tn, _exc)
        # Symmetric of the train-side ``min_keep`` guard at line ~1021.
        # If OD rejected almost all val rows (typically because train
        # was fit on a very different distribution and OD flags every
        # val row as an outlier), don't propagate a 0-row val_df:
        # downstream pre_pipeline / eval_set / metrics paths cope poorly
        # with empty val (4 separate "if len(val_df)==0: skip" guards
        # in trainer.py historically masked this). Log and keep the
        # ORIGINAL unfiltered val_df so evaluation has data; the user
        # sees a clear error in the log to investigate fit-distribution
        # mismatch between train and val.
        val_min_keep = max(1, int(len(val_df) * 0.01))
        if val_kept < val_min_keep:
            logger.error(
                "Outlier detector rejected %d of %d val samples, leaving "
                "only %d rows (< %d, 1%% floor). Continuing with the "
                "ORIGINAL (unfiltered) val_set so downstream evaluation "
                "has data; investigate contamination / fit-distribution "
                "mismatch between train and val.",
                len(val_df) - val_kept, len(val_df), val_kept, val_min_keep,
            )
            # Reset OD effect on val: keep the raw val_df / val_idx and
            # mark val_od_idx as None so downstream callers see "no OD
            # applied to val" cleanly.
            filtered_val_df = val_df
            filtered_val_idx = val_idx
            val_od_idx = None
        elif val_kept < len(val_df):
            logger.info("Outlier rejection: %d val samples -> %d kept.", len(val_df), val_kept)
            filtered_val_df = _filter_df_by_mask(val_df, val_od_idx)
            filtered_val_idx = val_idx[val_od_idx]

    baseline_rss_mb = maybe_clean_ram_and_gpu(baseline_rss_mb, df_size_mb, verbose=verbose, reason="post-outlier-detection")
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
    save_charts: bool = True,
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

    # Falsy check (not `is not None`) -- an empty string `data_dir=""` means "no
    # persistence", same as `None`. Treating "" as truthy would create a
    # relative "./charts" / "./models" leak in the CWD. Old artifacts from such
    # leaks can even be loaded back from disk on subsequent runs, which caused
    # a hard-to-diagnose sklearn 1.8 `SimpleImputer._fill_dtype` missing crash
    # when the leaked pickle was from sklearn 1.7.
    if data_dir and save_charts:
        plot_file = join(data_dir, "charts", *parts) + os.path.sep
        ensure_dir_exists(plot_file)
    else:
        plot_file = None

    if data_dir and models_dir:
        model_file = join(data_dir, models_dir, *parts) + os.path.sep
        ensure_dir_exists(model_file)
    else:
        model_file = None

    return plot_file, model_file




def _build_common_params_for_target(
    common_params_dict: Dict[str, Any],
    trainset_features_stats: Optional[Dict],
    plot_file: Optional[str],
    train_od_idx: Optional[np.ndarray],
    val_od_idx: Optional[np.ndarray],
    current_train_target: Optional[Any],
    current_val_target: Optional[Any],
    outlier_detector: Optional[Any],
    behavior_config: "TrainingBehaviorConfig",
    fairness_subgroups: Optional[Dict],
) -> Tuple[Dict[str, Any], "TrainingBehaviorConfig"]:
    """
    Build common_params and behavior_config for select_target call.

    Args:
        common_params_dict: Internal dict assembled from typed configs at the
            suite entry. Carries reporting/scaler/imputer/encoder fields down
            to the deep dict-key consumers in trainer.py. Built internally
            from the typed configs; no external dict pass-through on the
            suite signature.
        trainset_features_stats: Computed feature statistics
        plot_file: Path for saving plots
        train_od_idx: Outlier detection indices for training set
        val_od_idx: Outlier detection indices for validation set
        current_train_target: Training targets (filtered if OD applied)
        current_val_target: Validation targets (filtered if OD applied)
        outlier_detector: Outlier detector object (or None)
        behavior_config: Training behavior configuration
        fairness_subgroups: Pre-computed fairness subgroups

    Returns:
        Tuple containing:
            - od_common_params: Dict with common parameters for model training.
            - current_behavior_config: TrainingBehaviorConfig, possibly with
              _precomputed_fairness_subgroups attached.
    """
    # Add pre-computed fairness subgroups to behavior_config (extra fields allowed by BaseConfig)
    if fairness_subgroups is not None:
        current_behavior_config = behavior_config.model_copy(
            update={"_precomputed_fairness_subgroups": fairness_subgroups}
        )
    else:
        current_behavior_config = behavior_config

    # Build common_params dict
    # Filter out train_target/val_target so they don't conflict when OD applies.
    filtered_params = {k: v for k, v in common_params_dict.items() if k not in ("train_target", "val_target")}
    od_common_params = dict(
        trainset_features_stats=trainset_features_stats,
        plot_file=plot_file,
        train_od_idx=train_od_idx,  # Pass for metadata
        val_od_idx=val_od_idx,  # Pass for metadata
        **filtered_params,
    )

    # When outlier detection is applied, pass targets directly to avoid re-subsetting
    if outlier_detector is not None:
        od_common_params["train_target"] = current_train_target
        if current_val_target is not None:
            od_common_params["val_target"] = current_val_target

    return od_common_params, current_behavior_config




def _build_pre_pipelines(
    use_ordinary_models: bool,
    rfecv_models: List[str],
    rfecv_models_params: Dict[str, Any],
    use_mrmr_fs: bool,
    mrmr_kwargs: Dict[str, Any],
    custom_pre_pipelines: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Any], List[str]]:
    """
    Build lists of pre-pipelines and their names for feature selection.

    Args:
        use_ordinary_models: Whether to include no-pipeline (ordinary) models
        rfecv_models: List of RFECV model names to use
        rfecv_models_params: Dict mapping RFECV model names to their pipeline configurations
        use_mrmr_fs: Whether to include MRMR feature selection
        mrmr_kwargs: Keyword arguments for MRMR
        custom_pre_pipelines: Dict mapping pipeline names to sklearn transformers.
            Each transformer must implement fit() and transform() methods.
            Example: {"pca50": IncrementalPCA(n_components=50)}

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

    # Add custom pre-pipelines
    if custom_pre_pipelines:
        for pipeline_name, pipeline_obj in custom_pre_pipelines.items():
            pre_pipelines.append(pipeline_obj)
            pre_pipeline_names.append(f"{pipeline_name} ")

    return pre_pipelines, pre_pipeline_names




def _build_process_model_kwargs(
    model_file: str,
    model_name_with_weight: str,
    model_file_name:str,
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
    optimize_storage: bool = True,
    metadata_columns: Optional[List[str]] = None,
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
        "model_file_name": model_file_name,
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
        "optimize_storage": optimize_storage,
        "metadata_columns": metadata_columns,
    }

    # Skip preprocessing (scaler/imputer/encoder) if Polars-ds pipeline was already applied globally
    # but still run feature selectors (MRMR, RFECV) if present
    if polars_pipeline_applied:
        kwargs["skip_preprocessing"] = True

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
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Convert DataFrames to pandas format (zero-copy for Polars).

    Despite the "zero-copy" label, the conversion has real per-column cost
    when the source Polars DataFrame holds ``pl.Categorical`` columns: the
    pyarrow round-trip rebuilds each dict with int32 indices (polars' default
    uint32 indices aren't supported by ``to_pandas()``), and for
    high-cardinality categoricals that's the slow step. On a 1M x 98 frame
    with ~13 categoricals (a few of them text-like with 10k+ unique values)
    this step has been observed to take 5+ minutes with no intermediate
    logging. Per-split timers are logged here so the next time the step
    drags it is obvious which split is slow.

    Args:
        train_df: Training DataFrame (pandas or polars).
        val_df: Validation DataFrame (pandas or polars) or None.
        test_df: Test DataFrame (pandas or polars) or None.
        verbose: If truthy, log per-split conversion timing and total elapsed.

    Returns:
        Tuple of (train_df_pd, val_df_pd, test_df_pd)

    Raises:
        TypeError: If any DataFrame is not pandas, polars, or None.
    """
    # Validate input types
    for name, df in [("train_df", train_df), ("val_df", val_df), ("test_df", test_df)]:
        if df is not None and not isinstance(df, (pd.DataFrame, pl.DataFrame)):
            raise TypeError(f"{name} must be pandas DataFrame, polars DataFrame, or None, got {type(df).__name__}")

    def _convert_one(df, name):
        if df is None or isinstance(df, pd.DataFrame):
            return df
        t0 = timer()
        out = get_pandas_view_of_polars_df(df)
        if verbose:
            logger.info(
                "  polars->pandas(%s) %dx%d in %.1fs",
                name, df.shape[0], df.shape[1], timer() - t0,
            )
        return out

    t0_total = timer()
    train_df_pd = _convert_one(train_df, "train")
    val_df_pd = _convert_one(val_df, "val")
    test_df_pd = _convert_one(test_df, "test")
    if verbose:
        logger.info("  polars->pandas total: %.1fs", timer() - t0_total)

    return train_df_pd, val_df_pd, test_df_pd




def _get_pipeline_components(
    preprocessing_config: PreprocessingConfig,
    cat_features: List[str],
) -> Tuple[Optional[Any], SimpleImputer, StandardScaler]:
    """
    Get pipeline components (category_encoder, imputer, scaler) from typed config or defaults.

    Reads from ``preprocessing_config.{category_encoder, imputer, scaler}`` -
    these three fields absorbed the only transformer overrides that had no
    typed home before the 2026-04-27 refactor; everything else migrated to a
    sibling typed config.

    Args:
        preprocessing_config: Typed PreprocessingConfig. ``None`` defaults on its
            transformer fields trigger the context-aware default selection below
            (CatBoostEncoder when cat features exist, SimpleImputer always,
            StandardScaler always).
        cat_features: List of categorical feature names.

    Returns:
        Tuple containing:
            - category_encoder: Encoder for categorical features (e.g., CatBoostEncoder),
              or None if no categorical features exist.
            - imputer: SimpleImputer instance for handling missing values.
            - scaler: StandardScaler instance for feature normalization.
    """
    category_encoder = preprocessing_config.category_encoder
    imputer = preprocessing_config.imputer
    scaler = preprocessing_config.scaler

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
    behavior_config: "TrainingBehaviorConfig",
) -> Tuple[Optional[Dict], List[str]]:
    """
    Compute fairness subgroups from DataFrame if fairness_features are specified.

    Args:
        df: Full DataFrame (before splitting).
        behavior_config: Training behavior configuration.

    Returns:
        Tuple of (fairness subgroups dict or None, list of fairness feature names).
    """
    fairness_features = behavior_config.fairness_features or []
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
        cont_nbins=behavior_config.cont_nbins,
        min_pop_cat_thresh=behavior_config.fairness_min_pop_cat_thresh,
    )
    return subgroups, fairness_features




def _should_skip_catboost_metamodel(
    model_or_pipeline_name: str,
    target_type: TargetTypes,
    behavior_config: "TrainingBehaviorConfig",
) -> bool:
    """
    Check if CatBoost model should be skipped due to metamodel_func incompatibility.

    CatBoost regression with metamodel_func causes sklearn clone issues:
    RuntimeError: Cannot clone object <catboost.core.CatBoostRegressor...>,
    as the constructor either does not set or modifies parameter custom_metric.

    Args:
        model_or_pipeline_name: Model name or pre-pipeline name to check.
        target_type: Type of target (regression or classification).
        behavior_config: Training behavior configuration.

    Returns:
        True if this combination should be skipped.
    """
    if target_type != TargetTypes.REGRESSION:
        return False
    if behavior_config.metamodel_func is None:
        return False
    # Check if name contains 'cb' (for model names like 'cb') or 'cb_rfecv' (for pipelines)
    return model_or_pipeline_name in ("cb", "cb_rfecv")




def _create_initial_metadata(
    model_name: str,
    target_name: str,
    mlframe_models: List[str],
    preprocessing_config: PreprocessingConfig,
    pipeline_config: PreprocessingBackendConfig,
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
    def _as_dict(cfg):
        if cfg is None or isinstance(cfg, dict):
            return cfg
        if hasattr(cfg, "model_dump"):
            return cfg.model_dump()
        return cfg

    return {
        "model_name": model_name,
        "target_name": target_name,
        "mlframe_models": mlframe_models,
        "configs": {
            "preprocessing": _as_dict(preprocessing_config),
            "pipeline": _as_dict(pipeline_config),
            "split": _as_dict(split_config),
        },
    }




def _initialize_training_defaults(
    common_params_dict: Optional[Dict[str, Any]],
    rfecv_models: Optional[List[str]],
    mrmr_kwargs: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
    """
    Initialize default values for training parameters.

    Args:
        common_params_dict: Internal common-params dict (can be None).
        rfecv_models: List of RFECV models (can be None).
        mrmr_kwargs: MRMR keyword arguments (can be None).

    Returns:
        Tuple of initialized values:
        - common_params_dict: Dict (never None)
        - rfecv_models: List (never None)
        - mrmr_kwargs: Dict (never None)
    """
    if common_params_dict is None:
        common_params_dict = {}

    if rfecv_models is None:
        rfecv_models = []

    if mrmr_kwargs is None:
        mrmr_kwargs = dict(
            n_workers=max(1, psutil.cpu_count(logical=False)),
            verbose=2,
            fe_max_steps=1,
        )

    return (
        common_params_dict,
        rfecv_models,
        mrmr_kwargs,
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
    slug_to_original_target_type: Optional[Dict[str, str]] = None,
    slug_to_original_target_name: Optional[Dict[str, str]] = None,
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
        slug_to_original_target_type: Mapping from slugified target_type to original.
        slug_to_original_target_name: Mapping from slugified cur_target_name to original.
    """
    # Add shared objects to metadata
    metadata.update(
        {
            "outlier_detector": outlier_detector,
            "outlier_detection_result": outlier_detection_result,
            "trainset_features_stats": trainset_features_stats,
        }
    )

    # Add slug-to-original name mappings for load_mlframe_suite
    if slug_to_original_target_type:
        metadata["slug_to_original_target_type"] = slug_to_original_target_type
    if slug_to_original_target_name:
        metadata["slug_to_original_target_name"] = slug_to_original_target_name

    # Save metadata.
    # Atomic write: serialize -> temp file in same dir -> os.replace.
    # Prevents metadata.* corruption when two train runs race on the
    # same target (2026-04-19 probe finding). Load path then sees
    # either the complete old file or the complete new one, never a
    # partial write that surfaces as an opaque UnpicklingError.
    #
    # 2026-04-29: write ``metadata.pkl.zst`` (pickle protocol=5 + zstd
    # L3) instead of ``metadata.joblib``. Benchmarked on synthetic
    # mlframe metadata (5k x 50, 50k x 200 + 20 large numpy arrays):
    # pickle protocol=5 is 13-47x faster to write AND read than
    # ``joblib.dump`` and matches its numerical output bit-for-bit;
    # zstd L3 cuts the file 4.2x smaller than the uncompressed pickle
    # while still beating ``joblib.dump compress=3`` by 8-13x on
    # writes. Reader at ``train_mlframe_models_suite`` / load path
    # tries the new file first and falls back to the legacy
    # ``metadata.joblib`` so saves from older mlframe versions keep
    # loading without manual migration.
    if data_dir and models_dir:
        metadata_dir = join(data_dir, models_dir, slugify(target_name), slugify(model_name))
        metadata_file = join(metadata_dir, "metadata.pkl.zst")
        try:
            from mlframe.training.io import atomic_write_bytes
            import pickle as _pickle
            try:
                import zstandard as _zstd
                _cctx = _zstd.ZstdCompressor(level=3)
                def _writer(f):
                    f.write(_cctx.compress(_pickle.dumps(metadata, protocol=5)))
            except ImportError:
                # No zstd available - write uncompressed pickle (still
                # faster than joblib by 13x). Filename keeps ``.pkl``
                # so the reader's magic-byte sniff can route it.
                metadata_file = join(metadata_dir, "metadata.pkl")
                def _writer(f):
                    _pickle.dump(metadata, f, protocol=5)
            atomic_write_bytes(metadata_file, _writer)
            if verbose:
                logger.info("Saved metadata to %s", metadata_file)
        except (OSError, IOError) as e:
            logger.error(f"Failed to save metadata to {metadata_file}: {e}")
            raise


