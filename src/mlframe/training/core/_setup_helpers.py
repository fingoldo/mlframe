"""Suite setup and configuration helpers."""

from __future__ import annotations

import logging
import os
from os.path import join
from timeit import default_timer as timer
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, TypeVar, Union

if TYPE_CHECKING:
    from ..configs import (
        PreprocessingBackendConfig,
        PreprocessingConfig,
        TrainingBehaviorConfig,
        TrainingSplitConfig,
    )
    from ._training_context import TrainingContext

import numpy as np
import pandas as pd
import psutil

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import category_encoders as ce

from mlframe.feature_selection.filters import MRMR

from ..configs import TargetTypes
from .._ram_helpers import maybe_clean_ram_and_gpu
from ..utils import get_pandas_view_of_polars_df, log_ram_usage
from pyutilz.strings import slugify
from pyutilz.system import ensure_dir_exists

logger = logging.getLogger(__name__)


from mlframe.metrics.core import create_fairness_subgroups

DEFAULT_PROBABILITY_THRESHOLD = 0.5

ConfigT = TypeVar("ConfigT")


def _ensure_config(
    config: ConfigT | dict[str, Any] | None,
    config_class: type,
    kwargs: dict[str, Any],
) -> ConfigT:
    """Convert dict/None to Pydantic config object."""
    if isinstance(config, dict):
        return config_class(**config)
    elif config is None:
        return config_class(**{k: v for k, v in kwargs.items() if k in config_class.model_fields})
    return config


def _apply_outlier_detection_global(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    train_idx: np.ndarray,
    val_idx: np.ndarray | None,
    outlier_detector: Any,
    od_val_set: bool,
    verbose: bool,
    baseline_rss_mb: float = 0.0,
    df_size_mb: float = 0.0,
    targets_for_classbalance: dict[str, Any] | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame | None,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Apply outlier detection ONCE globally (unsupervised - no target needed)."""
    if outlier_detector is None:
        return train_df, val_df, train_idx, val_idx, None, None

    if verbose:
        logger.info("Fitting outlier detector (once for all targets)...")

    # sklearn outlier detectors coerce input via check_array; non-numeric columns
    # (string/categorical/embedding-list) crash fit. Drop non-numeric on each call
    # so polars and pandas paths stay symmetric.
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
    outlier_detector.fit(_train_numeric)

    is_inlier = outlier_detector.predict(_train_numeric)
    train_od_idx = is_inlier == 1

    filtered_train_df = train_df
    filtered_train_idx = train_idx

    def _filter_df_by_mask(_df, mask):
        if isinstance(_df, pl.DataFrame):
            return _df.filter(pl.Series(mask))
        return _df.loc[mask]

    train_kept = train_od_idx.sum()
    if train_kept < len(train_df):
        # When OD is fit on features that include a label-correlated leak feature, the unsupervised
        # detector can flag the rare class as outliers and remove it entirely, leaving train with
        # one unique target and crashing CB/XGB deep in C++. Skip the OD filter when this would
        # happen; fit stays intact for diagnostic logging via train_od_idx.
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
            # All-True mask so the downstream polars-fastpath filter is a no-op.
            train_kept = len(train_df)
            train_od_idx = np.ones(len(train_df), dtype=bool)

    # Fail fast on catastrophic misconfiguration (~all samples flagged); otherwise CatBoost/LightGBM
    # crashes 5+ min later with opaque "X is empty" errors.
    min_keep = max(1, int(len(train_df) * 0.01))
    if train_kept < min_keep:
        raise ValueError(
            f"Outlier detector rejected {len(train_df) - train_kept:_} of {len(train_df):_} "
            f"train samples, leaving only {train_kept:_} rows (< {min_keep:_}, 1% of input). "
            f"The detector is likely misconfigured (e.g. contamination too high, trained on "
            f"unrepresentative data, or a sign convention bug). Training cannot proceed."
        )

    filtered_val_df = val_df
    filtered_val_idx = val_idx
    val_od_idx = None

    if val_df is not None and od_val_set:
        is_inlier = outlier_detector.predict(_numeric_only_view(val_df))
        val_od_idx = is_inlier == 1
        val_kept = val_od_idx.sum()
        # Mirror of train-side class-balance pre-check: skip OD on val if it would wipe out a class.
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
                        # All-True mask so downstream polars filter is a no-op.
                        val_kept = len(val_df)
                        val_od_idx = np.ones(len(val_df), dtype=bool)
                        break
                except Exception as _exc:
                    logger.debug("Class-balance pre-check on val failed for target %s: %s", _tn, _exc)
        # Symmetric of train-side min_keep guard: don't propagate a near-empty val_df; downstream
        # eval paths handle empty val poorly. Keep original val_df and surface the error.
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
    data_dir: str | None,
    models_dir: str | None,
    save_charts: bool = True,
) -> tuple[str | None, str | None]:
    """Set up directories for model artifacts and charts."""
    parts = slugify(target_name), slugify(model_name), slugify(target_type.lower()), slugify(cur_target_name)

    # Falsy check (not `is not None`): empty string data_dir="" means "no persistence", same as None.
    # Truthy "" would leak ./charts / ./models into CWD and re-load stale pickles on later runs.
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
    common_params_dict: dict[str, Any],
    trainset_features_stats: dict | None,
    plot_file: str | None,
    train_od_idx: np.ndarray | None,
    val_od_idx: np.ndarray | None,
    current_train_target: Any | None,
    current_val_target: Any | None,
    outlier_detector: Any | None,
    behavior_config: TrainingBehaviorConfig,
    fairness_subgroups: dict | None,
) -> tuple[dict[str, Any], TrainingBehaviorConfig]:
    """Build common_params and behavior_config for select_target call."""
    if fairness_subgroups is not None:
        current_behavior_config = behavior_config.model_copy(
            update={"_precomputed_fairness_subgroups": fairness_subgroups}
        )
    else:
        current_behavior_config = behavior_config

    # Drop train_target/val_target so they don't conflict when OD applies.
    filtered_params = {k: v for k, v in common_params_dict.items() if k not in ("train_target", "val_target")}
    od_common_params = dict(
        trainset_features_stats=trainset_features_stats,
        plot_file=plot_file,
        train_od_idx=train_od_idx,
        val_od_idx=val_od_idx,
        **filtered_params,
    )

    # With OD applied, pass targets directly to avoid re-subsetting.
    if outlier_detector is not None:
        od_common_params["train_target"] = current_train_target
        if current_val_target is not None:
            od_common_params["val_target"] = current_val_target

    return od_common_params, current_behavior_config


def _build_pre_pipelines(
    use_ordinary_models: bool,
    rfecv_models: list[str],
    rfecv_models_params: dict[str, Any],
    use_mrmr_fs: bool,
    mrmr_kwargs: dict[str, Any],
    custom_pre_pipelines: dict[str, Any] | None = None,
) -> tuple[list[Any], list[str]]:
    """Build lists of pre-pipelines and their names for feature selection."""
    pre_pipelines = []
    pre_pipeline_names = []

    if use_ordinary_models:
        pre_pipelines.append(None)
        pre_pipeline_names.append("")

    if not rfecv_models:
        rfecv_models = []
    if not rfecv_models_params:
        rfecv_models_params = {}
    unknown_rfecv_models = [m for m in rfecv_models if m not in rfecv_models_params]
    if unknown_rfecv_models:
        raise ValueError(f"Unknown RFECV model(s): {unknown_rfecv_models}. " f"Available: {list(rfecv_models_params.keys())}")
    for rfecv_model_name in rfecv_models:
        pre_pipelines.append(rfecv_models_params[rfecv_model_name])
        pre_pipeline_names.append(f"{rfecv_model_name} ")

    if use_mrmr_fs:
        # MRMR handles NaN natively via ``nan_strategy`` (default
        # "separate_bin" routes NaN rows to a dedicated discretization bin
        # instead of imputing them; see MRMR._validate_inputs). Wrapping
        # in SimpleImputer would discard that signal and silently degrade
        # downstream NaN-aware backends (catboost / lgb / xgb).
        pre_pipelines.append(MRMR(**mrmr_kwargs))
        pre_pipeline_names.append("MRMR ")

    if custom_pre_pipelines:
        # Clone every user-supplied pre-pipeline before insertion so fit-time
        # state from one model never leaks across the others in this suite.
        # sklearn.base.clone is the canonical path; non-BaseEstimator objects
        # fall back to copy.deepcopy so callers can pass custom transformers
        # that don't implement the sklearn estimator protocol.
        import copy as _copy
        try:
            from sklearn.base import clone as _sk_clone
        except Exception:
            _sk_clone = None
        for pipeline_name, pipeline_obj in custom_pre_pipelines.items():
            try:
                _cloned = _sk_clone(pipeline_obj) if _sk_clone is not None else _copy.deepcopy(pipeline_obj)
            except Exception:
                _cloned = _copy.deepcopy(pipeline_obj)
            pre_pipelines.append(_cloned)
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
    models: dict,
    model_params: dict[str, Any],
    common_params: dict[str, Any],
    ens_models: list | None,
    trainset_features_stats: dict | None,
    verbose: int,
    cached_dfs: tuple | None,
    polars_pipeline_applied: bool = False,
    mlframe_model_name: str | None = None,
    optimize_storage: bool = True,
    metadata_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Build kwargs dictionary for process_model call."""
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

    # Skip scaler/imputer/encoder if Polars-ds pipeline already ran globally; selectors still run.
    if polars_pipeline_applied:
        kwargs["skip_preprocessing"] = True

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
    train_df: pd.DataFrame | pl.DataFrame,
    val_df: pd.DataFrame | pl.DataFrame | None,
    test_df: pd.DataFrame | pl.DataFrame | None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Convert DataFrames to pandas format (nominally zero-copy for Polars).

    Per-split timers exist because Polars pl.Categorical columns force a pyarrow round-trip that
    rebuilds each dict with int32 indices (uint32 isn't supported by to_pandas); on
    high-cardinality categoricals this step can take 5+ minutes per split.
    """
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
    cat_features: list[str],
    random_seed: int | None = None,
) -> tuple[Any | None, SimpleImputer, StandardScaler]:
    """Get pipeline components (category_encoder, imputer, scaler) from typed config or defaults.

    ``random_seed`` is used to seed the default ``CatBoostEncoder`` so two calls
    with the same seed produce deterministic encodings. CatBoostEncoder draws
    permutations internally; without an explicit ``random_state`` the encoder
    re-shuffles on every fit, breaking determinism across reruns (fix audit
    row FE-P2-5).
    """
    category_encoder = preprocessing_config.category_encoder
    imputer = preprocessing_config.imputer
    scaler = preprocessing_config.scaler

    if category_encoder is None and cat_features:
        _seed = int(random_seed) if random_seed is not None else 42
        category_encoder = ce.CatBoostEncoder(random_state=_seed)

    if imputer is None:
        imputer = SimpleImputer()

    if scaler is None:
        scaler = StandardScaler()

    return category_encoder, imputer, scaler


def _compute_fairness_subgroups(
    df: pd.DataFrame | pl.DataFrame,
    behavior_config: TrainingBehaviorConfig,
) -> tuple[dict | None, list[str]]:
    """Compute fairness subgroups from DataFrame if fairness_features are specified."""
    fairness_features = behavior_config.fairness_features or []
    if not fairness_features:
        return None, fairness_features

    # Select only required columns - memory matters on large frames.
    cols_to_select = [f for f in fairness_features if f not in ("**ORDER**", "**RANDOM**") and f in df.columns]

    if cols_to_select:
        if isinstance(df, pl.DataFrame):
            # Arrow-backed split-blocks bridge: ~32x faster than .to_pandas() default on
            # 9M-row frames -- consolidation copy eliminated for numeric / bool columns.
            df_subset = get_pandas_view_of_polars_df(df.select(cols_to_select))
        else:
            df_subset = df[cols_to_select]
    else:
        # Only **ORDER**/**RANDOM** markers - no actual columns needed.
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
    behavior_config: TrainingBehaviorConfig,
) -> bool:
    """Skip CatBoost regression + metamodel_func: sklearn clone fails on CatBoostRegressor
    (RuntimeError: constructor does not set or modifies parameter custom_metric)."""
    if target_type != TargetTypes.REGRESSION:
        return False
    if behavior_config.metamodel_func is None:
        return False
    return model_or_pipeline_name in ("cb", "cb_rfecv")


def _create_initial_metadata(
    model_name: str,
    target_name: str,
    mlframe_models: list[str],
    preprocessing_config: PreprocessingConfig,
    pipeline_config: PreprocessingBackendConfig,
    split_config: TrainingSplitConfig,
) -> dict[str, Any]:
    """Create the initial metadata dictionary for tracking training."""
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
    common_params_dict: dict[str, Any] | None,
    rfecv_models: list[str] | None,
    mrmr_kwargs: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Initialize default values for training parameters."""
    if common_params_dict is None:
        common_params_dict = {}

    if rfecv_models is None:
        rfecv_models = []

    if mrmr_kwargs is None:
        mrmr_kwargs = dict(
            n_workers=max(1, psutil.cpu_count(logical=False)),
            verbose=2,
            fe_max_steps=1,
            max_runtime_mins=300,
        )

    return (
        common_params_dict,
        rfecv_models,
        mrmr_kwargs,
    )


def _finalize_and_save_metadata(ctx: TrainingContext, *, verbose: int | None = None) -> None:
    """Finalize ``ctx.metadata`` (set outlier_detector / OD result / trainset stats / slug maps) and atomically save to disk.

    ``verbose=None`` reads ``ctx.verbose``; explicit ``0`` silences the save-log -- used by ``finalize_suite`` to avoid
    the duplicate "Saved metadata to ..." line when the suite saves once after main.py already saved partway.
    """
    metadata = ctx.metadata
    _verbose = ctx.verbose if verbose is None else verbose
    metadata.update(
        {
            "outlier_detector": ctx.outlier_detector,
            "outlier_detection_result": ctx.outlier_detection_result,
            "trainset_features_stats": ctx.trainset_features_stats,
        }
    )

    if ctx.slug_to_original_target_type:
        metadata["slug_to_original_target_type"] = ctx.slug_to_original_target_type
    if ctx.slug_to_original_target_name:
        metadata["slug_to_original_target_name"] = ctx.slug_to_original_target_name

    # Atomic write (serialize -> temp file -> os.replace) avoids metadata.* corruption when two
    # train runs race on the same target. Reader sees the complete old or new file, never partial.
    if ctx.data_dir and ctx.models_dir:
        metadata_dir = join(ctx.data_dir, ctx.models_dir, slugify(ctx.target_name), slugify(ctx.model_name))
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
                # Fallback: uncompressed pickle. .pkl extension lets the reader's magic-byte sniff route it.
                metadata_file = join(metadata_dir, "metadata.pkl")
                def _writer(f):
                    _pickle.dump(metadata, f, protocol=5)
            atomic_write_bytes(metadata_file, _writer)
            if _verbose:
                logger.info("Saved metadata to %s", metadata_file)
        except OSError as e:
            logger.error(f"Failed to save metadata to {metadata_file}: {e}")
            raise


