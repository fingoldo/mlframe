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

# Mirrors the BorutaShap pattern below -- MRMR transitively pulls in
# the entire mlframe.feature_selection package (numba kernels + filter wrappers
# + sklearn estimators), which adds ~10-25s to first-call import time even when
# the suite doesn't opt into MRMR. Module-level import is deferred to the single
# call site (``_build_pre_pipelines`` in ``_setup_helpers_pre_pipelines``);
# other helpers in this module that only need the class for typing use
# TYPE_CHECKING-guarded references.
if TYPE_CHECKING:
    from mlframe.feature_selection.filters import MRMR  # noqa: F401

from ..configs import TargetTypes
from .._ram_helpers import maybe_clean_ram_and_gpu
from ..utils import get_pandas_view_of_polars_df, log_ram_usage
from pyutilz.strings import slugify
from pyutilz.system import ensure_dir_exists

logger = logging.getLogger(__name__)


# Polars-ds Pipeline JSON-roundtrip cache (in-memory + cross-process file cache).
# Re-exported from the sibling so historical
# ``from mlframe.training.core._setup_helpers import _pipeline_disk_cache_path``
# imports keep resolving.
from ._setup_helpers_pipeline_cache import (  # noqa: E402, F401
    _PIPELINE_JSON_ROUNDTRIP_CACHE,
    _PIPELINE_JSON_DISK_CACHE_PATH,
    _PIPELINE_JSON_DISK_CACHE_LOADED,
    _PIPELINE_JSON_DISK_CACHE_MAX_ENTRIES,
    _pipeline_disk_cache_path,
    _pipeline_disk_cache_version_tag,
    _load_pipeline_disk_cache_into_memory,
    _persist_pipeline_disk_cache,
    _PolarsDsPipelineJsonProxy,
    _polars_ds_pipeline_from_json,
)


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


# Global outlier-detection helper carved to ``_setup_helpers_outliers``;
# re-exported here so ``from mlframe.training.core._setup_helpers import
# _apply_outlier_detection_global`` keeps working.
from ._setup_helpers_outliers import _apply_outlier_detection_global  # noqa: E402, F401


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


# ``_build_pre_pipelines`` lives in ``_setup_helpers_pre_pipelines``; re-exported here.
from ._setup_helpers_pre_pipelines import _build_pre_pipelines  # noqa: E402, F401


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
            # Audit D P1-7 (2026-05-18): the polars->pandas conversion is NEEDED here because
            # ``create_fairness_subgroups`` from ``mlframe.metrics.core`` consumes a pandas
            # frame (pandas groupby / nunique). The conversion cannot be pushed further. Keep
            # the split-blocks bridge so the hop stays at zero-copy on numeric columns.
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


# Metadata builders / finalizers live in ``_setup_helpers_metadata``;
# re-exported here so historical
# ``from mlframe.training.core._setup_helpers import _finalize_and_save_metadata``
# imports keep resolving.
from ._setup_helpers_metadata import (  # noqa: E402, F401
    _create_initial_metadata,
    _initialize_training_defaults,
    _finalize_and_save_metadata,
)
