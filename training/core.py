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
import sys
from timeit import default_timer as timer

logger = logging.getLogger(__name__)


def _ensure_logging_visible(level: int = logging.INFO) -> None:
    """Make mlframe progress logs visible — with timestamps — in Jupyter and
    plain scripts.

    Two cases:
      1. No root handlers at all (bare Python / fresh Jupyter kernel before
         anyone called `logging.basicConfig`). → install a stdout handler
         with ``%(asctime)s %(levelname)s %(name)s: %(message)s`` format.
      2. Root already has handlers but their formatter lacks a timestamp
         (classic Jupyter/IPython default emits just ``LEVEL:name:message``,
         which is useless for profiling long training runs). → replace
         those formatters in place with the timestamped one. Handlers that
         already format with ``%(asctime)s`` are left untouched so a user
         who intentionally configured a custom format isn't clobbered.
    """
    root = logging.getLogger()
    desired_fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    desired_datefmt = "%H:%M:%S"
    timestamped = logging.Formatter(desired_fmt, datefmt=desired_datefmt)

    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(timestamped)
        root.addHandler(handler)
    else:
        for h in root.handlers:
            existing = getattr(h.formatter, "_fmt", None) if h.formatter else None
            if not existing or "%(asctime)" not in existing:
                h.setFormatter(timestamped)
    if root.level > level or root.level == logging.NOTSET:
        root.setLevel(level)


def _drop_cols_df(df, cols):
    """Drop ``cols`` from ``df`` (pandas or Polars), ignoring missing names.

    Centralizes the 4-line `isinstance(df, pd.DataFrame)` branch that appeared in
    both `predict_mlframe_models_suite` and `predict_from_models`.
    """
    import pandas as _pd  # local import to avoid top-level cost during helper init
    if not cols:
        return df
    existing = filter_existing(df, cols)
    if not existing:
        return df
    if isinstance(df, _pd.DataFrame):
        return df.drop(columns=existing, errors="ignore")
    return df.drop(existing)  # Polars


def _validate_trusted_path(path: str, trusted_root):
    """Raise ValueError if ``path`` is not inside ``trusted_root``.

    Mirrors the `mlframe.inference.read_trained_models` convention. Gating every
    ``joblib.load`` of a pickled metadata/model file keeps arbitrary-code-execution
    surface limited to explicitly-opted-in directories.
    """
    import os as _os
    if trusted_root is None:
        raise ValueError(
            "trusted_root is required for joblib.load() of metadata files. Pass an "
            "absolute directory under which the metadata artifact is stored."
        )
    abs_root = _os.path.abspath(trusted_root)
    abs_path = _os.path.abspath(path)
    try:
        common = _os.path.commonpath([abs_root, abs_path])
    except ValueError:
        raise ValueError(f"Path {abs_path} is not inside trusted_root {abs_root}")
    if common != abs_root:
        raise ValueError(f"Path {abs_path} is not inside trusted_root {abs_root}")


def _df_shape_str(df) -> str:
    """Format DataFrame shape as 'rows×cols' with thousands separators."""
    if df is None:
        return "None"
    nrows = df.shape[0] if hasattr(df, "shape") else len(df)
    ncols = df.shape[1] if hasattr(df, "shape") else 0
    return f"{nrows:_}×{ncols}"


def _elapsed_str(start: float) -> str:
    """Format elapsed time since start as human-readable string."""
    elapsed = timer() - start
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    return f"{elapsed / 60:.1f}min"


def _validate_input_columns_against_metadata(
    df,
    metadata: "Dict[str, Any]",
    verbose: bool = False,
):
    """Validate inference-time DataFrame columns against model metadata.

    Before this helper (inline in ``predict_mlframe_models_suite`` /
    ``predict_from_models`` up to 2026-04-19), the logic was:
      - WARN on missing columns, then proceed
      - Drop extra columns if any

    Problem: if a missing column was a load-bearing ``cat_features`` /
    ``text_features`` / ``embedding_features`` member, the pipeline
    transform + model predict ran on a shape-mismatched frame and
    either (a) crashed deep inside sklearn with ``X has N features,
    expected M``, or (b) produced garbage predictions. The WARN alone
    was not actionable.

    Now: columns are partitioned by severity:
      - Missing load-bearing features (cat/text/embedding): **raise
        ValueError** with a diagnostic naming them. These cannot be
        safely dropped — the pipeline was fitted with them.
      - Other missing columns: WARN + proceed. Some callers drop
        derived columns that the pipeline reconstructs; that's OK.
      - Extra columns: dropped silently (or logged at verbose=True).

    Returns the df (possibly with extra columns filtered out).
    """
    columns = metadata.get("columns", [])
    if not columns:
        return df

    missing_cols = set(columns) - set(df.columns)
    extra_cols = set(df.columns) - set(columns)

    if missing_cols:
        meta_cat = set(metadata.get("cat_features") or [])
        meta_text = set(metadata.get("text_features") or [])
        meta_emb = set(metadata.get("embedding_features") or [])
        critical_missing = missing_cols & (meta_cat | meta_text | meta_emb)
        if critical_missing:
            raise ValueError(
                f"Input DataFrame is missing {len(critical_missing)} "
                f"load-bearing feature column(s) that the model was "
                f"trained on: {sorted(critical_missing)}. These are "
                f"declared in metadata as cat/text/embedding features; "
                f"the pipeline + model cannot run correctly without "
                f"them. Either restore the upstream extraction that "
                f"produced these columns, or retrain the model on the "
                f"current feature set."
            )
        logger.warning(
            "Missing columns in input: %s. The pipeline will attempt "
            "to proceed — downstream errors about shape mismatches "
            "usually trace back here.",
            sorted(missing_cols),
        )

    if extra_cols:
        if verbose:
            logger.info(f"Dropping extra columns: {sorted(extra_cols)}")
        df = df[filter_existing(df, columns)]

    return df


def _filter_polars_cat_features_by_dtype(
    df: "pl.DataFrame",
    cat_features: "List[str]",
) -> "List[str]":
    """Defensive filter for CB Polars fastpath ``cat_features``.

    CatBoost 1.2.x's ``_set_features_order_data_polars_categorical_column``
    is a Cython fused cpdef with dispatch **only** for ``pl.Categorical``
    (and on some builds ``pl.Enum``). If a caller hands a column to CB
    via ``cat_features`` but the column's dtype in the DataFrame is
    ``pl.String``/``pl.Utf8``/numeric/etc, the dispatcher falls through
    to the opaque ``TypeError: No matching signature found`` — with no
    hint about which column or why.

    Production incident 2026-04-19: the orchestration in
    ``train_mlframe_models_suite`` short-circuited to a *stale* pre-promotion
    cat_features list that still contained 4 columns which had been
    cast ``pl.Categorical → pl.String`` for the text-features fastpath.
    CB saw ``cat_features=['category', 'skills_text', ...]`` with those
    columns being ``pl.String`` and raised "No matching signature found",
    burning 22 s + a 150 s pandas fallback on every run.

    This helper runs **right before** passing cat_features to
    ``model.fit()``:
      - keeps columns whose dtype is ``pl.Categorical`` or ``pl.Enum``
      - drops columns with any other dtype and logs a WARNING naming
        them and their observed dtype
      - silently drops columns missing from the DataFrame (defensive;
        a missing column would crash CB with a different error anyway)

    Returns the filtered list; empty list if nothing valid remains.
    """
    valid: list = []
    dropped: list = []  # list of (col_name, dtype_str)
    for c in cat_features or []:
        if c not in df.columns:
            continue
        dt = df.schema[c]
        is_cat = (dt == pl.Categorical) or (
            hasattr(pl, "Enum") and isinstance(dt, pl.Enum)
        )
        if is_cat:
            valid.append(c)
        else:
            dropped.append((c, str(dt)))
    if dropped:
        logger.warning(
            "Dropping %d column(s) from CB cat_features because their "
            "Polars dtype is not Categorical/Enum: %s. CatBoost's fastpath "
            "dispatcher has no overload for those types and would raise "
            "'No matching signature found'. Most likely cause: the column "
            "was promoted from cat_features to text_features and cast to "
            "pl.String, but the caller is still passing the pre-promotion "
            "list. Fix the caller to use the post-promotion cat_features.",
            len(dropped), dropped,
        )
    return valid


def _auto_detect_feature_types(
    df,
    feature_types_config,
    cat_features: list,
    verbose: bool = False,
) -> tuple:
    """Auto-detect text and embedding features from DataFrame schema and cardinality.

    Also *promotes* columns that were initially classified as categorical by the
    Polars schema (e.g. columns ending in ``_text`` that are ``pl.Categorical``
    in raw data) up to ``text_features`` when their cardinality exceeds the
    configured threshold. Previously such columns stayed in ``cat_features``
    and CatBoost wasted GB of memory on nominal encoding of what are really
    free-text blobs. The promotion happens iff:

        * user did not explicitly list the column in ``text_features``
          or ``embedding_features`` already, AND
        * the column's dtype is ``pl.String``/``pl.Utf8``/``pl.Categorical``
          (ordered categoricals like ``pl.Enum`` stay nominal), AND
        * ``n_unique > cat_text_cardinality_threshold``.

    Promoted columns are returned in ``text_features``; the caller is
    responsible for filtering its own ``cat_features`` against the returned
    ``text_features`` (see ``effective_cat_features`` construction in
    ``train_mlframe_models_suite``). This function does NOT mutate the
    ``cat_features`` argument — prior versions did, which created a latent
    repeat-call state-leak trap whenever a caller reused the same list.

    Args:
        df: Training DataFrame (Polars or pandas).
        feature_types_config: FeatureTypesConfig with user overrides and threshold.
        cat_features: Already-detected categorical features (from pipeline).
            Read-only. Used only to decide which auto-detected text columns
            were "promoted" from cat_features vs newly discovered.
        verbose: Whether to log detections.

    Returns:
        (text_features, embedding_features) — lists of column names.
    """
    import polars as pl

    text_features = list(feature_types_config.text_features or [])
    embedding_features = list(feature_types_config.embedding_features or [])

    if not feature_types_config.auto_detect_feature_types:
        return text_features, embedding_features

    # Defensive: callers sometimes pass ``cat_features=None`` (e.g. after a
    # model skipped categorical detection). Treat as empty list — the
    # ``if name in cat_features`` checks below would otherwise crash with
    # ``TypeError: argument of type 'NoneType' is not iterable``.
    if cat_features is None:
        cat_features = []

    threshold = feature_types_config.cat_text_cardinality_threshold
    user_assigned = set(text_features) | set(embedding_features)
    promoted: list = []  # cat_features -> text_features, tracked for diagnostic log only
    cardinalities: dict = {}  # per auto-detected text col: n_unique (for diagnostic log)

    if isinstance(df, pl.DataFrame):
        for name, dtype in df.schema.items():
            if name in user_assigned:
                continue
            # Embedding: pl.List(pl.Float32/Float64)
            if dtype == pl.List(pl.Float32) or dtype == pl.List(pl.Float64):
                if name not in cat_features:
                    embedding_features.append(name)
                continue
            # String/Categorical/Enum — evaluate cardinality to split cat vs text.
            # pl.Enum is a fixed-domain categorical; it has an instance-level
            # dtype object (not a class), so it doesn't compare equal to the
            # class-level check above. Use isinstance() for Enum specifically.
            is_text_like = (
                dtype in (pl.String, pl.Utf8, pl.Categorical)
                or isinstance(dtype, pl.Enum)
            )
            if is_text_like:
                n_unique = df[name].n_unique()
                if n_unique > threshold:
                    text_features.append(name)
                    cardinalities[name] = n_unique
                    if name in cat_features:
                        promoted.append(name)
    else:
        # pandas: only detect high-cardinality text (no reliable embedding auto-detect)
        for col in df.columns:
            if col in user_assigned:
                continue
            dtype_name = str(df[col].dtype)
            if dtype_name.startswith("object") or dtype_name.startswith("string") or dtype_name == "category":
                n_unique = df[col].nunique()
                if n_unique > threshold:
                    text_features.append(col)
                    cardinalities[col] = n_unique
                    if col in cat_features:
                        promoted.append(col)

    # Historical note: this function used to mutate ``cat_features`` in place
    # (calling ``.remove(name)`` for each promoted column). The in-place removal
    # was redundant — the actual caller filter lives at the call site, where
    # ``effective_cat_features`` is built via set-difference against
    # ``text_features``. We removed the mutation so repeat calls with a shared
    # list don't corrupt the caller's state.

    def _fmt_with_cardinality(names):
        """'col1:500, col2:12_345' — makes it obvious *why* a column was
        promoted vs the configured threshold."""
        parts = []
        for n in names:
            nu = cardinalities.get(n)
            parts.append(f"{n}:{nu:_}" if nu is not None else n)
        return "[" + ", ".join(parts) + "]"

    if verbose and (text_features or embedding_features or promoted):
        if promoted:
            logger.info(
                f"  Promoted {len(promoted)} high-cardinality column(s) from cat_features to text_features "
                f"(threshold>{threshold}): {_fmt_with_cardinality(promoted)}"
            )
        logger.info(
            f"  Auto-detected feature types — text: {_fmt_with_cardinality(text_features) if text_features else '(none)'}, "
            f"embedding: {embedding_features or '(none)'}"
        )

    return text_features, embedding_features


def _validate_feature_type_exclusivity(
    text_features: list,
    embedding_features: list,
    cat_features: list,
) -> None:
    """Raise ValueError if any column appears in multiple feature type lists.

    Each argument may be ``None`` (treated as empty) — callers that skipped
    one of the feature-type detection stages (e.g. models without
    categorical awareness) pass None rather than ``[]``.
    """
    text_features = text_features or []
    embedding_features = embedding_features or []
    cat_features = cat_features or []
    overlap_tc = set(text_features) & set(cat_features)
    if overlap_tc:
        raise ValueError(f"Columns cannot be both text_features and cat_features: {overlap_tc}")
    overlap_ec = set(embedding_features) & set(cat_features)
    if overlap_ec:
        raise ValueError(f"Columns cannot be both embedding_features and cat_features: {overlap_ec}")
    overlap_te = set(text_features) & set(embedding_features)
    if overlap_te:
        raise ValueError(f"Columns cannot be both text_features and embedding_features: {overlap_te}")


def _build_tier_dfs(
    base_dfs: dict,
    strategy,
    text_features: list,
    embedding_features: list,
    tier_cache: dict,
    verbose: bool = False,
) -> dict:
    """Get or create tier-specific DataFrames with unsupported columns removed.

    Uses .select() instead of .drop() to avoid unnecessary full-DF copies.

    Args:
        base_dfs: Dict with keys train_df, val_df, test_df.
        strategy: ModelPipelineStrategy for the current model.
        text_features: Text feature column names.
        embedding_features: Embedding feature column names.
        tier_cache: Mutable dict caching tier DFs (tier_key -> dict of DFs).
        verbose: Log column dropping.

    Returns:
        Dict with train_df, val_df, test_df (trimmed for tier).
    """
    import polars as pl

    tier = strategy.feature_tier()
    if tier in tier_cache:
        return tier_cache[tier]

    # Determine columns to exclude for this tier
    cols_to_exclude = set()
    if text_features and not strategy.supports_text_features:
        cols_to_exclude.update(text_features)
    if embedding_features and not strategy.supports_embedding_features:
        cols_to_exclude.update(embedding_features)

    if not cols_to_exclude:
        # Tier supports all features — use base DFs directly (no copy)
        tier_dfs = base_dfs
    else:
        if verbose:
            logger.info(f"  Tier {tier}: dropping {len(cols_to_exclude)} text/embedding columns: {sorted(cols_to_exclude)}")
        tier_dfs = {}
        for key in ("train_df", "val_df", "test_df"):
            df_ = base_dfs.get(key)
            if df_ is None:
                tier_dfs[key] = None
                continue
            if isinstance(df_, pl.DataFrame):
                cols_to_keep = [c for c in df_.columns if c not in cols_to_exclude]
                tier_dfs[key] = df_.select(cols_to_keep)
            else:
                cols_to_drop = filter_existing(df_, cols_to_exclude)
                tier_dfs[key] = df_.drop(columns=cols_to_drop) if cols_to_drop else df_

    tier_cache[tier] = tier_dfs
    return tier_dfs


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
from sklearn.base import clone
from sklearn.pipeline import Pipeline
import category_encoders as ce

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from pyutilz.system import ensure_dir_exists

from .configs import (
    PreprocessingConfig,
    TrainingSplitConfig,
    PolarsPipelineConfig,
    FeatureTypesConfig,
    ModelHyperparamsConfig,
    TrainingBehaviorConfig,
    TrainingConfig,
    TargetTypes,
    LinearModelConfig,
    PreprocessingExtensionsConfig,
)
from .preprocessing import (
    load_and_prepare_dataframe,
    preprocess_dataframe,
    save_split_artifacts,
    create_split_dataframes,
)
from .pipeline import fit_and_transform_pipeline, prepare_df_for_catboost, apply_preprocessing_extensions
from mlframe.feature_selection.filters import MRMR
from .utils import (
    log_ram_usage,
    log_phase,
    drop_columns_from_dataframe,
    get_pandas_view_of_polars_df,
    estimate_df_size_mb,
    get_process_rss_mb,
    maybe_clean_ram_and_gpu,
    filter_existing,
)
from .helpers import get_trainset_features_stats_polars, get_trainset_features_stats
from .models import is_linear_model, LINEAR_MODEL_TYPES
from .strategies import get_strategy, get_polars_cat_columns, PipelineCache
from .io import load_mlframe_model
from .splitting import make_train_test_split
from .phases import phase, reset_phase_registry, format_phase_summary

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
    baseline_rss_mb: float = 0.0,
    df_size_mb: float = 0.0,
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

    def _filter_df_by_mask(_df, mask):
        """Boolean-mask filter that works for both pandas and Polars."""
        if isinstance(_df, pl.DataFrame):
            return _df.filter(pl.Series(mask))
        return _df.loc[mask]

    train_kept = train_od_idx.sum()
    if train_kept < len(train_df):
        logger.info(f"Outlier rejection: {len(train_df):_} train samples -> {train_kept:_} kept.")
        filtered_train_df = _filter_df_by_mask(train_df, train_od_idx)
        filtered_train_idx = train_idx[train_od_idx]

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
        is_inlier = outlier_detector.predict(val_df)
        val_od_idx = is_inlier == 1
        val_kept = val_od_idx.sum()
        if val_kept < len(val_df):
            logger.info(f"Outlier rejection: {len(val_df):_} val samples -> {val_kept:_} kept.")
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

    # Falsy check (not `is not None`) — an empty string `data_dir=""` means "no
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
    init_common_params: Dict[str, Any],
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
        init_common_params: Initial common parameters from caller
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
    high-cardinality categoricals that's the slow step. On a 1M × 98 frame
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
                f"  polars→pandas({name}) {df.shape[0]:_}×{df.shape[1]} in {timer() - t0:.1f}s"
            )
        return out

    t0_total = timer()
    train_df_pd = _convert_one(train_df, "train")
    val_df_pd = _convert_one(val_df, "val")
    test_df_pd = _convert_one(test_df, "test")
    if verbose:
        logger.info(f"  polars→pandas total: {timer() - t0_total:.1f}s")

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
    init_common_params: Optional[Dict[str, Any]],
    rfecv_models: Optional[List[str]],
    mrmr_kwargs: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
    """
    Initialize default values for training parameters.

    Args:
        init_common_params: Initial common parameters (can be None).
        rfecv_models: List of RFECV models (can be None).
        mrmr_kwargs: MRMR keyword arguments (can be None).

    Returns:
        Tuple of initialized values:
        - init_common_params: Dict (never None)
        - rfecv_models: List (never None)
        - mrmr_kwargs: Dict (never None)
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

    return (
        init_common_params,
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
    # Atomic write: joblib.dump → temp file in same dir → os.replace.
    # Prevents metadata.joblib corruption when two train runs race on
    # the same target (2026-04-19 probe finding). Load path then sees
    # either the complete old file or the complete new one, never a
    # partial write that surfaces as an opaque UnpicklingError.
    if data_dir and models_dir:
        metadata_file = join(data_dir, models_dir, slugify(target_name), slugify(model_name), "metadata.joblib")
        try:
            from mlframe.training.io import atomic_write_bytes
            atomic_write_bytes(metadata_file, lambda f: joblib.dump(metadata, f))
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
    recurrent_models: Optional[List[str]] = None,
    recurrent_config: Optional[Any] = None,
    sequences: Optional[List[np.ndarray]] = None,
    use_ordinary_models: bool = True,
    use_mlframe_ensembles: bool = True,
    # Configurations (can be dicts or Pydantic objects)
    preprocessing_config: Optional[Union[PreprocessingConfig, Dict]] = None,
    split_config: Optional[Union[TrainingSplitConfig, Dict]] = None,
    pipeline_config: Optional[Union[PolarsPipelineConfig, Dict]] = None,
    preprocessing_extensions: Optional[Union["PreprocessingExtensionsConfig", Dict]] = None,
    feature_types_config: Optional[Union[FeatureTypesConfig, Dict]] = None,
    # Model-specific configurations
    linear_model_config: Optional[LinearModelConfig] = None,
    # Feature selection
    use_mrmr_fs: bool = False,
    mrmr_kwargs: Optional[Dict] = None,
    rfecv_models: Optional[List[str]] = None,
    custom_pre_pipelines: Optional[Dict[str, Any]] = None,
    # Model hyperparameters and training behavior
    hyperparams_config: Optional[Union[ModelHyperparamsConfig, Dict]] = None,
    behavior_config: Optional[Union[TrainingBehaviorConfig, Dict]] = None,
    init_common_params: Optional[Dict] = None,
    # Paths
    data_dir: str = "",
    models_dir: str = "models",
    save_charts: bool = True,
    # Misc
    verbose: int = 1,
    # Outlier detection (run once for all models)
    outlier_detector: Optional[Any] = None,
    od_val_set: bool = True,
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
        recurrent_models: List of recurrent model types to train (lstm, gru, rnn, transformer).
            These models handle sequential data and support variable-length sequences.
        recurrent_config: RecurrentConfig object for recurrent model hyperparameters.
            If None, uses default configuration.
        sequences: Pre-extracted sequences as list of (seq_len, n_features) arrays.
            If None and extractor has sequence_columns configured, sequences will be
            extracted automatically using extractor.get_sequences().
        use_ordinary_models: Whether to train regular models
        use_mlframe_ensembles: Whether to create ensembles

        preprocessing_config: Preprocessing configuration
        split_config: Train/val/test split configuration
        pipeline_config: Pipeline configuration

        use_mrmr_fs: Whether to use MRMR feature selection
        mrmr_kwargs: MRMR parameters
        rfecv_models: Models to use for RFECV
        custom_pre_pipelines: Dict mapping pipeline names to sklearn transformers.
            Each transformer must implement fit() and transform() methods.
            Example: {"pca50": IncrementalPCA(n_components=50)}

        hyperparams_config: Model hyperparameters (iterations, learning rate, per-model kwargs).
            Accepts ModelHyperparamsConfig or dict. Defaults are built in.
        behavior_config: Training behavior flags (GPU preference, calibration, fairness).
            Accepts TrainingBehaviorConfig or dict. Defaults are built in.
        init_common_params: Common initialization parameters (pipeline components, etc.)

        data_dir: Directory for saving artifacts
        models_dir: Directory for saving models

        verbose: Verbosity level (0=silent, 1=info, 2=debug)

    Returns:
        Tuple of (models_dict, metadata_dict)

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

    if verbose:
        _ensure_logging_visible()

    reset_phase_registry()

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
    preprocessing_config = _ensure_config(preprocessing_config, PreprocessingConfig, {})
    pipeline_config = _ensure_config(pipeline_config, PolarsPipelineConfig, {})
    feature_types_config = _ensure_config(feature_types_config, FeatureTypesConfig, {})
    split_config = _ensure_config(split_config, TrainingSplitConfig, {})
    hyperparams_config = _ensure_config(hyperparams_config, ModelHyperparamsConfig, {})
    behavior_config = _ensure_config(behavior_config, TrainingBehaviorConfig, {})

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
    t0_phase1 = timer()
    with phase("load_and_prepare_dataframe"):
        df = load_and_prepare_dataframe(df, preprocessing_config, verbose=verbose)
    if verbose:
        logger.info(f"  load_and_prepare_dataframe done — {_df_shape_str(df)}, {_elapsed_str(t0_phase1)}")

    # Apply features_and_targets_extractor to extract targets
    if verbose:
        logger.info("Create additional features & extracting targets...")

    t0_fte = timer()
    df, target_by_type, group_ids_raw, group_ids, timestamps, artifacts, additional_columns_to_drop, sample_weights = features_and_targets_extractor.transform(
        df
    )
    if verbose:
        logger.info(f"  features_and_targets_extractor done — {_df_shape_str(df)}, {_elapsed_str(t0_fte)}")

    # Capture baseline RSS + DF size NOW — before any downstream steps that may allocate
    # transient state (get_sequences, drop_columns, preprocess). Used by
    # maybe_clean_ram_and_gpu() at later sites to skip ~0.6s gc calls when memory
    # pressure is low. On 100GB production DFs the growth/free-RAM thresholds trip and
    # clean_ram fires; on small test DFs all sites are skipped.
    baseline_rss_mb = get_process_rss_mb()
    df_size_mb = estimate_df_size_mb(df)

    # Extract sequences for recurrent models (if not provided directly)
    if recurrent_models and sequences is None:
        extracted_sequences = features_and_targets_extractor.get_sequences(df)
        if extracted_sequences is not None:
            sequences = extracted_sequences
            if verbose:
                logger.info(f"Extracted {len(sequences)} sequences from DataFrame")
        elif verbose:
            logger.warning("recurrent_models specified but no sequences provided or extracted")

    baseline_rss_mb = maybe_clean_ram_and_gpu(baseline_rss_mb, df_size_mb, verbose=verbose, reason="post-FTE")
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
    t0_preproc = timer()
    df = preprocess_dataframe(df, preprocessing_config, verbose=verbose)
    if verbose:
        logger.info(f"  preprocess_dataframe done — {_df_shape_str(df)}, {_elapsed_str(t0_preproc)}")
        logger.info(f"  PHASE 1 total: {_elapsed_str(t0_phase1)}")

    # ==================================================================================
    # 3. TRAIN/VAL/TEST SPLITTING
    # ==================================================================================

    if verbose:
        log_phase("PHASE 2: Train/Val/Test Splitting")

    t0_phase2 = timer()
    if verbose:
        logger.info(f"Making train_val_test split...")
    with phase("split_data"):
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
    fairness_subgroups, fairness_features = _compute_fairness_subgroups(df, behavior_config)
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
    if verbose:
        logger.info(f"  Split shapes — train: {_df_shape_str(train_df)}, val: {_df_shape_str(val_df)}, test: {_df_shape_str(test_df)}")
        logger.info(f"  PHASE 2 total: {_elapsed_str(t0_phase2)}")

    # Split sequences by train/val/test indices (for recurrent models)
    train_sequences, val_sequences, test_sequences = None, None, None
    if sequences is not None:
        train_sequences = [sequences[i] for i in train_idx]
        val_sequences = [sequences[i] for i in val_idx] if val_idx is not None else None
        test_sequences = [sequences[i] for i in test_idx]
        if verbose:
            logger.info(f"Split sequences: train={len(train_sequences)}, val={len(val_sequences) if val_sequences else 0}, test={len(test_sequences)}")

    # Delete original df to free RAM
    if verbose:
        logger.info("Deleting original DataFrame to free RAM...")

    del df
    # Refresh baseline BEFORE the check: `del df` just freed df_size_mb worth of RAM,
    # so an unrefreshed baseline would yield negative growth and skip cleanup precisely
    # when arena-release would be most effective.
    baseline_rss_mb = get_process_rss_mb()
    baseline_rss_mb = maybe_clean_ram_and_gpu(baseline_rss_mb, df_size_mb, verbose=verbose, reason="post-split (del df)")

    if verbose:
        log_ram_usage()

    # ==================================================================================
    # 4. PIPELINE FITTING & TRANSFORMATION
    # ==================================================================================

    t0_phase3 = timer()
    if verbose:
        log_phase("PHASE 3: Pipeline Fitting & Transformation")

    # Track if input is Polars before pipeline transformation
    was_polars_input = isinstance(train_df, pl.DataFrame)

    # Resolve strategies once for subsequent polars-native gating (avoids redundant lookups).
    _strategies_for_polars_check = [get_strategy(m) for m in mlframe_models] if mlframe_models else []
    all_models_polars_native = bool(_strategies_for_polars_check) and all(
        s.supports_polars for s in _strategies_for_polars_check
    )

    # Auto-skip categorical encoding when all models handle categoricals natively.
    # This avoids wasting time encoding columns that polars-native models don't need,
    # and avoids the .clone() overhead for preserving pre-pipeline originals.
    if was_polars_input and not pipeline_config.skip_categorical_encoding:
        if all_models_polars_native:
            pipeline_config = pipeline_config.model_copy(update={"skip_categorical_encoding": True})
            if verbose:
                logger.info(f"  All models {mlframe_models} support Polars natively — skipping categorical encoding in pipeline")

    # Save pre-pipeline Polars originals for the Polars fastpath.
    # Only clone when the pipeline will actually modify categorical columns;
    # when skip_categorical_encoding=True the pipeline preserves dtypes so the
    # original DF reference is sufficient (B1 optimization — saves 100GB+ clone).
    needs_polars_pre_clone = (
        was_polars_input
        and not pipeline_config.skip_categorical_encoding
        and pipeline_config.categorical_encoding is not None
    )
    if was_polars_input:
        if needs_polars_pre_clone:
            train_df_polars_pre = train_df.clone()
            val_df_polars_pre = val_df.clone() if isinstance(val_df, pl.DataFrame) else None
            test_df_polars_pre = test_df.clone() if isinstance(test_df, pl.DataFrame) else None
            if verbose:
                logger.info(f"  Cloned pre-pipeline Polars originals (pipeline will modify categoricals)")
        else:
            # No clone needed — pipeline won't touch categoricals, reuse references
            train_df_polars_pre = train_df
            val_df_polars_pre = val_df if isinstance(val_df, pl.DataFrame) else None
            test_df_polars_pre = test_df if isinstance(test_df, pl.DataFrame) else None
            if verbose:
                logger.info(f"  Skipped pre-pipeline clone (skip_categorical_encoding=True)")
        cat_features_polars = get_polars_cat_columns(train_df)
    else:
        train_df_polars_pre = None
        val_df_polars_pre = None
        test_df_polars_pre = None
        cat_features_polars = []

    # Pass user-specified text/embedding features to exclude from encoding/scaling.
    # Auto-detection happens later (after pipeline, when cat_features are known).
    t0_fit_pipeline = timer()
    train_df, val_df, test_df, pipeline, cat_features = fit_and_transform_pipeline(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config=pipeline_config,
        ensure_float32=preprocessing_config.ensure_float32_dtypes,
        verbose=verbose,
        text_features=feature_types_config.text_features,
        embedding_features=feature_types_config.embedding_features,
    )
    if verbose:
        logger.info(f"  fit_and_transform_pipeline done in {_elapsed_str(t0_fit_pipeline)}")

    # Track if Polars-ds pipeline was applied (to skip redundant pre_pipeline transforms later)
    polars_pipeline_applied = was_polars_input and pipeline_config.use_polarsds_pipeline and pipeline is not None

    # Apply shared sklearn-based extensions (scaler override / poly / dim-reducer / …).
    # When preprocessing_extensions is None (default) this is a zero-cost noop and the
    # Polars-native fastpath is preserved byte-for-byte.
    if preprocessing_extensions is not None and isinstance(preprocessing_extensions, dict):
        preprocessing_extensions = PreprocessingExtensionsConfig(**preprocessing_extensions)
    t0_ext = timer()
    train_df, val_df, test_df, extensions_pipeline = apply_preprocessing_extensions(
        train_df, val_df, test_df, preprocessing_extensions, verbose=verbose,
    )
    if verbose and preprocessing_extensions is not None:
        logger.info(f"  apply_preprocessing_extensions done in {_elapsed_str(t0_ext)}")
    if extensions_pipeline is not None:
        # cat_features are materialised into numeric columns by the sklearn stack.
        cat_features = []

    metadata["pipeline"] = pipeline
    metadata["extensions_pipeline"] = extensions_pipeline
    metadata["cat_features"] = cat_features
    metadata["columns"] = train_df.columns.tolist() if isinstance(train_df, pd.DataFrame) else train_df.columns

    if verbose:
        logger.info(f"  Pipeline done — train: {_df_shape_str(train_df)}, cat_features: {cat_features or '(none)'}")
        if was_polars_input and cat_features_polars:
            logger.info(f"  Pre-pipeline Polars cat_features: {cat_features_polars}")
        logger.info(f"  PHASE 3 total: {_elapsed_str(t0_phase3)}")

    # ==================================================================================
    # 4.5. AUTO-DETECT TEXT & EMBEDDING FEATURES
    # ==================================================================================

    # Use pre-pipeline DF for auto-detection (original dtypes preserved).
    detect_df = train_df_polars_pre if was_polars_input else train_df
    # Merge pipeline-detected and pre-pipeline Polars categorical columns
    raw_cat_features = list(set((cat_features or []) + (cat_features_polars or [])))
    # Honor ONLY strictly-user-declared pl.Categorical columns as already-assigned:
    # the user explicitly marked them as categorical so they must not be promoted to
    # text. pl.Utf8/String columns (which `cat_features_polars` also includes) and
    # pipeline-detected string cats are eligible for text-promotion based on
    # cardinality. `effective_cat_features` below removes promoted columns from the
    # cat list so no column double-counts.
    if was_polars_input:
        user_polars_cats = [
            c for c, dt in zip(detect_df.columns, detect_df.dtypes)
            if dt == pl.Categorical
        ]
    else:
        user_polars_cats = []
    text_features, embedding_features = _auto_detect_feature_types(
        detect_df, feature_types_config, user_polars_cats, verbose=verbose,
    )
    # Remove auto-detected text/embedding features from cat list (they're not categoricals)
    text_emb_set = set(text_features) | set(embedding_features)
    effective_cat_features = [c for c in raw_cat_features if c not in text_emb_set]
    _validate_feature_type_exclusivity(text_features, embedding_features, effective_cat_features)

    # CRITICAL: downstream code (select_target, strategy.build_pipeline, the CB
    # pandas-fallback path in _train_model_with_fallback, etc.) must see the
    # *deduplicated* list. Before this reassignment the unfiltered ``cat_features``
    # still contained text-promoted columns like 'category' / 'skills_text', and
    # CatBoost's pandas path then rejected the run with
    #   "column 'category' has dtype 'category' but is not in cat_features list"
    # — the column was pd.Categorical in the pandas view (preserved from the
    # original Polars schema) AND listed in text_features, so CB's Pool
    # refused to accept it. Reassigning here flows the correct set to every
    # downstream user via the single ``cat_features`` binding.
    cat_features = effective_cat_features

    # One-time Polars string→Categorical cast (shared across all models in the loop).
    # XGBoost's arrow bridge rejects pl.Utf8/large_string ("KeyError: DataType(large_string)");
    # HGB/LightGBM/CatBoost all prefer Categorical over raw strings when encoding is skipped.
    # Exclude text/embedding features — those must remain as raw strings for CatBoost text/emb handling.
    if was_polars_input and all_models_polars_native and pipeline_config.skip_categorical_encoding:
        _string_types = (pl.Utf8, pl.String) if hasattr(pl, "String") else (pl.Utf8,)
        _keep_as_string = text_emb_set
        def _precast_strings(df):
            if df is None:
                return df
            str_cols = [c for c, dt in zip(df.columns, df.dtypes)
                        if dt in _string_types and c not in _keep_as_string]
            return df.with_columns([pl.col(c).cast(pl.Categorical) for c in str_cols]) if str_cols else df
        _pre_train = _precast_strings(train_df)
        if _pre_train is not train_df:
            train_df = _pre_train
            val_df = _precast_strings(val_df)
            test_df = _precast_strings(test_df)
            # Always cast the pre-pipeline Polars refs too: downstream Polars fastpath
            # uses `train_df_polars_pre` as the model input, and without the cast XGBoost
            # would hit `DataType(large_string)` again. Identity-based re-pointing is not
            # sufficient — a no-op pipeline may return a new object that is not `is` train_df.
            train_df_polars_pre = _precast_strings(train_df_polars_pre)
            val_df_polars_pre = _precast_strings(val_df_polars_pre)
            test_df_polars_pre = _precast_strings(test_df_polars_pre)
            if verbose:
                logger.info("  Cast Polars string columns → Categorical once (shared across model loop)")

    if verbose and (text_features or embedding_features):
        logger.info(f"  Feature types — text: {text_features}, embedding: {embedding_features}, cat: {cat_features or '(none)'}")

    metadata["text_features"] = text_features
    metadata["embedding_features"] = embedding_features

    # ==================================================================================
    # 5. MODEL TRAINING
    # ==================================================================================

    if verbose:
        log_phase("PHASE 4: Model Training")

    # Initialize default values for training parameters
    with phase("initialize_training_defaults"):
        (
            init_common_params,
            rfecv_models,
            mrmr_kwargs,
        ) = _initialize_training_defaults(
            init_common_params=init_common_params,
            rfecv_models=rfecv_models,
            mrmr_kwargs=mrmr_kwargs,
        )

    # Get pipeline components (category_encoder, imputer, scaler) from params or defaults
    category_encoder, imputer, scaler = _get_pipeline_components(init_common_params, cat_features)

    # Compute trainset stats (Polars is more efficient, but pandas works too)
    if isinstance(train_df, pl.DataFrame):
        if verbose:
            logger.info("Computing trainset_features_stats on Polars...")
        with phase("trainset_features_stats", backend="polars"):
            trainset_features_stats = get_trainset_features_stats_polars(train_df)
    else:
        if verbose:
            logger.info("Computing trainset_features_stats on pandas...")
        with phase("trainset_features_stats", backend="pandas"):
            trainset_features_stats = get_trainset_features_stats(train_df)

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Actual training
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    if verbose:
        logger.info("Zero-copy conversion to pandas...")

    # Use pre-pipeline Polars originals for models that support native Polars input.
    # Post-pipeline DFs may have string/categorical columns converted to float by
    # polars-ds, losing dtype info needed by CatBoost and HGB.
    train_df_polars = train_df_polars_pre
    val_df_polars = val_df_polars_pre
    test_df_polars = test_df_polars_pre

    # Cache pandas versions for select_target (zero-copy Arrow-backed view for Polars).
    # Skip the conversion entirely when every model supports Polars natively — the Polars
    # fastpath in process_model substitutes Polars DFs back anyway, so pandas views would
    # be unused. Saves ~1-2s on CB-only runs on small-to-medium DFs.
    # sklearn 1.4+ accepts Polars DataFrames as input (verified on 1.7.2 with
    # IsolationForest, LocalOutlierFactor novelty=True, and Pipeline wrappers);
    # _apply_outlier_detection_global's boolean-mask filter handles both pandas and
    # Polars via _filter_df_by_mask. So we can skip the conversion regardless of
    # outlier_detector presence.
    # Guard: recurrent models use fit() signatures that predate Polars support
    # (core.py passes train_df_pd as `features_train=` / `val_features=` / `features=`).
    # Force pandas conversion if recurrent_models is non-empty.
    # RFECV goes through sklearn.feature_selection.RFECV, which calls estimator.fit(X)
    # where X is indexed via integer positional slicing internally — pandas-only path.
    # Force pandas conversion whenever any RFECV variant is requested.
    _has_rfecv = bool(rfecv_models)
    can_skip_pandas_conv = (
        was_polars_input and all_models_polars_native
        and not recurrent_models and not _has_rfecv
    )
    if can_skip_pandas_conv:
        train_df_pd, val_df_pd, test_df_pd = train_df, val_df, test_df
        if verbose:
            logger.info("  Skipped pandas conversion — all models are Polars-native (no outlier_detector)")
    else:
        # Diagnostic: on large high-cardinality frames the polars→pandas
        # conversion costs minutes (per-column dict rebuild for every split),
        # so if users assume fastpath is active but see the conversion fire
        # anyway they need to know *which condition* blocked the skip. Log
        # the exact reasons instead of the single non-skip signal.
        if verbose:
            reasons = []
            if not was_polars_input:
                reasons.append("input is not a Polars DataFrame")
            if not all_models_polars_native:
                non_native = [
                    m for m, s in zip(mlframe_models or [], _strategies_for_polars_check)
                    if not s.supports_polars
                ]
                reasons.append(
                    f"non-Polars-native models requested: {non_native}"
                    if non_native
                    else "all_models_polars_native=False (no strategies)"
                )
            if recurrent_models:
                reasons.append(f"recurrent_models={recurrent_models}")
            if _has_rfecv:
                reasons.append(f"rfecv_models={rfecv_models}")
            logger.info(
                f"  polars→pandas conversion needed because: {'; '.join(reasons) or 'unknown'}"
            )
        train_df_pd, val_df_pd, test_df_pd = _convert_dfs_to_pandas(train_df, val_df, test_df, verbose=verbose)

    # Ensure categorical features have pandas category dtype for CatBoost.
    # After the 2026-04-17 optimization, get_pandas_view_of_polars_df already
    # preserves Polars Categorical as pd.Categorical (int32 dict rebuild), so
    # this is usually a no-op. The call is kept for safety when pandas frames
    # come in with plain string columns that still need the category cast.
    #
    # Gate (item 11 of 2026-04-18 log triage): when the Polars fastpath is
    # active (can_skip_pandas_conv=True) the models receive the Polars DFs
    # directly — running prepare_df_for_catboost on the *pandas views* that
    # are built for select_target only is pointless work that shows up as
    # a 2+ minute blocking step in PHASE 4 on 1M×100 frames. Skip it in
    # that case.
    if cat_features and not can_skip_pandas_conv:
        if verbose:
            logger.info(f"Preparing {len(cat_features)} categorical features for CatBoost: {cat_features}")
        for df_pd in [train_df_pd, val_df_pd, test_df_pd]:
            if df_pd is not None:
                prepare_df_for_catboost(df_pd, cat_features)
    elif cat_features and can_skip_pandas_conv and verbose:
        logger.info(
            f"Skipping pandas-side CatBoost prep for {len(cat_features)} categorical "
            "features — Polars fastpath receives the DFs natively."
        )

    # B2: Release post-pipeline Polars DFs after pandas conversion.
    # Arrow-backed pandas views hold their own Arrow buffer references,
    # so the Polars objects are no longer needed. Saves ~100GB peak memory.
    if was_polars_input and needs_polars_pre_clone:
        # Only release if we cloned (otherwise train_df IS train_df_polars_pre)
        train_df = val_df = test_df = None
        baseline_rss_mb = maybe_clean_ram_and_gpu(baseline_rss_mb, df_size_mb, verbose=verbose, reason="post-pipeline Polars release")
        if verbose:
            logger.info("  Released post-pipeline Polars DFs (pandas views retained)")

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
        baseline_rss_mb=baseline_rss_mb,
        df_size_mb=df_size_mb,
    )

    # Single global OD result (not per-target)
    outlier_detection_result = {
        "train_od_idx": train_od_idx,
        "val_od_idx": val_od_idx,
    }

    # Surface OD row-reduction evidence in returned metadata so callers/tests can assert
    # that OD actually ran without grepping logs. Only populated when an OD was applied.
    if outlier_detector is not None:
        n_train_pre_od = len(train_df_pd) if train_df_pd is not None else None
        n_val_pre_od = len(val_df_pd) if val_df_pd is not None else None
        n_train_post_od = int(train_od_idx.sum()) if train_od_idx is not None else n_train_pre_od
        n_val_post_od = int(val_od_idx.sum()) if val_od_idx is not None else n_val_pre_od
        n_train_dropped = (n_train_pre_od - n_train_post_od) if n_train_pre_od is not None else 0
        n_val_dropped = (n_val_pre_od - n_val_post_od) if (n_val_pre_od is not None and val_od_idx is not None) else 0
        metadata["outlier_detection"] = {
            "applied": True,
            "n_outliers_dropped_train": int(n_train_dropped),
            "n_outliers_dropped_val": int(n_val_dropped),
            "train_size_after_od": int(n_train_post_od) if n_train_post_od is not None else None,
            "val_size_after_od": int(n_val_post_od) if n_val_post_od is not None else None,
        }
    else:
        metadata["outlier_detection"] = {"applied": False}

    # Keep polars fastpath DFs in sync with pandas-filtered copies so that the
    # Polars-native training path operates on OD-filtered rows matching the
    # OD-filtered targets. Without this the downstream training call feeds the
    # unfiltered Polars DF but the OD-filtered target → length mismatch.
    if train_od_idx is not None and train_df_polars is not None:
        train_df_polars = train_df_polars.filter(pl.Series(train_od_idx))
    if val_od_idx is not None and val_df_polars is not None:
        val_df_polars = val_df_polars.filter(pl.Series(val_od_idx))

    # Save metadata EARLY (before training loops) so that if training is interrupted,
    # already-trained models are still usable with the saved pipeline/preprocessing
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

    models = defaultdict(lambda: defaultdict(list))

    # Track mapping from slugified names to original names for load_mlframe_suite
    slug_to_original_target_type = {}
    slug_to_original_target_name = {}

    for target_type, targets in tqdmu(target_by_type.items(), desc="target type"):
        # Store original target_type mapping
        slug_to_original_target_type[slugify(str(target_type).lower())] = target_type

        # !TODO ! optimize for creation of inner feature matrices of cb,lgb,xgb here. They should be created once per featureset, not once per target.
        for cur_target_name, cur_target_values in tqdmu(targets.items(), desc="target"):
            # Store original cur_target_name mapping
            slug_to_original_target_name[slugify(cur_target_name)] = cur_target_name
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
                    save_charts=save_charts,
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

                t0_select_target = timer()
                # Build common_params and behavior_config for select_target
                od_common_params, current_behavior_config = _build_common_params_for_target(
                    init_common_params=init_common_params,
                    trainset_features_stats=trainset_features_stats,
                    plot_file=plot_file,
                    train_od_idx=train_od_idx,
                    val_od_idx=val_od_idx,
                    current_train_target=current_train_target,
                    current_val_target=current_val_target,
                    outlier_detector=outlier_detector,
                    behavior_config=behavior_config,
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
                    text_features=text_features,
                    embedding_features=embedding_features,
                    hyperparams_config=hyperparams_config,
                    behavior_config=current_behavior_config,
                    common_params=od_common_params,
                    mlframe_models=mlframe_models,
                    linear_model_config=linear_model_config,
                )

            if verbose:
                logger.info(f"  select_target done in {_elapsed_str(t0_select_target)}")
                log_ram_usage()

            # Build list of pre-pipelines (feature selection methods) to try
            pre_pipelines, pre_pipeline_names = _build_pre_pipelines(
                use_ordinary_models=use_ordinary_models,
                rfecv_models=rfecv_models,
                rfecv_models_params=rfecv_models_params,
                use_mrmr_fs=use_mrmr_fs,
                mrmr_kwargs=mrmr_kwargs,
                custom_pre_pipelines=custom_pre_pipelines,
            )

            # Initialize pipeline cache ONCE - preprocessing output is reused across pre_pipelines.
            # Since custom transformers run AFTER preprocessing, the preprocessing output is the same
            # for ordinary models and all custom pipelines with the same model type (linear, tree, etc).
            pipeline_cache = PipelineCache()

            for pre_pipeline, pre_pipeline_name in tqdmu(zip(pre_pipelines, pre_pipeline_names), desc="pre_pipeline", total=len(pre_pipelines)):
                # Skip CatBoost RFECV pipeline with metamodel_func due to sklearn clone issue
                if _should_skip_catboost_metamodel(pre_pipeline_name.strip(), target_type, behavior_config):
                    continue
                ens_models = [] if use_mlframe_ensembles else None
                orig_pre_pipeline = pre_pipeline

                # Build weight schemas from extractor output
                if sample_weights:
                    weight_schemas = sample_weights
                    if "uniform" in sample_weights:
                        logger.info(f"Using {len(weight_schemas)} weighting schema(s) from extractor: {list(weight_schemas.keys())}")
                    else:
                        logger.info(f"Using {len(weight_schemas)} weighting schema(s) from extractor: {list(weight_schemas.keys())}. Note: uniform weighting not included.")
                else:
                    weight_schemas = {"uniform": None}
                    logger.info("No weighting schemas from extractor, defaulting to uniform weighting.")

                # -----------------------------------------------------------------------
                # MODEL LOOP: Train each model type with all weight variations
                # Models sorted by feature tier (most features first) so that
                # text/embedding columns are dropped once per tier, not per model.
                # -----------------------------------------------------------------------
                # Resolve strategies once and reuse — avoids re-calling get_strategy() inside
                # sort key, main loop, and tier-transition check (was ~3x redundant calls).
                # Keyed by id(m) so unhashable estimator instances / tuples don't break lookup,
                # and two identical-by-value-but-distinct-by-identity entries stay distinct.
                strategy_by_model = {id(m): get_strategy(m) for m in mlframe_models}
                sorted_models = sorted(
                    mlframe_models,
                    key=lambda m: strategy_by_model[id(m)].feature_tier(),
                    reverse=True,  # (True, True) before (False, False)
                )
                tier_dfs_cache = {}  # feature_tier -> {train_df, val_df, test_df}
                prev_tier = None

                for mlframe_model_name in tqdmu(sorted_models, desc="mlframe model"):
                    # Skip CatBoost model with metamodel_func due to sklearn clone issue
                    if _should_skip_catboost_metamodel(mlframe_model_name, target_type, behavior_config):
                        continue

                    if mlframe_model_name not in models_params:
                        logger.warning(f"mlframe model {mlframe_model_name} not known, skipping...")
                        continue

                    # Use strategy pattern to determine pipeline and cache key
                    strategy = strategy_by_model[id(mlframe_model_name)]
                    pre_pipeline = strategy.build_pipeline(
                        base_pipeline=orig_pre_pipeline,
                        cat_features=cat_features,
                        category_encoder=category_encoder if cat_features else None,
                        imputer=imputer,
                        scaler=scaler,
                    )
                    # Cache key composition (2026-04-19 round-9 verify):
                    #   1. strategy.cache_key (e.g. "tree", "hgb", "neural")
                    #   2. pre_pipeline_name (MRMR vs RFECV vs ordinary)
                    #   3. feature_tier() — CRITICAL: CB/LGB/XGB all inherit
                    #      cache_key="tree", but CB has tier=(True,True) while
                    #      LGB/XGB have tier=(False,False). Without tier in the
                    #      key, CB (running first by tier-desc sort) cached its
                    #      polars DF with text/embedding cols under "tree";
                    #      LGB/XGB then retrieved that cache and received cols
                    #      they can't handle (either polars when they need
                    #      pandas, or text cols they don't support). Adding
                    #      feature_tier() partitions the cache so same-tier
                    #      models still share, different-tier models don't
                    #      collide.
                    _tier_suffix = f"_tier{strategy.feature_tier()}"
                    cache_key = f"{strategy.cache_key}_{pre_pipeline_name}{_tier_suffix}" if pre_pipeline_name else f"{strategy.cache_key}{_tier_suffix}"

                    # Polars fastpath: substitute original Polars DataFrames for models
                    # that support native Polars input (e.g. CatBoost >= 1.2.7, HGB).
                    polars_fastpath_active = train_df_polars is not None and strategy.supports_polars

                    # B3: Prepare Polars DFs once per model (outside weight loop).
                    # prepare_polars_dataframe() calls .with_columns() which allocates —
                    # doing it per weight schema wastes memory for 100GB+ DataFrames.
                    if polars_fastpath_active:
                        if verbose:
                            logger.info(f"  Polars fastpath active for {mlframe_model_name} (strategy={type(strategy).__name__})")
                        # Use ``cat_features`` (the post-promotion, deduplicated
                        # list from the Phase 3.5 reassignment at line ~1526) —
                        # NOT the stale ``cat_features_polars`` captured at the
                        # start of Phase 3 before auto-detection ran.
                        # Production 2026-04-19 bug: the old
                        #   _cat_features = cat_features_polars or cat_features or []
                        # short-circuit picked the stale 13-column list even
                        # after text-promotion removed 4 of them. Those 4 were
                        # later cast Categorical→String for CatBoost's text
                        # path, so CB saw cat_features=[..., 'skills_text'] with
                        # skills_text being pl.String — its fused-cpdef
                        # ``_set_features_order_data_polars_categorical_column``
                        # has no String overload and raised "No matching
                        # signature found", burning 22 s + a 150 s pandas
                        # fallback on every run.
                        _cat_features = list(cat_features or [])

                        # Build tier-specific DFs with text/embedding columns dropped for non-supporting models
                        tier_base = {
                            "train_df": train_df_polars,
                            "val_df": val_df_polars,
                            "test_df": test_df_polars,
                        }
                        tier_polars = _build_tier_dfs(
                            tier_base, strategy, text_features, embedding_features, tier_dfs_cache, verbose=verbose,
                        )

                        prepared_train = strategy.prepare_polars_dataframe(tier_polars["train_df"], _cat_features)
                        prepared_val = strategy.prepare_polars_dataframe(tier_polars["val_df"], _cat_features) if tier_polars.get("val_df") is not None else None
                        prepared_test = strategy.prepare_polars_dataframe(tier_polars["test_df"], _cat_features) if tier_polars.get("test_df") is not None else None

                        # Null-fill text features for CatBoost (requires no nulls in text columns).
                        # Also cast Polars Categorical/Enum to pl.String: CatBoost's Polars
                        # fastpath for text_features (`_set_features_order_data_polars_text_column`)
                        # rejects Categorical with
                        #   "Unsupported data type Categorical for a text feature column".
                        # This happens whenever a column was auto-promoted from cat_features to
                        # text_features (done in `_auto_detect_feature_types`) but its backing
                        # dtype was left as Categorical — CB's text path requires plain string.
                        if text_features and mlframe_model_name == "cb":
                            text_cols_present = filter_existing(prepared_train, text_features)
                            if text_cols_present:
                                # Determine which of the text columns need a dtype cast.
                                needs_cast = [
                                    c for c in text_cols_present
                                    if prepared_train.schema[c] == pl.Categorical
                                    or isinstance(prepared_train.schema[c], pl.Enum)
                                ]
                                prep_exprs = []
                                for c in text_cols_present:
                                    expr = pl.col(c)
                                    if c in needs_cast:
                                        expr = expr.cast(pl.String)
                                    prep_exprs.append(expr.fill_null(""))
                                prepared_train = prepared_train.with_columns(prep_exprs)
                                if prepared_val is not None:
                                    prepared_val = prepared_val.with_columns(prep_exprs)
                                if prepared_test is not None:
                                    prepared_test = prepared_test.with_columns(prep_exprs)
                                if needs_cast and verbose:
                                    logger.info(
                                        f"  Cast {len(needs_cast)} text feature(s) from Polars Categorical to String "
                                        f"for CatBoost: {needs_cast}"
                                    )

                        polars_fastpath_skip_preprocessing = strategy.requires_encoding
                    else:
                        polars_fastpath_skip_preprocessing = False

                        # For non-Polars models, build tier DFs from pandas common_params
                        tier_pandas = _build_tier_dfs(
                            {"train_df": common_params.get("train_df"), "val_df": common_params.get("val_df"), "test_df": common_params.get("test_df")},
                            strategy, text_features, embedding_features, tier_dfs_cache, verbose=verbose,
                        )

                    # --- WEIGHT SCHEMA LOOP: Train with each sample weighting ---
                    for weight_name, weight_values in tqdmu(weight_schemas.items(), desc="weighting schema"):
                        # Create model name with weight suffix
                        model_name_with_weight = common_params["model_name"]
                        model_file_name=f"{mlframe_model_name}"
                        if weight_name != "uniform":
                            model_name_with_weight += f" w={weight_name}"
                            model_file_name +=f"_{weight_name}"

                        # Shallow copy common_params - only sample_weight changes per iteration
                        current_common_params = common_params.copy()
                        current_common_params["sample_weight"] = weight_values

                        # Apply tier DFs (text/embedding columns dropped for non-supporting models)
                        if polars_fastpath_active:
                            current_common_params["train_df"] = prepared_train
                            if prepared_val is not None:
                                current_common_params["val_df"] = prepared_val
                            if prepared_test is not None:
                                current_common_params["test_df"] = prepared_test
                        else:
                            current_common_params["train_df"] = tier_pandas["train_df"]
                            if tier_pandas.get("val_df") is not None:
                                current_common_params["val_df"] = tier_pandas["val_df"]
                            if tier_pandas.get("test_df") is not None:
                                current_common_params["test_df"] = tier_pandas["test_df"]

                        # Append weight_name to plot_file for non-uniform weights
                        if weight_name != "uniform" and current_common_params.get("plot_file"):
                            current_common_params["plot_file"] = current_common_params["plot_file"] + weight_name + "_"

                        # Check if we have cached transformed DataFrames for this pipeline type
                        cached_dfs = pipeline_cache.get(cache_key)

                        # ============================================================================
                        # INTENTIONAL: Clone model for EACH weight schema iteration.
                        # DO NOT "OPTIMIZE" BY MOVING CLONE OUTSIDE THE LOOP!
                        # ============================================================================
                        # Each weight schema produces a DIFFERENT trained model that gets stored
                        # separately in models[type][target]. Without cloning per iteration:
                        #   - All entries would point to the SAME sklearn object (last-trained state)
                        #   - In-memory model.model.predict() would give WRONG results
                        #   - Only saved .dump files would work correctly (they capture snapshots)
                        # The clone() cost is negligible compared to training time.
                        # ============================================================================
                        original_model = models_params[mlframe_model_name]["model"]
                        try:
                            cloned_model = clone(original_model)
                        except RuntimeError:
                            # CatBoost wraps custom eval_metric objects internally, causing sklearn's
                            # identity check (param1 is not param2) to fail. Fall back to direct
                            # constructor call with the same params, which produces an equivalent
                            # fresh unfitted model without the verification step.
                            cloned_model = type(original_model)(**original_model.get_params())
                        except TypeError:
                            # NGBoost: get_params() exposes attributes (validation_fraction,
                            # early_stopping_rounds) that __init__ doesn't accept. Filter get_params
                            # to those actually in the signature.
                            import inspect as _inspect
                            _cls = type(original_model)
                            _sig_params = set(_inspect.signature(_cls.__init__).parameters) - {"self"}
                            _raw = original_model.get_params(deep=False)
                            cloned_model = _cls(**{k: v for k, v in _raw.items() if k in _sig_params})
                        # Preserve the mlframe posthoc calibration tag across clone.
                        # sklearn.clone() strips non-param attributes, which would
                        # drop the calibration directive and silently revert to an
                        # uncalibrated model. Fix 2026-04-15.
                        if getattr(original_model, "_mlframe_posthoc_calibrate", False):
                            try:
                                cloned_model._mlframe_posthoc_calibrate = True
                            except Exception:
                                pass
                        current_model_params = models_params[mlframe_model_name].copy()
                        current_model_params["model"] = cloned_model

                        # Polars fastpath: update cat_features/text_features/embedding_features
                        # in fit_params for CatBoost only.
                        # XGBoost/HGB auto-detect pl.Categorical via enable_categorical=True
                        # and do NOT accept cat_features/text_features/embedding_features as fit() params.
                        if polars_fastpath_active and mlframe_model_name == "cb" and "fit_params" in current_model_params:
                            extra_fit = {}
                            if _cat_features:
                                _valid_cat = _filter_polars_cat_features_by_dtype(
                                    prepared_train, _cat_features
                                )
                                if _valid_cat:
                                    extra_fit["cat_features"] = _valid_cat
                            if text_features:
                                cb_text = filter_existing(prepared_train, text_features)
                                if cb_text:
                                    extra_fit["text_features"] = cb_text
                            if embedding_features:
                                cb_emb = filter_existing(prepared_train, embedding_features)
                                if cb_emb:
                                    extra_fit["embedding_features"] = cb_emb
                            if extra_fit:
                                current_model_params["fit_params"] = {**current_model_params["fit_params"], **extra_fit}

                        # Build process_model kwargs using helper
                        process_model_kwargs = _build_process_model_kwargs(
                            model_file=model_file,
                            model_name_with_weight=model_name_with_weight,
                            model_file_name=model_file_name,
                            target_type=target_type,
                            pre_pipeline=pre_pipeline,
                            pre_pipeline_name=pre_pipeline_name,
                            cur_target_name=cur_target_name,
                            models=models,
                            model_params=current_model_params,
                            common_params=current_common_params,
                            ens_models=ens_models,
                            trainset_features_stats=trainset_features_stats,
                            verbose=verbose,
                            cached_dfs=cached_dfs,
                            polars_pipeline_applied=polars_pipeline_applied or polars_fastpath_skip_preprocessing,
                            mlframe_model_name=mlframe_model_name,
                            metadata_columns=metadata.get("columns"),
                        )

                        t0_model = timer()
                        with phase("process_model", model=mlframe_model_name, weight=weight_name):
                            trainset_features_stats, pre_pipeline, train_df_transformed, val_df_transformed, test_df_transformed = process_model(
                                **process_model_kwargs
                            )
                        if verbose:
                            logger.info(f"  process_model({mlframe_model_name}, w={weight_name}) done — {_elapsed_str(t0_model)}")

                        # Cache the transformed DataFrames if not already cached
                        if cached_dfs is None:
                            pipeline_cache.set(cache_key, train_df_transformed, val_df_transformed, test_df_transformed)

                    # Update orig_pre_pipeline for tree models only.
                    # Tree models return just the base_pipeline (feature selector) from build_pipeline(),
                    # so after process_model() fits it, we preserve the fitted version for subsequent models.
                    # Non-tree models wrap base_pipeline in a full Pipeline (with encoder/imputer/scaler),
                    # which we don't want to use as the base for other model types.
                    # For optimal performance, list tree models first in mlframe_models.
                    if cache_key.startswith("tree"):
                        orig_pre_pipeline = pre_pipeline

                    # B5: Release Polars originals after all tier-1 (Polars-native) models finish.
                    # When transitioning to a lower tier, pre-pipeline Polars DFs are no longer needed.
                    cur_tier = strategy.feature_tier()
                    if prev_tier is not None and cur_tier != prev_tier and not strategy.supports_polars:
                        if train_df_polars is not None:
                            del train_df_polars, val_df_polars, test_df_polars
                            train_df_polars = val_df_polars = test_df_polars = None
                            tier_dfs_cache.clear()
                            baseline_rss_mb = maybe_clean_ram_and_gpu(baseline_rss_mb, df_size_mb, verbose=verbose, reason="tier transition")
                            if verbose:
                                logger.info("  Released pre-pipeline Polars originals (tier transition)")
                    prev_tier = cur_tier

                if ens_models and len(ens_models) > 1:
                    if verbose:
                        logger.info(f"evaluating simple ensembles...")
                    # Get feature count from transformed DataFrame for display
                    ens_n_features = train_df_transformed.shape[1] if train_df_transformed is not None else None
                    _ensembles = score_ensemble(  # Result used for side effects (logging/metrics)
                        models_and_predictions=ens_models,
                        ensemble_name=f"{pre_pipeline_name}{len(ens_models)}models ",
                        n_features=ens_n_features,
                        **common_params,
                    )

    # ==================================================================================
    # 6. RECURRENT MODEL TRAINING
    # ==================================================================================

    if recurrent_models and (train_sequences is not None or train_df is not None):
        if verbose:
            log_phase("PHASE 5: Recurrent Model Training")

        from .trainer import _configure_recurrent_params

        # Determine if this is a regression task
        use_regression = TargetTypes.REGRESSION in target_by_type

        # Configure recurrent model parameters
        recurrent_params = _configure_recurrent_params(
            recurrent_models=recurrent_models,
            recurrent_config=recurrent_config,
            sequences_train=train_sequences,
            features_train=train_df_pd if train_df_pd is not None else train_df,
            use_regression=use_regression,
        )

        # Train recurrent models
        for recurrent_model_name in tqdmu(recurrent_models, desc="recurrent model"):
            model_name_lower = recurrent_model_name.lower()
            if model_name_lower not in recurrent_params:
                logger.warning(f"Recurrent model {recurrent_model_name} not configured, skipping...")
                continue

            recurrent_model = recurrent_params[model_name_lower]["model"]

            # Iterate over target types and targets
            for target_type, targets in target_by_type.items():
                for cur_target_name, target_values in targets.items():
                    if verbose:
                        logger.info(f"Training {recurrent_model_name} for target {cur_target_name}...")

                    # Extract train/val/test targets
                    train_target = target_values[train_idx] if hasattr(target_values, '__getitem__') else target_values.iloc[train_idx]
                    val_target = target_values[val_idx] if val_idx is not None and hasattr(target_values, '__getitem__') else None
                    test_target = target_values[test_idx] if hasattr(target_values, '__getitem__') else target_values.iloc[test_idx]

                    # Convert to numpy if needed
                    if hasattr(train_target, 'to_numpy'):
                        train_target = train_target.to_numpy()
                    elif hasattr(train_target, 'values'):
                        train_target = train_target.values

                    if val_target is not None:
                        if hasattr(val_target, 'to_numpy'):
                            val_target = val_target.to_numpy()
                        elif hasattr(val_target, 'values'):
                            val_target = val_target.values

                    if hasattr(test_target, 'to_numpy'):
                        test_target = test_target.to_numpy()
                    elif hasattr(test_target, 'values'):
                        test_target = test_target.values

                    # Clone model for this target
                    model_clone = clone(recurrent_model)

                    try:
                        # Fit the model
                        model_clone.fit(
                            sequences=train_sequences,
                            features=train_df_pd if train_df_pd is not None else None,
                            labels=train_target,
                            val_sequences=val_sequences,
                            val_features=val_df_pd if val_df_pd is not None else None,
                            val_labels=val_target,
                        )

                        # Store the trained model
                        models[target_type][cur_target_name].append(model_clone)

                        if verbose:
                            logger.info(f"Successfully trained {recurrent_model_name} for {cur_target_name}")

                    except Exception as e:
                        logger.error(f"Failed to train {recurrent_model_name} for {cur_target_name}: {e}")
                        continue

    if verbose:
        log_phase(f"Training suite completed for {model_name}, {sum(len(v) for targets in models.values() for v in targets.values())} models.")
        log_ram_usage()

    # Aggregate per-model fairness_report into metadata so callers can access it without
    # re-walking the models dict. Trainer stores fairness_report in model.metrics[split].
    # Fix date 2026-04-15: bug B (fairness_report not propagated into suite metadata).
    fairness_reports: Dict[str, Any] = {}
    for _ttype, _targets in models.items():
        for _tname, _model_list in _targets.items():
            for _m in _model_list:
                _m_metrics = getattr(_m, "metrics", None)
                if not isinstance(_m_metrics, dict):
                    continue
                for _split in ("test", "val", "train"):
                    _split_metrics = _m_metrics.get(_split)
                    if isinstance(_split_metrics, dict) and "fairness_report" in _split_metrics:
                        _key = f"{_ttype}__{_tname}__{getattr(_m, 'model_name', type(getattr(_m, 'model', _m)).__name__)}__{_split}"
                        fairness_reports[_key] = _split_metrics["fairness_report"]
    if fairness_reports:
        metadata["fairness_report"] = fairness_reports

    # Save metadata again with slug-to-original name mappings (for load_mlframe_suite)
    _finalize_and_save_metadata(
        metadata=metadata,
        outlier_detector=outlier_detector,
        outlier_detection_result=outlier_detection_result,
        trainset_features_stats=trainset_features_stats,
        data_dir=data_dir,
        models_dir=models_dir,
        target_name=target_name,
        model_name=model_name,
        verbose=0,  # Silent to avoid duplicate log messages
        slug_to_original_target_type=slug_to_original_target_type,
        slug_to_original_target_name=slug_to_original_target_name,
    )

    if verbose:
        logger.info("[phases] Top phases by wall-clock time:\n" + format_phase_summary())

    return dict(models), metadata


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

    metadata_file = join(models_path, "metadata.joblib")
    if not exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    if verbose:
        logger.info(f"Loading metadata from {metadata_file}...")
    # Default trusted_root to the models directory if not provided — matches the
    # in-process "we just wrote this file" flow while still refusing path escape.
    _root = trusted_root if trusted_root is not None else os.path.abspath(models_path)
    _validate_trusted_path(metadata_file, _root)
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
        # Drop extra columns (target, etc.) — unified via helper.
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
        # Also expose the ensemble inside the per-model probabilities dict under
        # the canonical "ensemble" key so downstream consumers can iterate a
        # single dict and see both per-model and ensemble streams (2026-04-15).
        if isinstance(results.get("probabilities"), dict):
            results["probabilities"]["ensemble"] = avg_probs

        # Ensemble predictions from averaged probabilities
        if avg_probs.ndim == 2:
            ensemble_preds = (avg_probs[:, 1] > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
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
        logger.info(f"Generated predictions for {len(results['predictions'])} models")

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
        # Drop extra columns (target, etc.) — unified via helper.
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
                    logger.info(f"Predicting with model: {model_name}")

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

                        # For binary classification, get class 1 probability for threshold
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

        # Ensemble predictions from averaged probabilities
        if avg_probs.ndim == 2:
            ensemble_preds = (avg_probs[:, 1] > DEFAULT_PROBABILITY_THRESHOLD).astype(int)
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
        logger.info(f"Generated predictions for {len(results['predictions'])} models")

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

    metadata_file = join(models_path, "metadata.joblib")
    if not exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    _root = trusted_root if trusted_root is not None else os.path.abspath(models_path)
    _validate_trusted_path(metadata_file, _root)
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
