"""Leaf-level utility helpers extracted from core.py.

27 helper functions covering: logging visibility, metric extraction, column
augmentation/restoration across splits, frame-shape pretty-print, dataset-reuse
capability detection, input-column-vs-metadata validation, feature-type
auto-detection + exclusivity guard, outlier-detection orchestrator,
directory setup, common-params builder, pre-pipeline builder, process-model
kwargs assembly, Polars->pandas conversion, pipeline-component extraction,
fairness-subgroup compute, CB-metamodel skip heuristic, metadata
initialisation + finalisation.

Each function is a leaf (zero internal-helper dependencies on its peers in
this module). Pulled out of core.py so the giant train_mlframe_models_suite
orchestrator lives in isolation; core.py re-exports every symbol below at
its bottom for full back-compat.
"""
from __future__ import annotations

import glob
import logging
import os
import sys
from collections import defaultdict
from copy import deepcopy
from os.path import exists, join
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import joblib
import numpy as np
import pandas as pd
import polars as pl
import psutil
import scipy.stats as stats
from pyutilz.strings import slugify
from pyutilz.system import (
    clean_ram, ensure_dir_exists, tqdmu, tqdmu_lazy_start,
)
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import category_encoders as ce

from ..configs import (
    BaselineDiagnosticsConfig,
    CompositeTargetDiscoveryConfig,
    ConfidenceAnalysisConfig,
    DummyBaselinesConfig,
    FeatureSelectionConfig,
    FeatureTypesConfig,
    LinearModelConfig,
    ModelHyperparamsConfig,
    MultilabelDispatchConfig,
    OutlierDetectionConfig,
    OutputConfig,
    PreprocessingBackendConfig,
    PreprocessingConfig,
    PreprocessingExtensionsConfig,
    QuantileRegressionConfig,
    ReportingConfig,
    TargetTypes,
    TrainingBehaviorConfig,
    TrainingConfig,
    TrainingSplitConfig,
)
from ..extractors import FeaturesAndTargetsExtractor
from ..helpers import (
    get_trainset_features_stats,
    get_trainset_features_stats_polars,
)
from ..io import load_mlframe_model
from ..models import LINEAR_MODEL_TYPES, is_linear_model
from ..phases import format_phase_summary, phase, reset_phase_registry
from ..pipeline import (
    apply_preprocessing_extensions,
    fit_and_transform_pipeline,
    prepare_df_for_catboost,
)
from ..preprocessing import (
    create_split_dataframes,
    load_and_prepare_dataframe,
    preprocess_dataframe,
    save_split_artifacts,
)
from ..splitting import make_train_test_split
from ..strategies import (
    PipelineCache,
    get_polars_cat_columns,
    get_strategy,
)
from ..utils import (
    compute_model_input_fingerprint,
    drop_columns_from_dataframe,
    estimate_df_size_mb,
    filter_existing,
    get_pandas_view_of_polars_df,
    get_process_rss_mb,
    log_phase,
    log_ram_usage,
    maybe_clean_ram_and_gpu,
)
from mlframe.feature_selection.filters import MRMR
from mlframe.metrics import create_fairness_subgroups

logger = logging.getLogger(__name__)


# Module-level constants

DEFAULT_PROBABILITY_THRESHOLD: float = 0.5


def _phase_pandas_conversion_and_cat_prep(
    *,
    train_df,
    val_df,
    test_df,
    train_df_polars_pre,
    val_df_polars_pre,
    test_df_polars_pre,
    cat_features,
    was_polars_input,
    all_models_polars_native,
    needs_polars_pre_clone,
    mlframe_models,
    recurrent_models,
    rfecv_models,
    baseline_rss_mb,
    df_size_mb,
    verbose,
):
    """Phase 4 pre-loop (pandas conversion + CatBoost cat prep + Polars release).

    Two main responsibilities:

    1. **Pandas conversion gating**. Skip the polars->pandas conversion entirely
       when all models are Polars-native (CB/XGB on supported builds) OR when
       only non-native sklearn models block the fastpath (those do their own
       lazy conversion later). Forces conversion when recurrent_models or
       rfecv_models are requested (those paths predate Polars support).

    2. **CatBoost cat-feature prep + size capture**. Calls
       ``prepare_df_for_catboost`` on the pandas views when actually needed
       (i.e. when conversion wasn't skipped). Captures Polars-side estimated
       sizes BEFORE conversion to avoid the pathological pandas
       ``memory_usage(deep=True)`` scan downstream.

    3. **Post-pandas Polars release**. When a clone was made, frees the
       post-pipeline Polars frames after conversion -- the Arrow-backed
       pandas views hold their own buffer references, so the Polars objects
       are no longer needed (~100 GB peak saved on the user's 4M-row TVT run).

    Returns
    -------
    13-tuple:
        train_df_pd, val_df_pd, test_df_pd,
        train_df_polars, val_df_polars, test_df_polars,
        train_df, val_df, test_df (possibly cleared to None on Polars release),
        train_df_size_bytes_cached, val_df_size_bytes_cached,
        can_skip_pandas_conv, baseline_rss_mb_refreshed
    """
    if verbose:
        logger.info("Zero-copy conversion to pandas...")

    # Pre-pipeline Polars references for the Polars fastpath.
    train_df_polars = train_df_polars_pre
    val_df_polars = val_df_polars_pre
    test_df_polars = test_df_polars_pre

    # Re-resolve strategies locally -- cheap O(M) lookup.
    strategies_for_check = [get_strategy(m) for m in mlframe_models] if mlframe_models else []

    _has_rfecv = bool(rfecv_models)
    _has_non_native_mlframe_strategy = was_polars_input and not all_models_polars_native
    can_skip_pandas_conv = (
        was_polars_input
        and not recurrent_models and not _has_rfecv
        and (all_models_polars_native or _has_non_native_mlframe_strategy)
    )

    # Pre-conversion Polars size capture (Fix 3B).
    train_df_size_bytes_cached: Optional[float] = None
    val_df_size_bytes_cached: Optional[float] = None
    if was_polars_input:
        try:
            if isinstance(train_df, pl.DataFrame):
                train_df_size_bytes_cached = float(train_df.estimated_size())
            if val_df is not None and isinstance(val_df, pl.DataFrame):
                val_df_size_bytes_cached = float(val_df.estimated_size())
        except Exception:
            train_df_size_bytes_cached = None
            val_df_size_bytes_cached = None

    if can_skip_pandas_conv:
        train_df_pd, val_df_pd, test_df_pd = train_df, val_df, test_df
        if verbose:
            if all_models_polars_native:
                logger.info("  Skipped pandas conversion -- all models are Polars-native")
            else:
                non_native = [
                    m for m, s in zip(mlframe_models or [], strategies_for_check)
                    if not s.supports_polars
                ]
                logger.info(
                    "  Deferred pandas conversion -- Polars-native models run on the fastpath; "
                    "non-native %s will convert lazily at their strategy branch.",
                    non_native,
                )
    else:
        if verbose:
            reasons = []
            if not was_polars_input:
                reasons.append("input is not a Polars DataFrame")
            if not all_models_polars_native:
                non_native = [
                    m for m, s in zip(mlframe_models or [], strategies_for_check)
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
                "  polars->pandas conversion needed because: %s",
                "; ".join(reasons) or "unknown",
            )
        train_df_pd, val_df_pd, test_df_pd = _convert_dfs_to_pandas(train_df, val_df, test_df, verbose=verbose)

    # CatBoost cat-feature prep.
    if cat_features and not can_skip_pandas_conv:
        if verbose:
            logger.info("Preparing %d categorical features for CatBoost: %s", len(cat_features), cat_features)
        for df_pd in [train_df_pd, val_df_pd, test_df_pd]:
            if df_pd is not None:
                prepare_df_for_catboost(df_pd, cat_features)
    elif cat_features and can_skip_pandas_conv and verbose:
        logger.info(
            "Skipping pandas-side CatBoost prep for %d categorical "
            "features -- Polars fastpath receives the DFs natively.",
            len(cat_features),
        )

    # B2 -- post-pipeline Polars release.
    if was_polars_input and needs_polars_pre_clone:
        train_df = val_df = test_df = None
        baseline_rss_mb = maybe_clean_ram_and_gpu(baseline_rss_mb, df_size_mb, verbose=verbose, reason="post-pipeline Polars release")
        if verbose:
            logger.info("  Released post-pipeline Polars DFs (pandas views retained)")

    if verbose:
        log_ram_usage()

    return (
        train_df_pd, val_df_pd, test_df_pd,
        train_df_polars, val_df_polars, test_df_polars,
        train_df, val_df, test_df,
        train_df_size_bytes_cached, val_df_size_bytes_cached,
        can_skip_pandas_conv, baseline_rss_mb,
    )


def _log_cardinality_and_drift_snapshot(
    *,
    train_df,
    val_df,
    test_df,
    cat_features,
    text_features,
    embedding_features,
) -> None:
    """Pre-train cardinality + val/test drift logging (pure side-effect).

    Cardinality: without this, a native XGB/CB crash on high-cardinality
    categoricals leaves us guessing at the input.

    Drift: for time-ordered splits (the common case here), val and test can
    contain category values that never appeared in train -- XGB 3.x on
    Windows crashes silently during val IterativeDMatrix construction when
    this happens (observed 2026-04-20 on prod_jobsdetails). Helper emits a
    WARNING for any column with non-trivial train-vs-val drift along with a
    concrete healing suggestion keyed on the train-side cardinality so the
    operator sees the crash suspect BEFORE the kernel dies.

    Skip if cardinality > 100k (text-sized columns): the anti-join is
    expensive and unseen-category semantics don't cleanly apply to free-text
    columns (CB handles them via TF-IDF, XGB drops them).

    Pure-logging helper -- no return value, no mutation of any inputs.
    """
    all_cat_cols = list(cat_features or []) + list(text_features or []) + list(embedding_features or [])
    if not (all_cat_cols and train_df is not None):
        return
    try:
        _DRIFT_SKIP_CARD = 100_000
        is_polars = isinstance(train_df, pl.DataFrame)
        pairs = []
        for c in all_cat_cols:
            if c not in train_df.columns:
                continue
            if is_polars:
                n_unique = train_df[c].n_unique()
            else:
                n_unique = int(train_df[c].nunique(dropna=False))
            pairs.append((c, n_unique))
        pairs.sort(key=lambda x: -x[1])
        summary = ", ".join(f"{c}:{n:_}" for c, n in pairs)
        logger.info("  Categorical cardinalities (train, n_unique, desc): %s", summary)

        # Drift log: val/test categories not seen in train.
        if is_polars and val_df is not None and test_df is not None and val_df.height > 0:
            drift_rows = []
            for c, card_train in pairs:
                if card_train > _DRIFT_SKIP_CARD:
                    continue
                if c not in val_df.columns or c not in test_df.columns:
                    continue
                tr_uniq = train_df.select(pl.col(c).drop_nulls().unique().alias(c))
                v_uniq  = val_df.select(pl.col(c).drop_nulls().unique().alias(c))
                te_uniq = test_df.select(pl.col(c).drop_nulls().unique().alias(c))
                val_only  = v_uniq.join(tr_uniq, on=c, how="anti").height
                test_only = te_uniq.join(tr_uniq, on=c, how="anti").height
                drift_rows.append((c, card_train, val_only, test_only))

            if drift_rows:
                drift_rows.sort(key=lambda x: -x[2])
                drift_summary = ", ".join(
                    f"{c}:val_only={v},test_only={t}"
                    for c, _, v, t in drift_rows if v > 0 or t > 0
                ) or "(none)"
                logger.info("  Category drift (val/test values missing from train): %s", drift_summary)

                # WARN + healing-suggestion for non-trivial train-vs-val drift.
                # Test-side drift reported above for visibility but NOT used
                # in healing decisions (would leak test info into training).
                for c, card_tr, v_only, t_only in drift_rows:
                    if v_only == 0 and t_only == 0:
                        continue
                    v_frac = v_only / max(card_tr, 1)
                    if v_only >= 5 or v_frac >= 0.05:
                        if card_tr >= 1000:
                            _healing = (
                                f"        suggested actions (pick one):\n"
                                f"          a) hash-bucket via FeatureHasher / target-encoding "
                                f"(card {card_tr:_} >= 1 000 -> model will memorize train-only "
                                f"values and generalize poorly on val/test);\n"
                                f"          b) drop '{c}' from cat_features and keep only the "
                                f"top-K most frequent (K=100-300) as one-hot, route the rest "
                                f"into an '__OTHER__' bucket;\n"
                                f"          c) drop '{c}' entirely if it's an identifier or "
                                f"free-text field -- promote to text_features via use_text_features=True "
                                f"so CatBoost handles it natively and other backends ignore it."
                            )
                        elif card_tr >= 100:
                            _healing = (
                                f"        suggested actions (pick one):\n"
                                f"          a) target-encoding (CatBoostEncoder) to collapse "
                                f"{card_tr:_} levels into a continuous feature;\n"
                                f"          b) keep top-K by train frequency, bucket the rest "
                                f"into '__OTHER__' before fit (K~=30-80)."
                            )
                        else:
                            _healing = (
                                f"        suggested actions (pick one):\n"
                                f"          a) add an explicit '__UNSEEN__' bucket in the "
                                f"Enum domain so val values absent from train resolve to a "
                                f"known category instead of raising;\n"
                                f"          b) widen the training window (temporal split) so "
                                f"val_only categories are observed at fit time."
                            )
                        logger.warning(
                            f"  Category drift suspect: {c} -- val has {v_only} categories "
                            f"({v_frac:.1%} of train card {card_tr:_}) that train never saw. "
                            f"XGB/CB may crash when constructing val DMatrix with ref=train.\n"
                            f"{_healing}"
                        )
    except Exception as _e:
        logger.warning(f"  Failed to compute categorical cardinality/drift: {_e}")


def _phase_auto_detect_feature_types(
    *,
    train_df,
    val_df,
    test_df,
    train_df_polars_pre,
    val_df_polars_pre,
    test_df_polars_pre,
    cat_features,
    cat_features_polars,
    was_polars_input,
    all_models_polars_native,
    pipeline_config,
    feature_types_config,
    metadata,
    verbose,
):
    """Phase 3.5 (Auto-detect text + embedding features) of
    ``train_mlframe_models_suite``.

    Steps:
    1. Run ``_auto_detect_feature_types`` on the pre-pipeline frame (original
       dtypes) using the pre-pipeline cat_features list + user-declared
       Polars categoricals.
    2. When ``use_text_features=False``, drop the high-card columns from the
       train/val/test splits AND the pre-pipeline clones; capture pre-drop
       column data for dummy-baselines ``per_group_mean`` downstream.
    3. Compute ``effective_cat_features`` (raw cat - text/embedding).
    4. Validate feature-type exclusivity.
    5. One-time Polars string -> Categorical cast across all frames so XGB's
       arrow bridge doesn't choke on ``large_string`` later.

    Mutates ``metadata`` in-place with ``columns`` (post-drop) and
    ``cat_features`` (post-effective).

    Returns
    -------
    11-tuple:
        train_df, val_df, test_df,
        train_df_polars_pre, val_df_polars_pre, test_df_polars_pre,
        text_features, embedding_features, cat_features,
        text_emb_set, dropped_high_card_data
    """
    # Use pre-pipeline DF for auto-detection (original dtypes preserved).
    detect_df = train_df_polars_pre if was_polars_input else train_df
    raw_cat_features = list(set((cat_features or []) + (cat_features_polars or [])))
    # Honor ONLY strictly-user-declared pl.Categorical columns as already-assigned.
    if was_polars_input:
        user_polars_cats = [
            c for c, dt in zip(detect_df.columns, detect_df.dtypes)
            if dt == pl.Categorical
        ]
    else:
        user_polars_cats = []
    text_features, embedding_features, auto_high_card_drop = _auto_detect_feature_types(
        detect_df, feature_types_config, user_polars_cats, verbose=verbose,
    )

    # Capture pre-drop column DATA so dummy_baselines per_group_mean can use
    # these as group keys downstream. Tree models drop them to avoid XGB
    # QuantileDMatrix OOM, but a simple groupby gives an excellent baseline.
    dropped_high_card_data = {}
    if auto_high_card_drop:
        for _col in auto_high_card_drop:
            _col_frames = {}
            for _label, _frame in (("train", train_df), ("val", val_df), ("test", test_df)):
                if _frame is None:
                    continue
                _cols = _frame.columns if hasattr(_frame, "columns") else []
                if _col not in _cols:
                    continue
                try:
                    if isinstance(_frame, pl.DataFrame):
                        _col_frames[_label] = _frame[_col].to_numpy()
                    else:
                        _col_frames[_label] = np.asarray(_frame[_col])
                except Exception:
                    continue
            if _col_frames:
                dropped_high_card_data[_col] = _col_frames
        train_df = _drop_cols_df(train_df, auto_high_card_drop)
        val_df = _drop_cols_df(val_df, auto_high_card_drop)
        test_df = _drop_cols_df(test_df, auto_high_card_drop)
        if was_polars_input:
            if train_df_polars_pre is not None:
                train_df_polars_pre = _drop_cols_df(train_df_polars_pre, auto_high_card_drop)
            if val_df_polars_pre is not None:
                val_df_polars_pre = _drop_cols_df(val_df_polars_pre, auto_high_card_drop)
            if test_df_polars_pre is not None:
                test_df_polars_pre = _drop_cols_df(test_df_polars_pre, auto_high_card_drop)
        raw_cat_features = [c for c in raw_cat_features if c not in auto_high_card_drop]
        metadata["columns"] = train_df.columns.tolist() if isinstance(train_df, pd.DataFrame) else train_df.columns

    text_emb_set = set(text_features) | set(embedding_features)
    effective_cat_features = [c for c in raw_cat_features if c not in text_emb_set]
    _validate_feature_type_exclusivity(text_features, embedding_features, effective_cat_features)
    cat_features = effective_cat_features
    metadata["cat_features"] = cat_features

    # One-time Polars string->Categorical cast shared across all models.
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
            train_df_polars_pre = _precast_strings(train_df_polars_pre)
            val_df_polars_pre = _precast_strings(val_df_polars_pre)
            test_df_polars_pre = _precast_strings(test_df_polars_pre)
            if verbose:
                logger.info("  Cast Polars string columns -> Categorical once (shared across model loop)")

    if verbose and (text_features or embedding_features):
        logger.info("  Feature types -- text: %s, embedding: %s, cat: %s", text_features, embedding_features, cat_features or '(none)')

    return (
        train_df, val_df, test_df,
        train_df_polars_pre, val_df_polars_pre, test_df_polars_pre,
        text_features, embedding_features, cat_features,
        text_emb_set, dropped_high_card_data,
    )


def _phase_fit_pipeline(
    *,
    train_df,
    val_df,
    test_df,
    mlframe_models,
    pipeline_config,
    preprocessing_config,
    feature_types_config,
    preprocessing_extensions,
    metadata,
    verbose,
):
    """Phase 3 (Pipeline Fitting & Transformation) of ``train_mlframe_models_suite``.

    Decomposes datetime columns BEFORE the pre-pipeline clone, saves
    Polars originals for the fastpath when needed, runs
    ``fit_and_transform_pipeline`` (categorical encoding + imputation +
    scaling + ensure_float32), then applies any
    ``PreprocessingExtensionsConfig`` (custom scaler / poly / dim-reducer).

    Mutates ``metadata`` in-place with ``pipeline``, ``extensions_pipeline``,
    ``cat_features``, ``columns``.

    Returns
    -------
    15-tuple:
        train_df, val_df, test_df,
        pipeline, extensions_pipeline,
        cat_features, cat_features_polars,
        was_polars_input, all_models_polars_native, polars_pipeline_applied,
        train_df_polars_pre, val_df_polars_pre, test_df_polars_pre,
        pipeline_config (possibly updated), preprocessing_extensions (possibly normalised)
    """
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
    if was_polars_input and not pipeline_config.skip_categorical_encoding:
        if all_models_polars_native:
            pipeline_config = pipeline_config.model_copy(update={"skip_categorical_encoding": True})
            if verbose:
                logger.info("  All models %s support Polars natively -- skipping categorical encoding in pipeline", mlframe_models)

    # 2026-04-24 (fuzz extension): datetime columns must be decomposed
    # BEFORE the pre-pipeline clone, otherwise ``train_df_polars_pre`` and
    # friends retain the raw datetime and reach downstream (linear
    # pre_pipeline, MRMR, sklearn encoders, CB Pool) where numpy /
    # sklearn / CB all raise on DateTime64DType.
    def _detect_datetime_cols(df_):
        if df_ is None:
            return []
        if isinstance(df_, pl.DataFrame):
            return [name for name, dt in df_.schema.items()
                    if isinstance(dt, (pl.Datetime, pl.Date))]
        if hasattr(df_, "dtypes"):
            return [c for c in df_.columns
                    if pd.api.types.is_datetime64_any_dtype(df_[c])]
        return []

    _dt_cols = _detect_datetime_cols(train_df)
    if _dt_cols:
        from mlframe.feature_engineering.basic import create_date_features
        _dt_methods = {
            "day": np.int8,
            "weekday": np.int8,
            "month": np.int8,
            "hour": np.int8,
        }
        if verbose:
            logger.info(
                "Decomposing %d datetime column(s) into numeric features "
                "(day/weekday/month/hour) before pre-pipeline clone: %s",
                len(_dt_cols), _dt_cols,
            )
        train_df = create_date_features(
            train_df, cols=_dt_cols, delete_original_cols=True,
            methods=_dt_methods,
        )
        if val_df is not None:
            v_cols = [c for c in _dt_cols if c in val_df.columns]
            if v_cols:
                val_df = create_date_features(
                    val_df, cols=v_cols, delete_original_cols=True,
                    methods=_dt_methods,
                )
        if test_df is not None:
            t_cols = [c for c in _dt_cols if c in test_df.columns]
            if t_cols:
                test_df = create_date_features(
                    test_df, cols=t_cols, delete_original_cols=True,
                    methods=_dt_methods,
                )

    # Save pre-pipeline Polars originals for the Polars fastpath.
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
        logger.info("  fit_and_transform_pipeline done in %s", _elapsed_str(t0_fit_pipeline))

    polars_pipeline_applied = was_polars_input and pipeline_config.prefer_polarsds and pipeline is not None

    # Apply shared sklearn-based extensions
    if preprocessing_extensions is not None and isinstance(preprocessing_extensions, dict):
        preprocessing_extensions = PreprocessingExtensionsConfig(**preprocessing_extensions)
    t0_ext = timer()
    train_df, val_df, test_df, extensions_pipeline = apply_preprocessing_extensions(
        train_df, val_df, test_df, preprocessing_extensions, verbose=verbose,
    )
    if verbose and preprocessing_extensions is not None:
        logger.info("  apply_preprocessing_extensions done in %s", _elapsed_str(t0_ext))
    if extensions_pipeline is not None:
        cat_features = []

    metadata["pipeline"] = pipeline
    metadata["extensions_pipeline"] = extensions_pipeline
    metadata["cat_features"] = cat_features
    metadata["columns"] = train_df.columns.tolist() if isinstance(train_df, pd.DataFrame) else train_df.columns

    if verbose:
        logger.info("  Pipeline done -- train: %s, cat_features: %s", _df_shape_str(train_df), cat_features or '(none)')
        if was_polars_input and cat_features_polars and list(cat_features_polars) != list(cat_features or []):
            logger.info("  Pre-pipeline Polars cat_features: %s", cat_features_polars)
        logger.info("  PHASE 3 total: %s", _elapsed_str(t0_phase3))

    return (
        train_df, val_df, test_df,
        pipeline, extensions_pipeline,
        cat_features, cat_features_polars,
        was_polars_input, all_models_polars_native, polars_pipeline_applied,
        train_df_polars_pre, val_df_polars_pre, test_df_polars_pre,
        pipeline_config, preprocessing_extensions,
    )


def _phase_train_val_test_split(
    *,
    df,
    target_by_type,
    timestamps,
    group_ids,
    group_ids_raw,
    artifacts,
    sequences,
    split_config,
    behavior_config,
    metadata,
    data_dir,
    models_dir,
    target_name,
    model_name,
    df_size_mb,
    verbose,
):
    """Phase 2 (Train/Val/Test Splitting) of ``train_mlframe_models_suite``.

    Resolves auto-stratification + group-aware splitting flags from the
    config + extractor side-channels, calls ``make_train_test_split``, saves
    split artifacts to disk when ``data_dir`` is set, computes fairness
    subgroups on the pre-split frame, materialises ``train/val/test`` splits
    of the dataframe + sequences, frees the original df, refreshes the RSS
    baseline.

    Mutates ``metadata`` in-place with split sizes + train/val/test details.

    Returns
    -------
    15-tuple:
        train_idx, val_idx, test_idx,
        train_details, val_details, test_details,
        train_df, val_df, test_df,
        fairness_subgroups, fairness_features,
        train_sequences, val_sequences, test_sequences,
        baseline_rss_mb_refreshed
    """
    if verbose:
        log_phase("PHASE 2: Train/Val/Test Splitting")

    t0_phase2 = timer()
    if verbose:
        logger.info(f"Making train_val_test split...")
    # Auto-stratify by target for classification when no timestamps are
    # available. Without this, the unstratified shuffle path can hand
    # an unlucky val/test slice with zero minority-class rows for
    # rare imbalance ratios (fuzz default-seed c0134, seed=99 c0040 --
    # rare_1pct + binary class produces 50 positives out of 5000;
    # random 400-row val_shuf can land all-class-0). Stratification
    # preserves class proportions across train/val/test. Skipped when
    # timestamps are present (the splitter prefers temporal ordering
    # there) or for multitarget setups where picking ONE target as
    # the stratify key is arbitrary.
    _stratify_y = None
    if timestamps is None and isinstance(target_by_type, dict):
        _classification_targets = []
        _has_multilabel = False
        for _tt, _named in target_by_type.items():
            _tt_name = getattr(_tt, "name", str(_tt)).upper()
            if "MULTILABEL" in _tt_name:
                _has_multilabel = True
                break
            if "CLASS" in _tt_name and isinstance(_named, dict):
                for _tn, _tv in _named.items():
                    if _tv is not None:
                        _classification_targets.append(_tv)
        # Multilabel stratification needs the optional ``iterative-
        # stratification`` package. Skip it to avoid forcing the dep on
        # users who don't have it (the ``MultilabelStratifiedShuffleSplit``
        # branch raises ``ModuleNotFoundError`` deep in the splitter).
        # Single-label classification (binary / multiclass) uses sklearn's
        # built-in ``StratifiedShuffleSplit`` which is always available.
        if _has_multilabel:
            _stratify_y = None
        elif len(_classification_targets) == 1:
            try:
                _arr = np.asarray(_classification_targets[0])
                # Guard: only stratify when stratification is meaningful --
                # all classes have at least 2 rows, otherwise sklearn's
                # StratifiedShuffleSplit raises "least populated class has
                # only 1 member". Also limit to 1-D targets -- 2-D would
                # route to the multilabel splitter (already excluded above
                # but defense in depth).
                if _arr.ndim == 1:
                    _u, _c = np.unique(_arr, return_counts=True)
                    if len(_u) >= 2 and _c.min() >= 2:
                        _stratify_y = _arr
            except Exception:
                _stratify_y = None
    # Group-aware splitting opt-in. When the extractor produced ``group_ids``
    # (e.g. ``SimpleFeaturesAndTargetsExtractor(group_field="well_id")``) and
    # ``split_config.use_groups`` is True (default), route through
    # ``GroupShuffleSplit`` so that no well straddles train/val/test.
    _groups = group_ids if (split_config.use_groups and group_ids is not None and len(group_ids) > 0) else None
    with phase("split_data"):
        train_idx, val_idx, test_idx, train_details, val_details, test_details = make_train_test_split(
            df=df,
            timestamps=timestamps,
            stratify_y=_stratify_y,
            groups=_groups,
            **split_config.model_dump(exclude={"use_groups"}),
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
    fairness_subgroups, fairness_features = _compute_fairness_subgroups(df, behavior_config)
    if verbose:
        if fairness_features and fairness_subgroups is None:
            logger.warning(f"Fairness features {fairness_features} specified but subgroups could not be computed")
        elif fairness_subgroups is not None:
            logger.info("Computed %d fairness subgroups", len(fairness_subgroups))

    # Create split dataframes
    train_df, val_df, test_df = create_split_dataframes(
        df=df,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    if verbose:
        logger.info("  Split shapes -- train: %s, val: %s, test: %s", _df_shape_str(train_df), _df_shape_str(val_df), _df_shape_str(test_df))
        logger.info("  PHASE 2 total: %s", _elapsed_str(t0_phase2))

    # Split sequences by train/val/test indices (for recurrent models)
    train_sequences, val_sequences, test_sequences = None, None, None
    if sequences is not None:
        train_sequences = [sequences[i] for i in train_idx]
        val_sequences = [sequences[i] for i in val_idx] if val_idx is not None else None
        test_sequences = [sequences[i] for i in test_idx]
        if verbose:
            logger.info("Split sequences: train=%d, val=%d, test=%d", len(train_sequences), len(val_sequences) if val_sequences else 0, len(test_sequences))

    # Delete original df to free RAM (caller already did ``del df`` after
    # we return; we still nudge the GC + arena-release here because the
    # baseline-RSS refresh is meaningful only AFTER the parent frees df).
    if verbose:
        logger.info("Deleting original DataFrame to free RAM...")

    # Caller's `del df` happens after the return; we refresh baseline using
    # the current RSS (which will reflect the post-del state on the next
    # ``maybe_clean_ram_and_gpu`` call inside the caller).
    baseline_rss_mb = get_process_rss_mb()
    baseline_rss_mb = maybe_clean_ram_and_gpu(baseline_rss_mb, df_size_mb, verbose=verbose, reason="post-split (del df)")
    if verbose:
        log_ram_usage()

    return (
        train_idx, val_idx, test_idx,
        train_details, val_details, test_details,
        train_df, val_df, test_df,
        fairness_subgroups, fairness_features,
        train_sequences, val_sequences, test_sequences,
        baseline_rss_mb,
    )


def _phase_load_and_preprocess(
    *,
    df,
    preprocessing_config,
    features_and_targets_extractor,
    recurrent_models,
    sequences,
    verbose,
):
    """Phase 1 (Data Loading & Preprocessing) of ``train_mlframe_models_suite``.

    Loads the input dataframe (file path or in-memory), runs the
    features-and-targets extractor to surface ``target_by_type`` +
    side-channels (timestamps, group ids, sample weights, artifacts), extracts
    sequences for any recurrent models, then drops the FTE-flagged columns
    and runs the final preprocessing pass.

    Captures the RAM baseline + DF-size estimate AFTER the FTE so the
    downstream ``maybe_clean_ram_and_gpu`` calls have a meaningful
    pre-transient-allocation reference point.

    Returns
    -------
    11-tuple:
        df, target_by_type, group_ids_raw, group_ids, timestamps, artifacts,
        additional_columns_to_drop, sample_weights, baseline_rss_mb,
        df_size_mb, sequences
    """
    if verbose:
        log_phase("PHASE 1: Data Loading & Preprocessing")

    # Load and prepare dataframe
    t0_phase1 = timer()
    with phase("load_and_prepare_dataframe"):
        df = load_and_prepare_dataframe(df, preprocessing_config, verbose=verbose)
    if verbose:
        logger.info("  load_and_prepare_dataframe done -- %s, %s", _df_shape_str(df), _elapsed_str(t0_phase1))

    # Apply features_and_targets_extractor to extract targets
    if verbose:
        logger.info("Create additional features & extracting targets...")

    t0_fte = timer()
    df, target_by_type, group_ids_raw, group_ids, timestamps, artifacts, additional_columns_to_drop, sample_weights = features_and_targets_extractor.transform(
        df
    )
    if verbose:
        logger.info("  features_and_targets_extractor done -- %s, %s", _df_shape_str(df), _elapsed_str(t0_fte))

    # Capture baseline RSS + DF size NOW -- before any downstream steps that may allocate
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
                logger.info("Extracted %d sequences from DataFrame", len(sequences))
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
        logger.info("  preprocess_dataframe done -- %s, %s", _df_shape_str(df), _elapsed_str(t0_preproc))
        logger.info("  PHASE 1 total: %s", _elapsed_str(t0_phase1))

    return (
        df, target_by_type, group_ids_raw, group_ids, timestamps,
        artifacts, additional_columns_to_drop, sample_weights,
        baseline_rss_mb, df_size_mb, sequences,
    )


def _build_suite_common_params_dict(
    *,
    reporting_config,
    preprocessing_config,
    confidence_analysis_config,
) -> Dict[str, Any]:
    """Assemble the ``common_params_dict`` carried down through the suite.

    Three sources feed in:
    - ``reporting_config`` -- dumped via ``.model_dump(exclude=...)`` with
      ``title_metrics_tokens`` excluded (derived field auto-recomputed by
      ``_build_configs_from_params``) and ``plot_inline_display`` excluded
      (consumed at suite level only).
    - ``preprocessing_config`` -- conditionally adds ``scaler`` / ``imputer`` /
      ``category_encoder`` when each is non-None.
    - ``confidence_analysis_config`` -- 8 scalar fields under the
      ``confidence_analysis_*`` / ``include_confidence_analysis`` prefix.

    Pure read-only function: no side effects, no mutation of the input
    configs. Tests can unit-check the output dict without spinning up the
    full suite.
    """
    common: Dict[str, Any] = {}
    common.update(
        reporting_config.model_dump(exclude={
            "title_metrics_tokens",
            "plot_inline_display",
        })
    )
    if preprocessing_config.scaler is not None:
        common["scaler"] = preprocessing_config.scaler
    if preprocessing_config.imputer is not None:
        common["imputer"] = preprocessing_config.imputer
    if preprocessing_config.category_encoder is not None:
        common["category_encoder"] = preprocessing_config.category_encoder
    common["include_confidence_analysis"] = confidence_analysis_config.include
    common["confidence_analysis_use_shap"] = confidence_analysis_config.use_shap
    common["confidence_analysis_max_features"] = confidence_analysis_config.max_features
    common["confidence_analysis_cmap"] = confidence_analysis_config.cmap
    common["confidence_analysis_alpha"] = confidence_analysis_config.alpha
    common["confidence_analysis_ylabel"] = confidence_analysis_config.ylabel
    common["confidence_analysis_title"] = confidence_analysis_config.title
    common["confidence_model_kwargs"] = dict(confidence_analysis_config.model_kwargs)
    return common


def _maybe_dispatch_to_ltr_ranker_suite(
    *,
    target_type,
    df,
    target_name,
    model_name,
    features_and_targets_extractor,
    mlframe_models,
    use_mlframe_ensembles,
    ranking_config,
    split_config,
    hyperparams_config,
    reporting_config,
    output_config,
    verbose,
):
    """If ``target_type == LEARNING_TO_RANK``, route to the focused ranker suite.

    The standard classification/regression machinery in ``train_mlframe_models_suite``
    doesn't know how to consume per-row scores or per-query metrics, so we hand
    LTR runs off to ``train_mlframe_ranker_suite`` (CB/XGB/LGB native rankers
    + RRF/Borda ensembling). Helper inspects the bag of config objects, mirrors
    the relevant fields onto the ranker-suite signature, and returns its result.

    Returns
    -------
    ``None`` when the call site is NOT LTR (caller continues with the standard
    pipeline); the ranker-suite return tuple otherwise (caller forwards verbatim).
    """
    if target_type is None or target_type != TargetTypes.LEARNING_TO_RANK:
        return None
    from mlframe.training.ranker_suite import train_mlframe_ranker_suite

    # Resolve a save_dir from output_config if available, else None.
    _save_dir = None
    if output_config is not None:
        _data_dir = (
            output_config.get("data_dir") if isinstance(output_config, dict)
            else getattr(output_config, "data_dir", None)
        )
        _models_dir = (
            output_config.get("models_dir") if isinstance(output_config, dict)
            else getattr(output_config, "models_dir", None)
        ) or "models"
        if _data_dir:
            _save_dir = os.path.join(_data_dir, _models_dir, model_name)

    # Pull split sizes from split_config if provided.
    _test_size, _val_size = 0.15, 0.15
    if split_config is not None:
        _test_size = (
            split_config.get("test_size", 0.15) if isinstance(split_config, dict)
            else getattr(split_config, "test_size", 0.15)
        )
        _val_size = (
            split_config.get("val_size", 0.15) if isinstance(split_config, dict)
            else getattr(split_config, "val_size", 0.15)
        )

    # Hyperparams from hyperparams_config if provided.
    _iter, _lr, _es = 200, 0.1, 30
    _mlp_kwargs = None
    if hyperparams_config is not None:
        _iter = (
            hyperparams_config.get("iterations", 200) if isinstance(hyperparams_config, dict)
            else getattr(hyperparams_config, "iterations", 200)
        )
        _lr = (
            hyperparams_config.get("learning_rate", 0.1) if isinstance(hyperparams_config, dict)
            else getattr(hyperparams_config, "learning_rate", 0.1)
        )
        _es = (
            hyperparams_config.get("early_stopping_rounds", 30) if isinstance(hyperparams_config, dict)
            else getattr(hyperparams_config, "early_stopping_rounds", 30)
        )
        # mlp_kwargs forwarded to MLPRanker.__init__ when LTR + 'mlp'
        # is in mlframe_models. Mirrors the non-LTR mlp_kwargs path
        # via _configure_mlp_params; users who want to flip
        # enable_checkpointing, change hidden_layers, etc. for the
        # ranker MLP should put those keys in
        # ``hyperparams_config["mlp_kwargs"]``.
        _mlp_kwargs = (
            hyperparams_config.get("mlp_kwargs", None) if isinstance(hyperparams_config, dict)
            else getattr(hyperparams_config, "mlp_kwargs", None)
        )

    # Reporting wiring (auto-emit LTR panel grid per (model, split)).
    _plot_outputs = None
    _ltr_panels = None
    if reporting_config is not None:
        _plot_outputs = (
            reporting_config.get("plot_outputs") if isinstance(reporting_config, dict)
            else getattr(reporting_config, "plot_outputs", None)
        )
        _ltr_panels = (
            reporting_config.get("ltr_panels") if isinstance(reporting_config, dict)
            else getattr(reporting_config, "ltr_panels", None)
        )
    _plot_file = None
    if output_config is not None:
        _plot_file = (
            output_config.get("plot_file") if isinstance(output_config, dict)
            else getattr(output_config, "plot_file", None)
        )

    return train_mlframe_ranker_suite(
        df=df,
        target_name=target_name,
        model_name=model_name,
        features_and_targets_extractor=features_and_targets_extractor,
        mlframe_models=mlframe_models,
        use_mlframe_ensembles=use_mlframe_ensembles,
        ranking_config=ranking_config,
        test_size=_test_size,
        val_size=_val_size,
        iterations=_iter,
        learning_rate=_lr,
        early_stopping_rounds=_es,
        save_dir=_save_dir,
        verbose=verbose,
        plot_file=_plot_file,
        plot_outputs=_plot_outputs,
        ltr_panels=_ltr_panels,
        mlp_kwargs=_mlp_kwargs,
    )


def _ensure_logging_visible(level: int = logging.INFO) -> None:
    """Make mlframe progress logs visible -- with timestamps -- in Jupyter and
    plain scripts.

    Two cases:
      1. No root handlers at all (bare Python / fresh Jupyter kernel before
         anyone called `logging.basicConfig`). -> install a stdout handler
         with ``%(asctime)s %(levelname)s %(name)s: %(message)s`` format.
      2. Root already has handlers but their formatter lacks a timestamp
         (classic Jupyter/IPython default emits just ``LEVEL:name:message``,
         which is useless for profiling long training runs). -> replace
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




def _entry_metric(entry, split: str, name: str) -> float:
    """Pull a per-split per-name metric value out of an entry.

    Entries shipped by the per-target loop carry a ``metrics`` mapping
    in any of several shapes (legacy), e.g.:
    - ``entry.metrics[split][name]`` -> {'train': {'RMSE': 0.42, ...}}
    - ``entry.metrics[name]`` -> {'RMSE': 0.42, ...} (split-less)
    - flat key like ``f"{split}_{name}"``: ``{'train_RMSE': 0.42}``

    Returns ``float('nan')`` on any miss so callers can treat absent
    metrics uniformly.
    """
    metrics = getattr(entry, "metrics", None)
    if not isinstance(metrics, dict):
        return float("nan")
    inner = metrics.get(split)
    if isinstance(inner, dict):
        v = inner.get(name)
        if isinstance(v, (int, float)):
            return float(v)
    v = metrics.get(name)
    if isinstance(v, (int, float)):
        return float(v)
    v = metrics.get(f"{split}_{name}")
    if isinstance(v, (int, float)):
        return float(v)
    return float("nan")




def _augment_with_dropped_high_card_cols(
    dropped_data,
    train_df,
    val_df,
    test_df,
    *,
    train_od_idx=None,
    val_od_idx=None,
):
    """Re-attach pre-drop high-card cat columns to ``train/val/test_df``.

    Used by the dummy_baselines per-target call site so per_group_mean
    can use group keys (e.g. ``well_id`` with 600+ unique values) that
    were stripped from tree-model frames to prevent XGB QuantileDMatrix
    OOM. Captured pre-drop ndarrays are sliced by ``train_od_idx`` /
    ``val_od_idx`` so the re-added column row-aligns to the OD-filtered
    frame; test is never OD-filtered. Returns
    ``(train_df, val_df, test_df, added_col_names)``.
    """
    added = []
    if not dropped_data:
        return train_df, val_df, test_df, added

    train_extras, val_extras, test_extras = {}, {}, {}
    n_train = len(train_df) if train_df is not None else 0
    n_val = len(val_df) if val_df is not None else 0
    n_test = len(test_df) if test_df is not None else 0

    for col, data in dropped_data.items():
        if "train" in data and train_df is not None:
            arr = data["train"]
            if train_od_idx is not None and len(arr) != n_train:
                arr_aligned = arr[train_od_idx] if len(arr) == len(train_od_idx) else None
            elif len(arr) == n_train:
                arr_aligned = arr
            else:
                arr_aligned = None
            if arr_aligned is not None and len(arr_aligned) == n_train:
                train_extras[col] = arr_aligned
        if "val" in data and val_df is not None:
            arr = data["val"]
            if val_od_idx is not None and len(arr) != n_val:
                arr_aligned = arr[val_od_idx] if len(arr) == len(val_od_idx) else None
            elif len(arr) == n_val:
                arr_aligned = arr
            else:
                arr_aligned = None
            if arr_aligned is not None and len(arr_aligned) == n_val:
                val_extras[col] = arr_aligned
        if "test" in data and test_df is not None:
            arr = data["test"]
            if len(arr) == n_test:
                test_extras[col] = arr
        if col in train_extras:
            added.append(col)

    if not added:
        return train_df, val_df, test_df, added

    def _attach(frame, extras):
        if frame is None or not extras:
            return frame
        if isinstance(frame, pl.DataFrame):
            return frame.with_columns([pl.Series(c, v) for c, v in extras.items()])
        return frame.assign(**extras)

    return (
        _attach(train_df, train_extras),
        _attach(val_df, val_extras),
        _attach(test_df, test_extras),
        added,
    )




def _build_full_column_from_splits(
    col_name,
    train_df,
    val_df,
    test_df,
    train_idx,
    val_idx,
    test_idx,
    n_total,
):
    """Reassemble a single column at the FULL n_total row index space
    from the per-split frames produced upstream of the per-target loop.

    Used by composite-target discovery integration: discovery's
    ``forward`` transform needs the base column at every row in the
    full df (so val and test rows get T values for the per-target
    training loop's slicing). The base column is split across
    ``train_df`` / ``val_df`` / ``test_df`` after the suite's
    train/val/test partition; this helper writes each split's column
    back into a single n_total-sized ndarray indexed by the original
    split indices.

    Parameters
    ----------
    col_name : str
        Column to extract.
    train_df, val_df, test_df : pandas.DataFrame | polars.DataFrame | None
        Per-split frames. ``val_df`` / ``test_df`` may be None when
        the suite was configured without a val or test split.
    train_idx, val_idx, test_idx : ndarray | None
        Row indices INTO the n_total-sized full frame.
    n_total : int
        Size of the full row index space.

    Returns
    -------
    ndarray
        Float64 array of length ``n_total`` with the column values
        slotted at the appropriate indices. Rows not covered by any
        split (rare, but possible when the FTE drops rows) keep NaN.
    """
    import numpy as _np
    out = _np.full(n_total, _np.nan, dtype=_np.float64)
    for _split_df, _split_idx in (
        (train_df, train_idx), (val_df, val_idx), (test_df, test_idx),
    ):
        if _split_df is None or _split_idx is None:
            continue
        if col_name not in _split_df.columns:
            continue
        try:
            col_vals = _split_df[col_name].to_numpy() if hasattr(_split_df[col_name], "to_numpy") \
                else _np.asarray(_split_df[col_name])
        except Exception:
            continue
        col_vals = _np.asarray(col_vals).reshape(-1).astype(_np.float64, copy=False)
        idx_arr = _np.asarray(_split_idx).reshape(-1)
        if len(col_vals) != len(idx_arr):
            # Frame and index disagree (e.g. OD-filtered train_df
            # paired with raw train_idx). Skip rather than
            # mis-aligning silently.
            continue
        out[idx_arr] = col_vals
    return out




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
    """Format DataFrame shape as 'rowsxcols' with thousands separators."""
    if df is None:
        return "None"
    nrows = df.shape[0] if hasattr(df, "shape") else len(df)
    ncols = df.shape[1] if hasattr(df, "shape") else 0
    return f"{nrows:_}x{ncols}"




def _elapsed_str(start: float) -> str:
    """Format elapsed time since start as human-readable string."""
    elapsed = timer() - start
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    return f"{elapsed / 60:.1f}min"




def _detect_dataset_reuse_capabilities() -> "Dict[str, bool]":
    """Feature-detect which GBDT sklearn wrappers can accept a pre-built
    dataset as ``X``, enabling label/weight reuse across fits without
    rebuilding the native data structure.

    Matrix of capability keys:

    - ``cb_pool_set_label``: ``catboost.Pool.set_label`` callable.
    - ``cb_pool_set_weight``: ``catboost.Pool.set_weight`` callable.
    - ``cb_pool_label_swap``: both of the above AND
      ``CatBoostClassifier.fit(X=Pool)`` short-circuits the rebuild
      (verified via ``_build_train_pool`` code path in the installed
      catboost build).
    - ``xgb_dmatrix_set_label`` / ``xgb_dmatrix_set_weight``: ``DMatrix``
      exposes both mutators (true in every 3.x release).
    - ``xgb_sklearn_accepts_dmatrix``: ``XGBClassifier.fit(X=DMatrix)``
      short-circuits -- empirically False in 3.2.0 (upstream FR pending).
    - ``lgb_dataset_set_label`` / ``lgb_dataset_set_weight``: ``Dataset``
      exposes both mutators (true in every 4.x release).
    - ``lgb_sklearn_accepts_dataset``: ``LGBMClassifier.fit(X=Dataset)``
      short-circuits -- empirically False in 4.6.0 (upstream FR pending).

    Only the capability set produced here gates Fix 9.4.3 reuse; the
    upstream-pending items stay False until the libraries ship the
    short-circuit and a user upgrades.
    """
    caps: "Dict[str, bool]" = {}

    # CatBoost
    try:
        import catboost as _cb
        _pool_cls = getattr(_cb, "Pool", None)
        caps["cb_pool_set_label"] = callable(getattr(_pool_cls, "set_label", None))
        caps["cb_pool_set_weight"] = callable(getattr(_pool_cls, "set_weight", None))
        # Short-circuit check: CatBoostClassifier.fit(X=Pool) is supported
        # in every CB >= 1.0 via ``_build_train_pool`` (``isinstance(X,
        # Pool)`` return). The label-swap variant lands with the PR that
        # made Pool.set_label callable -- gate the reuse on BOTH.
        caps["cb_pool_label_swap"] = (
            caps["cb_pool_set_label"] and caps["cb_pool_set_weight"]
        )
    except ImportError:
        caps["cb_pool_set_label"] = False
        caps["cb_pool_set_weight"] = False
        caps["cb_pool_label_swap"] = False

    # XGBoost
    try:
        import xgboost as _xgb
        _dm = getattr(_xgb, "DMatrix", None)
        caps["xgb_dmatrix_set_label"] = callable(getattr(_dm, "set_label", None))
        caps["xgb_dmatrix_set_weight"] = callable(getattr(_dm, "set_weight", None))
        # Upstream wrapper does NOT short-circuit yet (verified 2026-04-21
        # on xgboost 3.2.0 -- ``_create_dmatrix`` rebuilds unconditionally).
        # Mark False until an upstream PR lands.
        caps["xgb_sklearn_accepts_dmatrix"] = False
    except ImportError:
        caps["xgb_dmatrix_set_label"] = False
        caps["xgb_dmatrix_set_weight"] = False
        caps["xgb_sklearn_accepts_dmatrix"] = False

    # LightGBM
    try:
        import lightgbm as _lgb
        _ds = getattr(_lgb, "Dataset", None)
        caps["lgb_dataset_set_label"] = callable(getattr(_ds, "set_label", None))
        caps["lgb_dataset_set_weight"] = callable(getattr(_ds, "set_weight", None))
        # Same story as XGBoost -- verified 2026-04-21 on lightgbm 4.6.0.
        caps["lgb_sklearn_accepts_dataset"] = False
    except ImportError:
        caps["lgb_dataset_set_label"] = False
        caps["lgb_dataset_set_weight"] = False
        caps["lgb_sklearn_accepts_dataset"] = False

    return caps




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
        safely dropped -- the pipeline was fitted with them.
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
            "to proceed -- downstream errors about shape mismatches "
            "usually trace back here.",
            sorted(missing_cols),
        )

    if extra_cols:
        if verbose:
            logger.info("Dropping extra columns: %s", sorted(extra_cols))
        df = df[filter_existing(df, columns)]

    # Fix 8f (2026-04-21, v2): per-model input-schema diff reporting.
    # ``metadata['model_schemas']`` (if present) maps model_file_name to
    # ``{schema_hash, input_schema, mlframe_model, weight_name}`` -- the
    # exact realised layout each fitted model saw at training time.
    #
    # Severity rules at load time:
    #   HARD-FAIL (ValueError) on changes sklearn / CB / XGB / LGB will
    #   silently produce wrong predictions for:
    #     * removed columns that were cat/text/embedding features
    #     * role changes (cat -> text, text -> numeric, etc.)
    #     * dtype FAMILY changes (string -> numeric, numeric -> categorical)
    #   SOFT-WARN on benign differences the downstream pipeline casts
    #   transparently:
    #     * float32 <-> float64, int32 <-> int64 (width-only)
    #     * added columns (caller superset -- already filtered to the
    #       trained subset above)
    # Silent pass on old metadata files that predate Fix 8.
    model_schemas = metadata.get("model_schemas")
    if model_schemas:
        from mlframe.training.utils import compute_model_input_fingerprint, _dtype_family
        live_hash, live_schema = compute_model_input_fingerprint(
            df,
            cat_features=metadata.get("cat_features") or [],
            text_features=metadata.get("text_features") or [],
            embedding_features=metadata.get("embedding_features") or [],
        )
        live_schema_idx = {entry["name"]: entry for entry in live_schema}
        for model_file_name, rec in model_schemas.items():
            expected_hash = rec.get("schema_hash")
            expected_schema = rec.get("input_schema") or []
            if expected_hash is None or not expected_schema:
                continue
            if expected_hash == live_hash:
                continue
            expected_idx = {entry["name"]: entry for entry in expected_schema}
            # Classify diffs by severity.
            # Trained snapshot is POST-pipeline (what fit() actually saw);
            # live snapshot is PRE-pipeline (raw serving df). For cat / text /
            # embedding columns the dtype/role is user-declared and stable
            # across train<->serve, so family changes there ARE critical
            # (silent drift in label encoding / tokenizer vocab / etc.). For
            # numeric-role columns the pipeline internally casts and encodes
            # (OHE of object, label-encoding, float32 downcast, etc.), so
            # family changes are expected and must be soft-warned.
            critical_removed: list = []
            family_changes: list = []
            role_changes: list = []
            soft_width_changes: list = []
            soft_family_changes: list = []
            for col, e in expected_idx.items():
                if col not in live_schema_idx:
                    if e["role"] in ("cat", "text", "embedding"):
                        critical_removed.append(col)
                    continue
                l = live_schema_idx[col]
                role_critical = e["role"] in ("cat", "text", "embedding") or l["role"] in ("cat", "text", "embedding")
                if l["role"] != e["role"]:
                    if role_critical:
                        role_changes.append(f"    {col}: trained role={e['role']} serving role={l['role']}")
                if l["dtype"] != e["dtype"]:
                    ef = _dtype_family(e["dtype"])
                    lf = _dtype_family(l["dtype"])
                    if ef != lf:
                        if role_critical:
                            family_changes.append(
                                f"    {col}: trained={e['dtype']!r} ({ef}) serving={l['dtype']!r} ({lf})"
                            )
                        else:
                            soft_family_changes.append(
                                f"    {col}: trained={e['dtype']!r} ({ef}) serving={l['dtype']!r} ({lf}) (numeric role)"
                            )
                    else:
                        soft_width_changes.append(
                            f"    {col}: trained={e['dtype']!r} serving={l['dtype']!r} (same family={lf})"
                        )
            hard_fail = bool(critical_removed or family_changes or role_changes)
            if hard_fail:
                diff_lines = []
                if critical_removed:
                    diff_lines.append(
                        f"  - critical missing (cat/text/embedding): {sorted(critical_removed)}"
                    )
                if family_changes:
                    diff_lines.append("  dtype FAMILY changes (trained -> serving):")
                    diff_lines.extend(family_changes)
                if role_changes:
                    diff_lines.append("  role changes (cat/text/embedding/numeric):")
                    diff_lines.extend(role_changes)
                if soft_width_changes:
                    diff_lines.append("  (soft) dtype width-only changes:")
                    diff_lines.extend(soft_width_changes)
                raise ValueError(
                    "Model input-schema mismatch at load time for "
                    f"{model_file_name!r} "
                    f"(trained hash={expected_hash}, serving hash={live_hash}):\n"
                    + "\n".join(diff_lines) + "\n"
                    "Either restore the upstream feature pipeline that produced "
                    "the trained-time layout, or retrain the model against the "
                    "current serving frame."
                )
            if soft_width_changes or soft_family_changes:
                lines = []
                if soft_width_changes:
                    lines.extend(s.strip() for s in soft_width_changes)
                if soft_family_changes:
                    lines.extend(s.strip() for s in soft_family_changes)
                logger.warning(
                    "Input-schema drift for %s (pipeline-internal casts on "
                    "numeric-role columns and/or width-only changes). "
                    "Accepting; trained pipeline is responsible for "
                    "casting the serving df: %s",
                    model_file_name,
                    "; ".join(lines),
                )

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
    to the opaque ``TypeError: No matching signature found`` -- with no
    hint about which column or why.

    Production incident 2026-04-19: the orchestration in
    ``train_mlframe_models_suite`` short-circuited to a *stale* pre-promotion
    cat_features list that still contained 4 columns which had been
    cast ``pl.Categorical -> pl.String`` for the text-features fastpath.
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
    ``cat_features`` argument -- prior versions did, which created a latent
    repeat-call state-leak trap whenever a caller reused the same list.

    Args:
        df: Training DataFrame (Polars or pandas).
        feature_types_config: FeatureTypesConfig with user overrides and threshold.
        cat_features: Already-detected categorical features (from pipeline).
            Read-only. Used only to decide which auto-detected text columns
            were "promoted" from cat_features vs newly discovered.
        verbose: Whether to log detections.

    Returns:
        (text_features, embedding_features) -- lists of column names.
    """
    import polars as pl

    text_features = list(feature_types_config.text_features or [])
    embedding_features = list(feature_types_config.embedding_features or [])
    # Auto-detected high-cardinality text-like columns that the caller
    # should DROP entirely from the training df when ``use_text_features=
    # False``. Semantic:
    #   ``use_text_features=True``  -> auto-detected cols go into
    #     ``text_features`` (CB uses them; XGB/LGB drop them via
    #     ``supports_text_features=False`` mechanism).
    #   ``use_text_features=False`` -> auto-detected cols go into THIS
    #     list so the caller can drop them from the df entirely (so
    #     no model -- including CB -- tries to consume them as a 2M-level
    #     categorical, which otherwise OOMs XGB's QuantileDMatrix and
    #     balloons CB's model artefact).
    # Regardless of the flag, if the user explicitly listed a column in
    # ``feature_types_config.text_features``/``embedding_features``, that
    # column is honored (not touched here).
    auto_detected_high_card_to_drop: list = []

    # Master switch ``use_text_features`` only gates AUTO-PROMOTION (the
    # cardinality-based heuristic below). User-supplied explicit
    # ``text_features`` list is honored regardless -- if the user passed it,
    # they intend those columns routed as text_features. (2026-04-21
    # refinement: earlier version cleared the explicit list too, which
    # broke ``test_non_catboost_drops_text_columns`` / auto-detection
    # tests that pass an explicit list AND expect no auto-promotion on
    # top. The ``promote_text`` flag below is the real gate.)

    if not feature_types_config.auto_detect_feature_types:
        return text_features, embedding_features, auto_detected_high_card_to_drop

    # Defensive: callers sometimes pass ``cat_features=None`` (e.g. after a
    # model skipped categorical detection). Treat as empty list -- the
    # ``if name in cat_features`` checks below would otherwise crash with
    # ``TypeError: argument of type 'NoneType' is not iterable``.
    if cat_features is None:
        cat_features = []

    threshold = feature_types_config.cat_text_cardinality_threshold
    # Minimum non-null FRACTION required to promote a string/categorical
    # column to text_features. CatBoost's text feature estimator builds
    # a TF-IDF vocabulary from the column's non-null content; with too
    # few non-null samples the ``occurrence_lower_bound`` filter prunes
    # everything and the estimator raises
    #   ``catboost/.../text_feature_estimators.cpp:89:
    #     Dictionary size is 0, check out data or try to decrease
    #     occurrence_lower_bound parameter``
    # (observed 2026-04-19 on ``_raw_countries`` and ``job_post_source``
    # in prod -- both ``n_unique > 50`` but >99.9% null, yielding a
    # handful of non-null strings total and an empty dictionary after
    # occurrence filtering).
    #
    # Using a FRACTION (not absolute count) so the guard scales with
    # dataset size: a 50-row test DF with 50 non-null rows (fraction 1.0)
    # passes, while an 810k-row prod DF with 6 non-null (fraction 7e-6)
    # fails. Default 0.01 = 1%: anything below that in a typical
    # many-hundred-row+ frame is a sparse column that CB's TF-IDF
    # cannot build a vocabulary from.
    min_non_null_frac = getattr(
        feature_types_config, "min_non_null_fraction_for_text_promotion", 0.01
    )
    # Total row count -- denominator for the fraction. For pandas this
    # is len(df); for polars, df.height.
    total_rows = df.height if hasattr(df, "height") else len(df)
    # Translate the fraction back to an absolute count for the guard --
    # avoids per-column float division inside the loop and reuses the
    # same floor for every column on this DF.
    min_non_null_abs = max(1, int(round(total_rows * min_non_null_frac)))
    user_assigned = set(text_features) | set(embedding_features)
    promoted: list = []  # cat_features -> text_features, tracked for diagnostic log only
    cardinalities: dict = {}  # per auto-detected text col: n_unique (for diagnostic log)
    skipped_low_non_null: list = []  # (name, n_unique, non_null_count) -- blocked by guard
    # Master-switch short-circuits the text-promotion branches (embedding
    # detection still runs). Cheaper than threading the flag through every
    # append site; one flag read per schema iteration at worst.
    promote_text = feature_types_config.use_text_features
    # 2026-04-21 ``honor_user_dtype``: when True, pre-cast categorical
    # dtypes (pl.Categorical / pl.Enum / pandas ``category``) are treated
    # as user-declared and exempt from auto-promotion. Raw pl.String /
    # pl.Utf8 / pandas object/string columns remain promotion candidates.
    honor_user_dtype = getattr(feature_types_config, "honor_user_dtype", False)
    honored_user_dtype_cols: list = []  # for diagnostic log

    if isinstance(df, pl.DataFrame):
        for name, dtype in df.schema.items():
            if name in user_assigned:
                continue
            # Embedding: pl.List(pl.Float32/Float64)
            if dtype == pl.List(pl.Float32) or dtype == pl.List(pl.Float64):
                if name not in cat_features:
                    embedding_features.append(name)
                continue
            # String/Categorical/Enum -- evaluate cardinality to split cat vs text.
            # pl.Enum is a fixed-domain categorical; it has an instance-level
            # dtype object (not a class), so it doesn't compare equal to the
            # class-level check above. Use isinstance() for Enum specifically.
            is_text_like = (
                dtype in (pl.String, pl.Utf8, pl.Categorical)
                or isinstance(dtype, pl.Enum)
            )
            # honor_user_dtype: skip promotion for already-categorical
            # dtypes; only raw Utf8/String remain auto-promotion candidates.
            is_user_categorical_dtype = (
                dtype == pl.Categorical or isinstance(dtype, pl.Enum)
            )
            if honor_user_dtype and is_user_categorical_dtype:
                honored_user_dtype_cols.append(name)
                continue
            if is_text_like:
                n_unique = df[name].n_unique()
                if n_unique > threshold:
                    # Non-null FRACTION guard -- block promotion/drop if
                    # the column is sparse relative to total rows. For
                    # the PROMOTE path this keeps CB's text estimator
                    # from producing an empty TF-IDF dictionary; for the
                    # DROP path (``use_text_features=False``) the
                    # sparseness check is still a useful signal that
                    # the column is unlikely to materially help any
                    # model anyway -- callers handle it identically.
                    non_null = int(df[name].count())
                    if non_null < min_non_null_abs:
                        skipped_low_non_null.append((name, n_unique, non_null))
                        continue
                    cardinalities[name] = n_unique
                    if promote_text:
                        text_features.append(name)
                        if name in cat_features:
                            promoted.append(name)
                    else:
                        # ``use_text_features=False``: caller MUST drop
                        # this column. Leaving it as cat_feature crashes
                        # XGB (QuantileDMatrix OOM on 2M-level cats) and
                        # balloons CB artefact size with a useless
                        # nominal-encoding vocabulary.
                        auto_detected_high_card_to_drop.append(name)
    else:
        # pandas: only detect high-cardinality text (no reliable embedding auto-detect)
        for col in df.columns:
            if col in user_assigned:
                continue
            dtype_name = str(df[col].dtype)
            # honor_user_dtype symmetry with the polars branch: ``category``
            # dtype is a user-declared categorical in pandas land; skip
            # promotion when the flag is on. ``object`` / ``string`` stay
            # as auto-promotion candidates.
            if honor_user_dtype and dtype_name == "category":
                honored_user_dtype_cols.append(col)
                continue
            if dtype_name.startswith("object") or dtype_name.startswith("string") or dtype_name == "category":
                n_unique = df[col].nunique()
                if n_unique > threshold:
                    non_null = int(df[col].notna().sum())
                    if non_null < min_non_null_abs:
                        skipped_low_non_null.append((col, n_unique, non_null))
                        continue
                    cardinalities[col] = n_unique
                    if promote_text:
                        text_features.append(col)
                        if col in cat_features:
                            promoted.append(col)
                    else:
                        # ``use_text_features=False``: drop-list for caller.
                        auto_detected_high_card_to_drop.append(col)

    # Historical note: this function used to mutate ``cat_features`` in place
    # (calling ``.remove(name)`` for each promoted column). The in-place removal
    # was redundant -- the actual caller filter lives at the call site, where
    # ``effective_cat_features`` is built via set-difference against
    # ``text_features``. We removed the mutation so repeat calls with a shared
    # list don't corrupt the caller's state.

    def _fmt_with_cardinality(names):
        """'col1:500, col2:12_345' -- makes it obvious *why* a column was
        promoted vs the configured threshold."""
        parts = []
        for n in names:
            nu = cardinalities.get(n)
            parts.append(f"{n}:{nu:_}" if nu is not None else n)
        return "[" + ", ".join(parts) + "]"

    if verbose and (text_features or embedding_features or promoted):
        if promoted:
            logger.info(
                "  Promoted %d high-cardinality column(s) from cat_features to text_features "
                "(threshold>%s): %s",
                len(promoted), threshold, _fmt_with_cardinality(promoted),
            )
        logger.info(
            "  Auto-detected feature types -- text: %s, embedding: %s",
            _fmt_with_cardinality(text_features) if text_features else "(none)",
            embedding_features or "(none)",
        )

    # Load-bearing: log the drop-list regardless of verbose. Operator
    # needs to see WHICH columns were auto-dropped and WHY -- a silent
    # drop is exactly the class of bug we just fixed (2026-04-22):
    # skills_text at 2M unique silently stayed as cat_feature under
    # ``use_text_features=False`` and OOM'd XGB on a prod 9M-row run.
    if auto_detected_high_card_to_drop:
        logger.warning(
            "  use_text_features=False: auto-dropping %d high-cardinality "
            "text-like column(s) (n_unique > %d) to prevent "
            "XGB QuantileDMatrix OOM / CB model-artefact bloat: %s. "
            "To keep these columns, set use_text_features=True (routes "
            "them to text_features -- CB uses them, XGB/LGB drop them) "
            "or add them explicitly to feature_types_config.text_features.",
            len(auto_detected_high_card_to_drop),
            threshold,
            _fmt_with_cardinality(auto_detected_high_card_to_drop),
        )

    # Always log the skipped-by-non-null-guard set, even at verbose=False
    # -- this is a load-bearing diagnostic: columns that would otherwise
    # have been promoted and crashed CatBoost with "Dictionary size is 0"
    # are silently kept as cat_features. The operator needs to know so
    # they can either (a) fix the upstream feature-extraction to produce
    # more non-null samples, or (b) accept the lower-quality cat usage
    # of these columns.
    if skipped_low_non_null:
        formatted = ", ".join(
            f"{name}:{n_unique:_} (non_null={nn:_}/{total_rows:_})"
            for name, n_unique, nn in skipped_low_non_null
        )
        logger.warning(
            "  Auto-detection: %d column(s) had n_unique>%d (would be "
            "promoted to text_features) but non_null<%d (%.1f%% of %d rows, "
            "below the %.2f%% floor) -- kept as cat_features to avoid "
            "CatBoost's 'Dictionary size is 0' error on sparse text "
            "columns: %s",
            len(skipped_low_non_null), threshold, min_non_null_abs,
            min_non_null_frac * 100, total_rows, min_non_null_frac * 100,
            formatted,
        )
    if honored_user_dtype_cols and verbose:
        logger.info(
            "  honor_user_dtype=True: %d column(s) with explicit categorical "
            "dtype (pl.Categorical / pl.Enum / pandas category) kept out of "
            "text-auto-promotion regardless of cardinality: %s",
            len(honored_user_dtype_cols), sorted(honored_user_dtype_cols),
        )

    return text_features, embedding_features, auto_detected_high_card_to_drop




def _validate_feature_type_exclusivity(
    text_features: list,
    embedding_features: list,
    cat_features: list,
) -> None:
    """Raise ValueError if any column appears in multiple feature type lists.

    Each argument may be ``None`` (treated as empty) -- callers that skipped
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

    # Cache key = (tier, input-container-kind). Without the kind component the
    # cache collides between Polars and pandas inputs: in a multi-model suite
    # where Linear (non-polars-native) runs first it stashes pandas tier-DFs
    # under tier=(False,False); XGB (polars-native) later asks for the same
    # tier and gets pandas back, then XGBoostStrategy.prepare_polars_dataframe
    # raises "'DataFrame' object has no attribute 'schema'" on strategies.py:323.
    # The "kind" tag is sampled from the first non-None input, matching what
    # the caller actually passed in this invocation.
    kind = "none"
    for k in ("train_df", "val_df", "test_df"):
        v = base_dfs.get(k)
        if v is not None:
            kind = "pl" if isinstance(v, pl.DataFrame) else "pd"
            break
    tier_key = (strategy.feature_tier(), kind)
    tier = tier_key  # preserved name for downstream logging + storage
    if tier_key in tier_cache:
        return tier_cache[tier_key]

    # Determine columns to exclude for this tier
    cols_to_exclude = set()
    if text_features and not strategy.supports_text_features:
        cols_to_exclude.update(text_features)
    if embedding_features and not strategy.supports_embedding_features:
        cols_to_exclude.update(embedding_features)

    if not cols_to_exclude:
        # Tier supports all features -- use base DFs directly (no copy)
        tier_dfs = base_dfs
    else:
        if verbose:
            logger.info("  Tier %s: dropping %d text/embedding columns: %s", tier, len(cols_to_exclude), sorted(cols_to_exclude))
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
from pyutilz.system import clean_ram, tqdmu, tqdmu_lazy_start
from pyutilz.strings import slugify
from sklearn.base import clone
from sklearn.pipeline import Pipeline
import category_encoders as ce

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from pyutilz.system import ensure_dir_exists

from ..configs import (
    PreprocessingConfig,
    TrainingSplitConfig,
    PreprocessingBackendConfig,
    FeatureTypesConfig,
    FeatureSelectionConfig,
    ModelHyperparamsConfig,
    TrainingBehaviorConfig,
    TrainingConfig,
    TargetTypes,
    LinearModelConfig,
    PreprocessingExtensionsConfig,
    MultilabelDispatchConfig,
    ReportingConfig,
    OutputConfig,
    OutlierDetectionConfig,
    ConfidenceAnalysisConfig,
    BaselineDiagnosticsConfig,
    CompositeTargetDiscoveryConfig,
    DummyBaselinesConfig,
    QuantileRegressionConfig,
)
from ..preprocessing import (
    load_and_prepare_dataframe,
    preprocess_dataframe,
    save_split_artifacts,
    create_split_dataframes,
)
from ..pipeline import fit_and_transform_pipeline, prepare_df_for_catboost, apply_preprocessing_extensions
from mlframe.feature_selection.filters import MRMR
from ..utils import (
    log_ram_usage,
    log_phase,
    drop_columns_from_dataframe,
    get_pandas_view_of_polars_df,
    estimate_df_size_mb,
    get_process_rss_mb,
    maybe_clean_ram_and_gpu,
    filter_existing,
    compute_model_input_fingerprint,
)
from ..helpers import get_trainset_features_stats_polars, get_trainset_features_stats
from ..models import is_linear_model, LINEAR_MODEL_TYPES
from ..strategies import get_strategy, get_polars_cat_columns, PipelineCache
from ..io import load_mlframe_model
from ..splitting import make_train_test_split
from ..phases import phase, reset_phase_registry, format_phase_summary

# Extractors from new module
from ..extractors import FeaturesAndTargetsExtractor

# score_ensemble is in ensembling module
from ...ensembling import score_ensemble

# Training execution functions from train_eval module
from ..train_eval import process_model, select_target
from ..drift_report import compute_label_distribution_drift, format_drift_report
from ..baseline_diagnostics import BaselineDiagnostics, format_baseline_diagnostics_report
from ..composite import CompositeTargetDiscovery

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
            fe_max_steps=0,
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


