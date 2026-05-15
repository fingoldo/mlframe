"""Phase helper functions for the training suite."""

from __future__ import annotations

import logging
import os
from timeit import default_timer as timer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..phases import phase
import polars as pl

if TYPE_CHECKING:
    from ._training_context import TrainingContext

from sklearn.pipeline import Pipeline

from ._misc_helpers import (
    _auto_detect_feature_types, _cfg_get, _df_shape_str, _drop_cols_df,
    _elapsed_str, _validate_feature_type_exclusivity,
)

from ..configs import PreprocessingExtensionsConfig, TargetTypes
from ..preprocessing import (
    create_split_dataframes, load_and_prepare_dataframe, preprocess_dataframe,
    save_split_artifacts,
)
from ..utils import (
    drop_columns_from_dataframe, estimate_df_size_mb,
    get_process_rss_mb, log_phase, log_ram_usage, maybe_clean_ram_and_gpu,
)
from ..strategies import get_strategy, get_polars_cat_columns
from ..splitting import make_train_test_split
from ..pipeline import apply_preprocessing_extensions, fit_and_transform_pipeline, prepare_df_for_catboost
from ._setup_helpers import _apply_outlier_detection_global, _compute_fairness_subgroups, _convert_dfs_to_pandas

logger = logging.getLogger(__name__)

_DRIFT_SKIP_CARD = 100_000
_DRIFT_MIN_ABS = 5
_DRIFT_MIN_FRAC = 0.05
_DEFAULT_TEST_SIZE = 0.15
_DEFAULT_VAL_SIZE = 0.15
_DEFAULT_LTR_ITER = 200
_DEFAULT_LTR_LR = 0.1
_DEFAULT_LTR_ES = 30

def _apply_plot_style_overrides(
    *,
    matplotlib_style=None,
    matplotlib_rcparams=None,
    plotly_template=None,
    verbose: bool = False,
) -> None:
    """Apply matplotlib/plotly style overrides process-wide.

    Overrides are not reverted on suite exit. Failures log at WARNING and don't abort.
    """
    if (matplotlib_style is None and not matplotlib_rcparams
            and plotly_template is None):
        return

    if matplotlib_style is not None or matplotlib_rcparams:
        try:
            import matplotlib.pyplot as plt
        except Exception as _imp_err:
            logger.warning(
                "[plot_style] matplotlib import failed (%s); matplotlib "
                "style override skipped.", _imp_err,
            )
            plt = None
        if plt is not None:
            if matplotlib_style is not None:
                try:
                    plt.style.use(matplotlib_style)
                    if verbose:
                        logger.info(
                            "[plot_style] matplotlib style: %r", matplotlib_style,
                        )
                except Exception as _style_err:
                    logger.warning(
                        "[plot_style] plt.style.use(%r) failed: %s. "
                        "Continuing with the current matplotlib style.",
                        matplotlib_style, _style_err,
                    )
            if matplotlib_rcparams:
                try:
                    plt.rcParams.update(dict(matplotlib_rcparams))
                    if verbose:
                        logger.info(
                            "[plot_style] applied %d matplotlib rcParams "
                            "override(s): %s",
                            len(matplotlib_rcparams),
                            sorted(matplotlib_rcparams.keys()),
                        )
                except Exception as _rc_err:
                    logger.warning(
                        "[plot_style] plt.rcParams.update(%r) failed: %s. "
                        "Some matplotlib keys may not have been applied.",
                        matplotlib_rcparams, _rc_err,
                    )

    if plotly_template is not None:
        try:
            import plotly.io as pio
        except Exception as _imp_err:
            logger.warning(
                "[plot_style] plotly import failed (%s); plotly template "
                "override skipped.", _imp_err,
            )
            return
        try:
            pio.templates.default = plotly_template
            if verbose:
                logger.info(
                    "[plot_style] plotly template: %r", plotly_template,
                )
        except Exception as _tpl_err:
            logger.warning(
                "[plot_style] plotly templates.default = %r failed: %s. "
                "Continuing with the current plotly template.",
                plotly_template, _tpl_err,
            )


def _defensive_copy_and_expand_multilabel_regression(
    *,
    target_by_type,
    composite_target_discovery_config,
    metadata,
):
    """Defensive copy of ``target_by_type`` with optional 2-D regression target expansion into 1-D sub-targets."""
    new_target_by_type = {
        tt: dict(named) if isinstance(named, dict) else named
        for tt, named in target_by_type.items()
    }
    ml_strategy = str(getattr(
        composite_target_discovery_config,
        "multilabel_strategy", "per_target",
    ))
    if ml_strategy == "per_target":
        expanded = dict(new_target_by_type[TargetTypes.REGRESSION])
        ml_expanded_map: dict[str, list[str]] = {}
        for _tn, _tv in list(new_target_by_type[TargetTypes.REGRESSION].items()):
            _arr = np.asarray(_tv)
            if _arr.ndim == 2 and _arr.shape[1] >= 1:
                sub_names = []
                for _j in range(_arr.shape[1]):
                    _sub_name = f"{_tn}_out{_j}"
                    expanded[_sub_name] = _arr[:, _j]
                    sub_names.append(_sub_name)
                expanded.pop(_tn, None)
                ml_expanded_map[_tn] = sub_names
                logger.info(
                    "[CompositeTargetDiscovery] multilabel target '%s' (shape=%s) expanded into %d 1-D sub-targets: %s",
                    _tn, _arr.shape, _arr.shape[1], sub_names,
                )
        new_target_by_type[TargetTypes.REGRESSION] = expanded
        if ml_expanded_map:
            metadata.setdefault("multilabel_target_expansion", {})[
                str(TargetTypes.REGRESSION)
            ] = ml_expanded_map
    return new_target_by_type


def _init_composite_discovery_metadata(
    *,
    composite_target_discovery_config,
    target_by_type,
    mlframe_models,
    metadata,
):
    """Composite-target discovery prologue: init metadata buckets, snapshot env signature, record skip-reasons for non-regression target types.

    Mutates ``metadata`` in-place. Returns ``(gpu_families, kept_spec_total=0)``.
    """
    metadata["composite_target_specs"] = {}
    metadata["composite_target_failures"] = {}
    metadata["composite_target_filter_drops"] = {}

    gpu_families: list[str] = []
    if composite_target_discovery_config.enabled:
        from ..composite import env_signature as _env_sig, detect_gpu_in_use as _detect_gpu
        metadata["composite_target_env_signature"] = _env_sig()
        gpu_families = _detect_gpu(mlframe_models or [])

    for _tt_skip, _named_skip in target_by_type.items():
        if not isinstance(_named_skip, dict):
            continue
        if _tt_skip == TargetTypes.REGRESSION:
            continue
        reason = None
        if _tt_skip == TargetTypes.LEARNING_TO_RANK:
            reason = "ltr_unsupported_pairwise_breaks_with_residual"
        elif _tt_skip == TargetTypes.MULTICLASS_CLASSIFICATION:
            reason = "multiclass_unsupported_no_residual_semantics"
        elif _tt_skip == TargetTypes.MULTILABEL_CLASSIFICATION:
            reason = "multilabel_classification_unsupported"
        elif _tt_skip == TargetTypes.QUANTILE_REGRESSION:
            reason = "quantile_regression_unsupported_per_quantile_inverse_undefined"
        elif _tt_skip == TargetTypes.BINARY_CLASSIFICATION:
            reason = "binary_classification_unsupported_init_score_logit_offset"
        if reason is not None:
            for _tn_skip in _named_skip:
                metadata["composite_target_failures"].setdefault(
                    str(_tt_skip), {})[_tn_skip] = [{
                        "name": _tn_skip, "kept": False, "rejected": True,
                        "reason": reason,
                    }]
    return gpu_families, 0


def _phase_global_outlier_detection(ctx: TrainingContext) -> None:
    """Global outlier detection before the model loops.

    Flattens ``target_by_type`` for the OD class-balance pre-check (so a detector that would
    eliminate the entire minority class gets rejected upfront), applies the OD masks to both
    pandas and Polars frames so the fastpath stays aligned.

    Mutates ``ctx`` in place: writes filtered_train_df, filtered_val_df, filtered_train_idx,
    filtered_val_idx, train_od_idx, val_od_idx, outlier_detection_result, train_df_polars, val_df_polars.
    """
    train_df_pd = ctx.train_df_pd
    val_df_pd = ctx.val_df_pd
    train_df_polars = ctx.train_df_polars
    val_df_polars = ctx.val_df_polars
    outlier_detector = ctx.outlier_detector
    verbose = ctx.verbose

    _targets_flat_for_classbalance = {}
    for _tt, _named in ctx.target_by_type.items():
        if isinstance(_named, dict):
            for _tn, _tv in _named.items():
                _targets_flat_for_classbalance[f"{_tt}/{_tn}"] = _tv
    (filtered_train_df, filtered_val_df, filtered_train_idx, filtered_val_idx,
     train_od_idx, val_od_idx) = _apply_outlier_detection_global(
        train_df=train_df_pd,
        val_df=val_df_pd,
        train_idx=ctx.train_idx,
        val_idx=ctx.val_idx,
        outlier_detector=outlier_detector,
        od_val_set=ctx.od_val_set,
        verbose=verbose,
        baseline_rss_mb=ctx.baseline_rss_mb,
        df_size_mb=ctx.df_size_mb,
        targets_for_classbalance=_targets_flat_for_classbalance or None,
    )

    outlier_detection_result = {
        "train_od_idx": train_od_idx,
        "val_od_idx": val_od_idx,
    }

    if outlier_detector is not None:
        n_train_pre_od = len(train_df_pd) if train_df_pd is not None else None
        n_val_pre_od = len(val_df_pd) if val_df_pd is not None else None
        n_train_post_od = int(train_od_idx.sum()) if train_od_idx is not None else n_train_pre_od
        n_val_post_od = int(val_od_idx.sum()) if val_od_idx is not None else n_val_pre_od
        n_train_dropped = (n_train_pre_od - n_train_post_od) if n_train_pre_od is not None else 0
        n_val_dropped = (n_val_pre_od - n_val_post_od) if (n_val_pre_od is not None and val_od_idx is not None) else 0
        ctx.metadata["outlier_detection"] = {
            "applied": True,
            "n_outliers_dropped_train": int(n_train_dropped),
            "n_outliers_dropped_val": int(n_val_dropped),
            "train_size_after_od": int(n_train_post_od) if n_train_post_od is not None else None,
            "val_size_after_od": int(n_val_post_od) if n_val_post_od is not None else None,
        }
    else:
        ctx.metadata["outlier_detection"] = {"applied": False}

    # Keep Polars fastpath DFs in sync with OD-filtered targets.
    if train_od_idx is not None and train_df_polars is not None:
        train_df_polars = train_df_polars.filter(pl.Series(train_od_idx))
    if val_od_idx is not None and val_df_polars is not None:
        val_df_polars = val_df_polars.filter(pl.Series(val_od_idx))

    ctx.filtered_train_df = filtered_train_df
    ctx.filtered_val_df = filtered_val_df
    ctx.filtered_train_idx = filtered_train_idx
    ctx.filtered_val_idx = filtered_val_idx
    ctx.train_od_idx = train_od_idx
    ctx.val_od_idx = val_od_idx
    ctx.outlier_detection_result = outlier_detection_result
    ctx.train_df_polars = train_df_polars
    ctx.val_df_polars = val_df_polars


def _phase_pandas_conversion_and_cat_prep(
    *,
    train_df: pl.DataFrame | pd.DataFrame | None,
    val_df: pl.DataFrame | pd.DataFrame | None,
    test_df: pl.DataFrame | pd.DataFrame | None,
    train_df_polars_pre: pl.DataFrame | None,
    val_df_polars_pre: pl.DataFrame | None,
    test_df_polars_pre: pl.DataFrame | None,
    cat_features: list[str],
    was_polars_input: bool,
    all_models_polars_native: bool,
    needs_polars_pre_clone: bool,
    mlframe_models: list[str],
    recurrent_models: list[str],
    rfecv_models: list[str],
    baseline_rss_mb: float,
    df_size_mb: float,
    verbose: bool,
) -> tuple:
    """Pandas conversion + CatBoost cat prep + Polars release.

    Skips polars->pandas conversion entirely when all models are Polars-native (or when only
    non-native sklearn models block the fastpath - those do lazy conversion later). Captures
    Polars-side sizes BEFORE conversion to avoid pandas ``memory_usage(deep=True)`` scans.
    Releases post-pipeline Polars frames when a clone was made; Arrow-backed pandas views
    retain their own buffers.
    """
    if verbose:
        logger.info("Zero-copy conversion to pandas...")

    train_df_polars = train_df_polars_pre
    val_df_polars = val_df_polars_pre
    test_df_polars = test_df_polars_pre

    strategies_for_check = [get_strategy(m) for m in mlframe_models] if mlframe_models else []

    _has_rfecv = bool(rfecv_models)
    _has_non_native_mlframe_strategy = was_polars_input and not all_models_polars_native
    can_skip_pandas_conv = (
        was_polars_input
        and not recurrent_models and not _has_rfecv
        and (all_models_polars_native or _has_non_native_mlframe_strategy)
    )

    train_df_size_bytes_cached: float | None = None
    val_df_size_bytes_cached: float | None = None
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

    # Post-pipeline Polars release: Arrow-backed pandas views retain their own buffers.
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
    train_df: pl.DataFrame | pd.DataFrame | None,
    val_df: pl.DataFrame | pd.DataFrame | None,
    test_df: pl.DataFrame | pd.DataFrame | None,
    cat_features: list[str],
    text_features: list[str],
    embedding_features: list[str],
) -> None:
    """Pre-train cardinality + val/test drift logging (pure side-effect).

    Cardinality surfaces the input shape before any native XGB/CB crash on high-cardinality
    categoricals. Drift detection: XGB 3.x on Windows can crash silently during val
    IterativeDMatrix construction when val/test contain categories absent from train; we emit
    a WARNING with a healing suggestion keyed on train-side cardinality. Columns with
    cardinality > 100k (free-text) are skipped.
    """
    all_cat_cols = list(cat_features or []) + list(text_features or []) + list(embedding_features or [])
    if not (all_cat_cols and train_df is not None):
        return
    try:
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

                # Test-side drift is reported above but NOT used in healing decisions
                # (would leak test info into training).
                for c, card_tr, v_only, t_only in drift_rows:
                    if v_only == 0 and t_only == 0:
                        continue
                    v_frac = v_only / max(card_tr, 1)
                    if v_only >= _DRIFT_MIN_ABS or v_frac >= _DRIFT_MIN_FRAC:
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
    train_df: pl.DataFrame | pd.DataFrame | None,
    val_df: pl.DataFrame | pd.DataFrame | None,
    test_df: pl.DataFrame | pd.DataFrame | None,
    train_df_polars_pre: pl.DataFrame | None,
    val_df_polars_pre: pl.DataFrame | None,
    test_df_polars_pre: pl.DataFrame | None,
    cat_features: list[str],
    cat_features_polars: list[str],
    was_polars_input: bool,
    all_models_polars_native: bool,
    pipeline_config: Any,
    feature_types_config: Any,
    metadata: dict,
    verbose: bool,
) -> tuple:
    """Auto-detect text + embedding features, optionally drop high-card columns, validate exclusivity, one-time Polars string->Categorical cast.

    Mutates ``metadata`` in-place with ``columns`` and ``cat_features``.
    """
    # Use pre-pipeline DF so auto-detection sees original dtypes.
    detect_df = train_df_polars_pre if was_polars_input else train_df
    raw_cat_features = list(set((cat_features or []) + (cat_features_polars or [])))
    # Honor only strictly-user-declared pl.Categorical columns as already-assigned.
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

    # Capture pre-drop column data so dummy_baselines per_group_mean can use these as group
    # keys downstream (tree models drop them to avoid XGB QuantileDMatrix OOM).
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

    # One-time Polars string->Categorical cast so XGB's arrow bridge doesn't choke on large_string.
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
    train_df: pl.DataFrame | pd.DataFrame | None,
    val_df: pl.DataFrame | pd.DataFrame | None,
    test_df: pl.DataFrame | pd.DataFrame | None,
    mlframe_models: list[str],
    pipeline_config: Any,
    preprocessing_config: Any,
    feature_types_config: Any,
    preprocessing_extensions: Any,
    metadata: dict,
    verbose: bool,
) -> tuple:
    """Pipeline fitting and transformation.

    Decomposes datetime columns BEFORE the pre-pipeline clone (otherwise the cloned frames
    retain raw datetimes that crash numpy/sklearn/CB downstream), saves Polars originals for
    the fastpath, runs ``fit_and_transform_pipeline``, then applies any
    ``PreprocessingExtensionsConfig``. Mutates ``metadata`` in-place.
    """
    t0_phase3 = timer()
    if verbose:
        log_phase("PHASE 3: Pipeline Fitting & Transformation")

    was_polars_input = isinstance(train_df, pl.DataFrame)

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

    # Datetime columns must be decomposed BEFORE the pre-pipeline clone, otherwise the
    # cloned frames retain raw datetimes and reach downstream where numpy/sklearn/CB raise.
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
        text_features=feature_types_config.text_features if feature_types_config else [],
        embedding_features=feature_types_config.embedding_features if feature_types_config else [],
    )
    if verbose:
        logger.info("  fit_and_transform_pipeline done in %s", _elapsed_str(t0_fit_pipeline))

    polars_pipeline_applied = was_polars_input and pipeline_config.prefer_polarsds and pipeline is not None

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
    df: pl.DataFrame | pd.DataFrame | None,
    target_by_type: dict,
    timestamps: np.ndarray | None,
    group_ids: np.ndarray | pd.Series | None,
    group_ids_raw: np.ndarray | pd.Series | None,
    artifacts: Any,
    sequences: list[np.ndarray] | None,
    split_config: Any,
    behavior_config: Any,
    metadata: dict,
    data_dir: str,
    models_dir: str,
    target_name: str,
    model_name: str,
    df_size_mb: float,
    verbose: bool,
) -> tuple:
    """Train/val/test splitting with auto-stratification + group-aware splitting.

    Mutates ``metadata`` in-place with split sizes + per-split details.
    """
    if verbose:
        log_phase("PHASE 2: Train/Val/Test Splitting")

    t0_phase2 = timer()
    if verbose:
        logger.info(f"Making train_val_test split...")
    # Auto-stratify by target for single-target classification when no timestamps are present
    # (without stratification, rare-imbalance shuffles can produce all-class-0 val slices).
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
        # Multilabel stratification needs the optional ``iterative-stratification`` dep;
        # skip rather than force it on every user.
        if _has_multilabel:
            _stratify_y = None
        elif len(_classification_targets) == 1:
            try:
                _arr = np.asarray(_classification_targets[0])
                # Only stratify when meaningful: all classes have >=2 rows (sklearn raises
                # otherwise) and target is 1-D.
                if _arr.ndim == 1:
                    _u, _c = np.unique(_arr, return_counts=True)
                    if len(_u) >= 2 and _c.min() >= 2:
                        _stratify_y = _arr
            except Exception:
                _stratify_y = None
    # Group-aware splitting opt-in: when the extractor produced ``group_ids`` and
    # ``split_config.use_groups`` is set, route through GroupShuffleSplit.
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

    # Compute fairness subgroups from full df BEFORE splitting.
    fairness_subgroups, fairness_features = _compute_fairness_subgroups(df, behavior_config)
    if verbose:
        if fairness_features and fairness_subgroups is None:
            logger.warning(f"Fairness features {fairness_features} specified but subgroups could not be computed")
        elif fairness_subgroups is not None:
            logger.info("Computed %d fairness subgroups", len(fairness_subgroups))

    train_df, val_df, test_df = create_split_dataframes(
        df=df,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    if verbose:
        logger.info("  Split shapes -- train: %s, val: %s, test: %s", _df_shape_str(train_df), _df_shape_str(val_df), _df_shape_str(test_df))
        logger.info("  PHASE 2 total: %s", _elapsed_str(t0_phase2))

    # Split sequences by train/val/test indices (for recurrent models).
    train_sequences, val_sequences, test_sequences = None, None, None
    if sequences is not None:
        train_sequences = [sequences[i] for i in train_idx]
        val_sequences = [sequences[i] for i in val_idx] if val_idx is not None else None
        test_sequences = [sequences[i] for i in test_idx]
        if verbose:
            logger.info("Split sequences: train=%d, val=%d, test=%d", len(train_sequences), len(val_sequences) if val_sequences else 0, len(test_sequences))

    if verbose:
        logger.info("Deleting original DataFrame to free RAM...")

    # Refresh baseline so the next maybe_clean_ram_and_gpu in the caller sees the post-del state.
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
    ctx: TrainingContext,
    *,
    features_and_targets_extractor: Any,
) -> None:
    """Data loading + features-and-targets extraction + preprocessing.

    Captures the RAM baseline + DF-size estimate AFTER the FTE so the downstream
    ``maybe_clean_ram_and_gpu`` calls have a meaningful pre-transient-allocation reference.
    Mutates ``ctx`` in place: writes df, target_by_type, group_ids_raw, group_ids, timestamps,
    artifacts, additional_columns_to_drop, sample_weights, baseline_rss_mb, df_size_mb, sequences.
    """
    verbose = ctx.verbose
    if verbose:
        log_phase("PHASE 1: Data Loading & Preprocessing")

    t0_phase1 = timer()
    with phase("load_and_prepare_dataframe"):
        df = load_and_prepare_dataframe(ctx.df, ctx.preprocessing_config, verbose=verbose)
    if verbose:
        logger.info("  load_and_prepare_dataframe done -- %s, %s", _df_shape_str(df), _elapsed_str(t0_phase1))

    if verbose:
        logger.info("Create additional features & extracting targets...")

    t0_fte = timer()
    df, target_by_type, group_ids_raw, group_ids, timestamps, artifacts, additional_columns_to_drop, sample_weights = features_and_targets_extractor.transform(
        df
    )
    if verbose:
        logger.info("  features_and_targets_extractor done -- %s, %s", _df_shape_str(df), _elapsed_str(t0_fte))

    # Capture baseline RSS + DF size BEFORE downstream transient allocations
    # (get_sequences, drop_columns, preprocess) so maybe_clean_ram_and_gpu has a stable ref.
    baseline_rss_mb = get_process_rss_mb()
    df_size_mb = estimate_df_size_mb(df)

    sequences = ctx.sequences
    if ctx.recurrent_models and sequences is None:
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

    # Drop columns AFTER the extractor: it may consume or create columns.
    df = drop_columns_from_dataframe(
        df,
        additional_columns_to_drop=additional_columns_to_drop,
        config_drop_columns=ctx.preprocessing_config.drop_columns,
        verbose=verbose,
    )

    t0_preproc = timer()
    df = preprocess_dataframe(df, ctx.preprocessing_config, verbose=verbose)
    if verbose:
        logger.info("  preprocess_dataframe done -- %s, %s", _df_shape_str(df), _elapsed_str(t0_preproc))
        logger.info("  PHASE 1 total: %s", _elapsed_str(t0_phase1))

    ctx.df = df
    ctx.target_by_type = target_by_type
    ctx.group_ids_raw = group_ids_raw
    ctx.group_ids = group_ids
    ctx.timestamps = timestamps
    ctx.artifacts = artifacts
    ctx.additional_columns_to_drop = additional_columns_to_drop
    ctx.sample_weights = sample_weights
    ctx.baseline_rss_mb = baseline_rss_mb
    ctx.df_size_mb = df_size_mb
    ctx.sequences = sequences


def _build_suite_common_params_dict(
    *,
    reporting_config,
    preprocessing_config,
    confidence_analysis_config,
) -> dict[str, Any]:
    """Assemble the ``common_params_dict`` carried down through the suite."""
    common: dict[str, Any] = {}
    common.update(
        reporting_config.model_dump(exclude={
            "title_metrics_tokens",
            "plot_inline_display",
            # matplotlib/plotly style overrides are consumed at suite-level only via
            # _apply_plot_style_overrides; _build_configs_from_params doesn't accept them.
            "matplotlib_style",
            "matplotlib_rcparams",
            "plotly_template",
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
    ctx: TrainingContext,
    *,
    target_type: Any,
    df: pl.DataFrame | pd.DataFrame | None,
    features_and_targets_extractor: Any,
) -> tuple | None:
    """If ``target_type == LEARNING_TO_RANK``, route to ``train_mlframe_ranker_suite``.

    Returns ``None`` for non-LTR call sites (caller continues with the standard pipeline);
    the ranker-suite return tuple otherwise.
    """
    if target_type is None or target_type != TargetTypes.LEARNING_TO_RANK:
        return None
    from mlframe.training.ranker_suite import train_mlframe_ranker_suite

    _save_dir = None
    _data_dir = _cfg_get(ctx.output_config, "data_dir")
    _models_dir = _cfg_get(ctx.output_config, "models_dir") or "models"
    if _data_dir:
        _save_dir = os.path.join(_data_dir, _models_dir, ctx.model_name)

    _test_size = _cfg_get(ctx.split_config, "test_size", _DEFAULT_TEST_SIZE)
    _val_size = _cfg_get(ctx.split_config, "val_size", _DEFAULT_VAL_SIZE)

    _iter = _cfg_get(ctx.hyperparams_config, "iterations", _DEFAULT_LTR_ITER)
    _lr = _cfg_get(ctx.hyperparams_config, "learning_rate", _DEFAULT_LTR_LR)
    _es = _cfg_get(ctx.hyperparams_config, "early_stopping_rounds", _DEFAULT_LTR_ES)
    # Forwarded to MLPRanker.__init__ when LTR + 'mlp' is in mlframe_models.
    _mlp_kwargs = _cfg_get(ctx.hyperparams_config, "mlp_kwargs")

    _plot_outputs = _cfg_get(ctx.reporting_config, "plot_outputs")
    _ltr_panels = _cfg_get(ctx.reporting_config, "ltr_panels")
    _plot_file = _cfg_get(ctx.output_config, "plot_file")

    return train_mlframe_ranker_suite(
        df=df,
        target_name=ctx.target_name,
        model_name=ctx.model_name,
        features_and_targets_extractor=features_and_targets_extractor,
        mlframe_models=ctx.mlframe_models,
        use_mlframe_ensembles=ctx.use_mlframe_ensembles,
        ranking_config=ctx.ranking_config,
        test_size=_test_size,
        val_size=_val_size,
        iterations=_iter,
        learning_rate=_lr,
        early_stopping_rounds=_es,
        save_dir=_save_dir,
        verbose=ctx.verbose,
        plot_file=_plot_file,
        plot_outputs=_plot_outputs,
        ltr_panels=_ltr_panels,
        mlp_kwargs=_mlp_kwargs,
    )
