"""Phase helper functions for the training suite."""

from __future__ import annotations

import logging
import os
from timeit import default_timer as timer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..phases import phase
import polars as pl

if TYPE_CHECKING:
    from ._training_context import TrainingContext

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
from ..pipeline import apply_preprocessing_extensions, fit_and_transform_pipeline, prepare_dfs_for_catboost_joint
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


# Backward-compatible named-tuple wrappers for the large positional return shapes that used to be
# bare tuples (H-CORE-19/20/21). NamedTuple stays iterable + indexable so existing
# ``(a, b, ...) = func()`` and ``func()[i]`` callers keep working; new callers can read ``.field``
# names for clarity and any future field addition does not silently shift existing positions.
class TrainValTestSplitResult(NamedTuple):
    """Return shape for ``_phase_train_val_test_split``."""
    train_idx: Any
    val_idx: Any
    test_idx: Any
    train_details: Any
    val_details: Any
    test_details: Any
    train_df: Any
    val_df: Any
    test_df: Any
    fairness_subgroups: Any
    fairness_features: Any
    train_sequences: Any
    val_sequences: Any
    test_sequences: Any
    baseline_rss_mb: Any


class FitPipelineResult(NamedTuple):
    """Return shape for ``_phase_fit_pipeline``."""
    train_df: Any
    val_df: Any
    test_df: Any
    pipeline: Any
    extensions_pipeline: Any
    cat_features: Any
    cat_features_polars: Any
    was_polars_input: Any
    all_models_polars_native: Any
    polars_pipeline_applied: Any
    train_df_polars_pre: Any
    val_df_polars_pre: Any
    test_df_polars_pre: Any
    pipeline_config: Any
    preprocessing_extensions: Any
    train_df_pandas_pre_meta: Any


class PolarsCategoricalFixesResult(NamedTuple):
    """Return shape for ``apply_polars_categorical_fixes`` (H-CORE-20)."""
    train_df_polars: Any
    val_df_polars: Any
    test_df_polars: Any
    train_df_pd: Any
    val_df_pd: Any
    test_df_pd: Any
    filtered_train_df: Any
    filtered_val_df: Any

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
    """Defensive copy of ``target_by_type`` with optional 2-D regression target expansion into 1-D sub-targets.

    MUTATION SIDE EFFECT: when multilabel expansion fires (config's
    ``multilabel_strategy == "per_target"`` and any regression target is 2-D),
    this function ALSO writes the expansion map to
    ``metadata["multilabel_target_expansion"][str(TargetTypes.REGRESSION)]``
    so the downstream report / save can name the sub-targets. The "defensive_copy"
    in the function name signals the deep-copy of ``target_by_type``; the
    metadata write is an additional side effect the name does NOT advertise.

    Renaming would touch 5 import sites including CHANGELOG; documenting the
    side effect here is the bounded fix. Future refactor candidate:
    ``_expand_multilabel_regression_and_record(...)``.
    """
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
    polars_pipeline_applied: bool = True,
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

    # Wave 69 (2026-05-20) closure: defer_pandas_conv heuristic landed in wave 4
    # of the F6 audit (predict-skew closure); it consults ctx.strategy_by_model
    # directly rather than rebuilding the per-model strategy list here. The
    # on-demand build inside the verbose-only log branches below is intentional
    # -- it avoids a get_strategy() pass on every call when verbose=False.
    _has_rfecv = bool(rfecv_models)
    # The earlier disjunction ``all_models_polars_native OR _has_non_native_mlframe_strategy`` was always True
    # given ``was_polars_input`` (the second disjunct collapses to ``was_polars_input AND NOT first``), so
    # the whole conjunction reduces to the three real gates: had a polars input, no recurrent model, no RFECV.
    # ``polars_pipeline_applied`` captures whether a polars-aware pipeline actually fitted on the polars frame;
    # when False the downstream pipeline state lives only in pandas representation, so the lazy-pandas fastpath
    # cannot keep frames as polars without losing that state.
    defer_pandas_conv = (
        was_polars_input
        and polars_pipeline_applied
        and not recurrent_models
        and not _has_rfecv
    )

    train_df_size_bytes_cached: float | None = None
    val_df_size_bytes_cached: float | None = None
    # CACHE-P1-7: cat-heavy frames inflate roughly 1.4-1.6x when going
    # polars->pandas (string columns gain per-row object overhead, category
    # columns lose dictionary compression). Apply a 1.5x safety factor when
    # the pre-conversion size is the pandas-sizing input; downstream GPU /
    # RAM heuristics over-allocate slightly rather than starving allocators.
    _CAT_SIZE_SAFETY_FACTOR = 1.5

    def _cat_heavy_size(df, raw_bytes):
        try:
            if not isinstance(df, pl.DataFrame):
                return raw_bytes
            cat_cols = [c for c in (cat_features or []) if c in df.columns]
            n_cols = max(int(len(df.columns)), 1)
            cat_frac = float(len(cat_cols)) / n_cols
            # Linear interp between 1.0x (no cat cols) and 1.5x (>=50% cat).
            scale = 1.0 + (_CAT_SIZE_SAFETY_FACTOR - 1.0) * min(cat_frac / 0.5, 1.0)
            return float(raw_bytes) * float(scale)
        except Exception:
            return raw_bytes

    if was_polars_input:
        try:
            if isinstance(train_df, pl.DataFrame):
                raw = float(train_df.estimated_size())
                train_df_size_bytes_cached = _cat_heavy_size(train_df, raw)
            if val_df is not None and isinstance(val_df, pl.DataFrame):
                raw_v = float(val_df.estimated_size())
                val_df_size_bytes_cached = _cat_heavy_size(val_df, raw_v)
        except Exception:
            train_df_size_bytes_cached = None
            val_df_size_bytes_cached = None

    if defer_pandas_conv:
        train_df_pd, val_df_pd, test_df_pd = train_df, val_df, test_df
        if verbose:
            if all_models_polars_native:
                logger.info("  Skipped pandas conversion -- all models are Polars-native")
            else:
                _strats = [get_strategy(m) for m in (mlframe_models or [])]
                non_native = [
                    m for m, s in zip(mlframe_models or [], _strats)
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
                _strats = [get_strategy(m) for m in (mlframe_models or [])]
                non_native = [
                    m for m, s in zip(mlframe_models or [], _strats)
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
        # CACHE-P1-7: recompute size from the post-conversion pandas frame so
        # downstream GPU-sizing heuristics use the actual realised footprint
        # instead of the polars pre-conversion estimate. The pandas frame
        # ``memory_usage(deep=True)`` is heavier than ``estimated_size`` but
        # only fires when conversion already paid the materialisation cost.
        try:
            if isinstance(train_df_pd, pd.DataFrame):
                train_df_size_bytes_cached = float(
                    train_df_pd.memory_usage(deep=True, index=False).sum()
                )
            if val_df_pd is not None and isinstance(val_df_pd, pd.DataFrame):
                val_df_size_bytes_cached = float(
                    val_df_pd.memory_usage(deep=True, index=False).sum()
                )
        except Exception:
            pass

    if cat_features and not defer_pandas_conv:
        if verbose:
            logger.info("Preparing %d categorical features for CatBoost: %s", len(cat_features), cat_features)
        # Joint train+val union for stable codes across splits. Test never
        # contributes to the union (held-out must look unseen); OOV values
        # land as null codes.
        prepare_dfs_for_catboost_joint(
            train_df=train_df_pd, val_df=val_df_pd, test_df=test_df_pd,
            cat_features=cat_features,
        )
    elif cat_features and defer_pandas_conv and verbose:
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
        defer_pandas_conv, baseline_rss_mb,
    )


def _log_cardinality_and_drift_snapshot(
    *,
    train_df: pl.DataFrame | pd.DataFrame | None,
    val_df: pl.DataFrame | pd.DataFrame | None,
    test_df: pl.DataFrame | pd.DataFrame | None,
    cat_features: list[str],
    text_features: list[str],
    embedding_features: list[str],
    ctx: Any | None = None,
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
        # Single lazy collect for ALL train cardinalities (was: N eager n_unique() calls -> N kernel launches).
        # Reference: helpers.py:1040-1047 -- the same implode-batch pattern that collapsed 14 collects to 1
        # in trainset_features_stats. Here on 100 cat cols this drops ~100 eager kernels to 1 collect.
        cols_present = [c for c in all_cat_cols if c in train_df.columns]
        if is_polars:
            if cols_present:
                _card_row = train_df.lazy().select(
                    [pl.col(c).n_unique().alias(c) for c in cols_present]
                ).collect()
                pairs = [(c, int(_card_row[c][0])) for c in cols_present]
            else:
                pairs = []
        else:
            pairs = [(c, int(train_df[c].nunique(dropna=False))) for c in cols_present]
        pairs.sort(key=lambda x: -x[1])
        summary = ", ".join(f"{c}:{n:_}" for c, n in pairs)
        logger.info("  Categorical cardinalities (train, n_unique, desc): %s", summary)

        # Drift log: val/test categories not seen in train.
        if is_polars and val_df is not None and test_df is not None and val_df.height > 0:
            # Per-col anti-join was 3 selects + 2 joins = ~5 eager passes; on 100 cols that's ~500 passes ~10-30 s.
            # Batched implode pattern: one lazy collect per frame yielding a 1-row frame whose cells are the
            # imploded unique-value lists. Anti-set is then a pure-Python set-difference on the materialised lists.
            drift_cols = [c for c, card in pairs
                          if card <= _DRIFT_SKIP_CARD
                          and c in val_df.columns and c in test_df.columns]
            drift_rows: list = []
            if drift_cols:
                # CAT-DRIFT-FULL-IMPLODE: cache the train-side ``unique().implode()`` result on
                # ctx keyed by (id(train_df), drift_cols_tuple). The drift snapshot is invoked up
                # to three times per suite (pre-split / post-split / pre-fit) on the same train
                # frame, each time paying a full lazy collect over hundreds of columns. Recompute
                # only when ctx is absent OR the train frame identity changes; val/test sides are
                # not cached because they're cheap relative to train and may rotate per pass.
                _cache_key = (id(train_df), tuple(drift_cols))
                _drift_cache = (
                    getattr(ctx, "_cat_drift_implode_cache", None)
                    if ctx is not None else None
                )
                _tr_sets_cached = (
                    _drift_cache.get(_cache_key) if isinstance(_drift_cache, dict) else None
                )
                if _tr_sets_cached is None:
                    _tr_uniq = train_df.lazy().select(
                        [pl.col(c).drop_nulls().unique().implode().alias(c) for c in drift_cols]
                    ).collect()
                    # Materialise to per-col sets ONCE; the previous code recomputed the
                    # ``set(_tr_uniq[c][0].to_list())`` per column inside the row loop.
                    _tr_sets_cached = {
                        c: set(_tr_uniq[c][0].to_list()) for c in drift_cols
                    }
                    if isinstance(_drift_cache, dict):
                        _drift_cache[_cache_key] = _tr_sets_cached
                _v_uniq = val_df.lazy().select(
                    [pl.col(c).drop_nulls().unique().implode().alias(c) for c in drift_cols]
                ).collect()
                _te_uniq = test_df.lazy().select(
                    [pl.col(c).drop_nulls().unique().implode().alias(c) for c in drift_cols]
                ).collect()
                _card_by_col = dict(pairs)
                for c in drift_cols:
                    tr_set = _tr_sets_cached[c]
                    val_only = sum(1 for x in _v_uniq[c][0].to_list() if x not in tr_set)
                    test_only = sum(1 for x in _te_uniq[c][0].to_list() if x not in tr_set)
                    drift_rows.append((c, _card_by_col[c], val_only, test_only))

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
    train_df_pandas_pre_meta: dict | None = None,
) -> tuple:
    """Auto-detect text + embedding features, optionally drop high-card columns, validate exclusivity, one-time Polars string->Categorical cast.

    Mutates ``metadata`` in-place with ``columns`` and ``cat_features``.
    """
    # Use pre-pipeline view so auto-detection sees original dtypes BEFORE the ordinal encoder converts strings to int codes.
    # Polars: ``train_df_polars_pre`` (frame alias, always populated; polars frames are conceptually immutable through the
    # public API so the alias is safe). Pandas: ``train_df_pandas_pre_meta`` dict, mutation-immune by construction
    # (column names / dtype-strings / cardinality / non-null counts baked at snapshot time). Fallback to post-pipeline
    # ``train_df`` only when both pre-views are absent (legacy callers / feature_types_first=False).
    if was_polars_input:
        detect_df = train_df_polars_pre
    else:
        detect_df = train_df
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
        pandas_meta=train_df_pandas_pre_meta if not was_polars_input else None,
    )

    # Capture pre-drop column data so dummy_baselines per_group_mean can use these as group
    # keys downstream (tree models drop them to avoid XGB QuantileDMatrix OOM).
    # Audit D P1-6 (2026-05-18): pre-fix loop ran ``_frame[c].to_numpy()`` per column per
    # split -- N independent Arrow batches per split. Now we do ONE ``_frame.select(cols)``
    # per split, materialise that 2D matrix through ``get_pandas_view_of_polars_df`` (split-
    # blocks Arrow bridge, ~32x faster than naive to_pandas on multi-col selects), then
    # peel the per-column numpy arrays from the resulting DataFrame view. Pandas-branch is
    # unchanged because pandas ``_frame[col]`` is already a Series view (no extra copy).
    dropped_high_card_data = {}
    if auto_high_card_drop:
        # Per-split single-pass materialisation for polars frames; pandas branch is per-col
        # (cheap Series view).
        _per_split_views: dict[str, Any] = {}
        for _label, _frame in (("train", train_df), ("val", val_df), ("test", test_df)):
            if _frame is None:
                continue
            _cols = _frame.columns if hasattr(_frame, "columns") else []
            _present = [c for c in auto_high_card_drop if c in _cols]
            if not _present:
                continue
            if isinstance(_frame, pl.DataFrame):
                try:
                    # Single select over ALL needed columns; Arrow split-blocks bridge.
                    from mlframe.training.utils import get_pandas_view_of_polars_df as _get_pd_view
                    _per_split_views[_label] = _get_pd_view(_frame.select(_present))
                except Exception:
                    # Fallback to bare to_pandas on the multi-col select; still 1 batch vs N.
                    try:
                        _per_split_views[_label] = _frame.select(_present).to_pandas()
                    except Exception:
                        _per_split_views[_label] = None
            else:
                _per_split_views[_label] = _frame
        for _col in auto_high_card_drop:
            _col_frames = {}
            for _label in ("train", "val", "test"):
                _view = _per_split_views.get(_label)
                if _view is None:
                    continue
                if _col not in getattr(_view, "columns", []):
                    continue
                try:
                    _col_frames[_label] = np.asarray(_view[_col])
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

    # One-time Polars string->Enum cast so XGB's arrow bridge doesn't choke on large_string.
    # Use pl.Enum (per-Series, no global cache impact) keyed off the train-only unique set;
    # val/test cast non-strict so OOV becomes null (matches the alignment semantics elsewhere
    # in the suite). pl.Categorical would widen the process-wide string cache (memory rule:
    # reference_polars_global_string_cache). Fixes audit B-P0-3 / Low-B11.
    if was_polars_input and all_models_polars_native and pipeline_config.skip_categorical_encoding:
        _string_types = (pl.Utf8, pl.String) if hasattr(pl, "String") else (pl.Utf8,)
        _keep_as_string = text_emb_set
        _str_cols = [c for c, dt in zip(train_df.columns, train_df.dtypes)
                     if dt in _string_types and c not in _keep_as_string]
        if _str_cols:
            # Build per-column Enum domain from train-only uniques (held-out test must stay unseen).
            _enum_domains: dict[str, list[str]] = {}
            for _c in _str_cols:
                try:
                    _u = train_df.select(pl.col(_c).drop_nulls().unique())[_c].to_list()
                    _enum_domains[_c] = sorted(_u)
                except Exception:
                    pass

            def _enum_cast(df, strict: bool):
                if df is None:
                    return df
                _existing = set(df.columns)
                _exprs = []
                for _c in _str_cols:
                    if _c not in _existing or _c not in _enum_domains:
                        continue
                    _exprs.append(pl.col(_c).cast(pl.Enum(_enum_domains[_c]), strict=strict))
                return df.with_columns(_exprs) if _exprs else df

            train_df = _enum_cast(train_df, strict=True)
            val_df = _enum_cast(val_df, strict=False)
            test_df = _enum_cast(test_df, strict=False)
            train_df_polars_pre = _enum_cast(train_df_polars_pre, strict=True)
            val_df_polars_pre = _enum_cast(val_df_polars_pre, strict=False)
            test_df_polars_pre = _enum_cast(test_df_polars_pre, strict=False)
            if verbose:
                logger.info("  Cast Polars string columns -> Enum once (shared across model loop)")

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
    target_by_type: Any = None,
    train_idx: np.ndarray | None = None,
) -> "FitPipelineResult":
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

    # Capture the RAW input column list BEFORE the main pipeline transform or
    # extensions stage runs. ``metadata["columns"]`` (set later, post-pipeline +
    # post-extensions) is the *output* schema the trained model was fit on; the
    # predict-time input-validation step needs the *input* schema instead. Without
    # this snapshot, ``_validate_input_columns_against_metadata`` filters predict
    # inputs against post-pipeline column names (truncatedsvd0..9 from a
    # PreprocessingExtensionsConfig dim_reducer, or sklearn one-hot expansions
    # cat_low_A / cat_low_B / ...) and drops every raw user column as "extra" -
    # leaving a (N, 0) frame that crashes the extensions transform with
    # ``Found array with 0 sample(s)`` before any model can run. Surfaced by
    # fuzz iter#189 (binary classification x lgb,linear,ridge x cat_enc=onehot
    # x dim_reducer=TruncatedSVD x 1M rows).
    if train_df is not None and hasattr(train_df, "columns"):
        if isinstance(train_df, pl.DataFrame):
            _raw_cols = list(train_df.columns)
        else:
            _raw_cols = train_df.columns.tolist()
        # SKEW-COL-ORDER: write both the explicit "raw_input_columns" key (post-fix canonical) and
        # the legacy "input_columns" alias. ``_validate_input_columns_against_metadata`` prefers the
        # explicit name; older serialised metadata still reads via the alias.
        metadata["raw_input_columns"] = list(_raw_cols)
        metadata["input_columns"] = list(_raw_cols)

    _strategies_for_polars_check = [get_strategy(m) for m in mlframe_models] if mlframe_models else []
    all_models_polars_native = bool(_strategies_for_polars_check) and all(
        s.supports_polars for s in _strategies_for_polars_check
    )

    # CatBoost-specific footgun warning: when CB is in the model suite AND categorical_encoding="ordinal"
    # AND the input frame carries categorical columns (polars Categorical/Enum, or pandas category dtype),
    # the ordinal encoder converts those columns to integer codes BEFORE CatBoost sees them. CatBoost then
    # loses its native categorical handling (combinations, target-statistics, one-hot small-cardinality
    # fast-path) and treats the int codes as ordered numerics, which silently degrades accuracy on
    # high-cardinality cats. Detection here is preventive (warning only; no code-path change). Fire the
    # WARN BEFORE the polars-fastpath auto-flip below otherwise the auto-flip silences the check on the
    # most common polars input path. Cat columns are detected directly from the train_df schema because
    # FeatureTypesConfig doesn't carry a cat_features list (the public surface is text_features +
    # embedding_features; cat_features are auto-detected downstream).
    _suite_models_lower = {str(m).lower() for m in (mlframe_models or [])}
    _has_cb = bool(_suite_models_lower & {"cb", "catboost"})
    _ordinal = (
        getattr(pipeline_config, "categorical_encoding", None) == "ordinal"
        and not getattr(pipeline_config, "skip_categorical_encoding", False)
    )
    _declared_cats: list[str] = []
    if train_df is not None:
        if isinstance(train_df, pl.DataFrame):
            # Use isinstance(d, pl.Enum) instead of str(d).startswith("Enum")
            # so dtype detection is API-stable across polars versions and survives any repr change.
            _enum_cls = getattr(pl, "Enum", None)
            _declared_cats = [
                n for n, d in train_df.schema.items()
                if d == pl.Categorical
                or (_enum_cls is not None and isinstance(d, _enum_cls))
                or d == pl.Utf8 or d == pl.String
            ]
        elif hasattr(train_df, "select_dtypes"):
            try:
                _declared_cats = train_df.select_dtypes(include=["category", "object", "string"]).columns.tolist()
            except Exception:
                _declared_cats = []
    if _has_cb and _ordinal and _declared_cats:
        # Previously a WARN-only check. Surfaced by the diverse-harness
        # fuzz profile (iter#36 with cat_low + text_col + cb): the ordinal
        # encoder turned text_col into ints, which CatBoost then refused
        # with "Invalid type for text_feature ... must have string type".
        # Auto-flip skip_categorical_encoding=True so CB sees the original
        # categorical/text columns and uses its native handling. Caller can
        # still force the old behaviour via
        # PreprocessingBackendConfig(skip_categorical_encoding=False).
        logger.warning(
            "  CatBoost in mlframe_models + categorical_encoding='ordinal' + %d "
            "categorical column(s) detected. Auto-flipping "
            "skip_categorical_encoding=True so CB keeps native cat-handling "
            "(combinations, target-statistics) and text_features stay string-typed. "
            "Set skip_categorical_encoding=False explicitly to restore the previous "
            "ordinal-then-CB behaviour.",
            len(_declared_cats),
        )
        pipeline_config = pipeline_config.model_copy(update={"skip_categorical_encoding": True})
        _ordinal = False

    # Auto-skip categorical encoding when all models handle categoricals natively. Runs AFTER the CB+ordinal
    # WARN above so the warning fires on the user's *requested* config rather than the auto-flipped one.
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
    # Skip datetime sources already decomposed by the FTE (it keeps the original via ``delete_original_cols=False``, so the suite would otherwise re-decompose and emit duplicate / overwriting cols on top of FTE's). The raw source is still removed below so downstream consumers (LGB / XGB / sklearn) don't see a leftover datetime64 column they can't dtype-promote.
    _fte_emitted_map = metadata.get("ftextractor_emitted_columns") or {}
    _fte_owned_dt_sources = [c for c in _dt_cols if c in _fte_emitted_map] if _fte_emitted_map else []
    if _fte_emitted_map:
        _dt_cols = [c for c in _dt_cols if c not in _fte_emitted_map]
    if _fte_owned_dt_sources:
        # FTE already produced derived cols for these; the raw datetime source must still be dropped so downstream model libs don't choke on a Datetime64 column. Match the ``delete_original_cols=True`` behaviour of the suite's own create_date_features call below.
        def _drop_source_cols(_frame, _cols):
            if _frame is None:
                return _frame
            _present = [c for c in _cols if c in _frame.columns]
            if not _present:
                return _frame
            if isinstance(_frame, pl.DataFrame):
                return _frame.drop(_present)
            return _frame.drop(columns=_present)
        train_df = _drop_source_cols(train_df, _fte_owned_dt_sources)
        val_df = _drop_source_cols(val_df, _fte_owned_dt_sources)
        test_df = _drop_source_cols(test_df, _fte_owned_dt_sources)
    if _dt_cols:
        from mlframe.feature_engineering.basic import create_date_features
        # Configurable set of dt accessors (year / ordinal_day / minute / ...).
        # Backward-compat default {day, weekday, month, hour} kept by
        # FeatureTypesConfig; callers opt into richer decomposition by passing
        # datetime_methods in their FeatureTypesConfig.
        _configured_methods = (
            set(feature_types_config.datetime_methods)
            if feature_types_config is not None and getattr(feature_types_config, "datetime_methods", None)
            else {"day", "weekday", "month", "hour"}
        )
        # ``create_date_features`` expects {accessor: np_dtype}. int8 fits most
        # cyclical fields; year exceeds int8 range so it needs int32. Pick
        # dtype per-method so the user doesn't have to know polars dtype rules.
        _wide_int_methods = {"year"}
        _dt_methods = {
            m: (np.int32 if m in _wide_int_methods else np.int8)
            for m in sorted(_configured_methods)
        }
        if verbose:
            logger.info(
                "Decomposing %d datetime column(s) into numeric features "
                "(%s) before pre-pipeline clone: %s",
                len(_dt_cols), "/".join(sorted(_dt_methods.keys())), _dt_cols,
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
        # Persist the resolved methods keyed by source column so predict can replay the same expansion deterministically. Stored as the accessor-name -> numpy-dtype-name map (json-friendly); the predict side resolves the name back to the numpy dtype.
        _persisted_methods = {m: _dt_methods[m].__name__ for m in _dt_methods}
        _store = metadata.setdefault("datetime_methods", {})
        for _src in _dt_cols:
            _store[_src] = dict(_persisted_methods)

    # Pre-pipeline polars-pre frames are unconditionally ALIASED to the input frames -- never cloned.
    # Audit-time concern (CONV-HIGH-1) was that polars-ds Blueprint.ordinal_encode / one_hot_encode
    # might mutate the source frame in place. Verified non-issue: ``bp.ordinal_encode(...)`` returns a
    # new Blueprint; ``bp.materialize()`` produces a pipeline; ``pipeline.transform(df)`` returns a
    # new DataFrame (see pipeline.py:1037). Polars frames are conceptually immutable through the
    # public API; Arrow buffers are Arc-counted (clone is a refcount bump, not a deep copy). The
    # global string cache (memory note: "polars 1.x global string cache") grows monotonically -- codes
    # for existing strings never shift, so aliasing the pre-encoding frame is safe even when the
    # encoder later sees additional strings. Downstream rebindings of ``train_df_polars_pre`` (via
    # ``_drop_cols_df`` at L645 / ``_precast_strings`` at L674) all return NEW frames and reassign,
    # so the aliased input is never mutated either.
    if was_polars_input:
        train_df_polars_pre = train_df
        val_df_polars_pre = val_df if isinstance(val_df, pl.DataFrame) else None
        test_df_polars_pre = test_df if isinstance(test_df, pl.DataFrame) else None
        cat_features_polars = get_polars_cat_columns(train_df)
    else:
        train_df_polars_pre = None
        val_df_polars_pre = None
        test_df_polars_pre = None
        cat_features_polars = []

    # Snapshot a pandas-input train_df BEFORE fit_pipeline applies ordinal /
    # one-hot encoding so the downstream auto-detect phase can see the raw
    # string / object dtypes. Without this snapshot the ordinal encoder runs
    # first, converts all string columns to integer codes, and the subsequent
    # auto-detect step (run on the post-pipeline frame) silently classifies
    # everything as numeric -- text columns never get promoted to text_features.
    # Polars input already has ``train_df_polars_pre`` for this purpose.
    # Gated on ``FeatureTypesConfig.feature_types_first`` so byte-for-byte
    # legacy reproductions can disable. fix audit row FE-P1-2.
    _feature_types_first = bool(
        getattr(feature_types_config, "feature_types_first", True)
        if feature_types_config is not None else True
    )
    # Mutation-immune snapshot for the downstream auto-detect phase. A frame-level snapshot (even ``.copy(deep=False)``) shares
    # the source block-manager and leaks any in-place numpy poke (``df[col].values[i] = x``) into the snapshot, silently
    # corrupting auto-detect's view of pre-encoding dtypes / cardinality. The dict captures every datum auto-detect needs
    # (column names, dtype strings, cardinality + non-null counts on text-candidate cols, embedding-shape sniff result on
    # object cols) at snapshot time so the recorded values are baked and immune to any later mutation on ``train_df``.
    # 100 GB frame discipline: only string/object/category columns get nunique / count scans; numeric blocks are never touched.
    train_df_pandas_pre_meta: dict | None = None
    if _feature_types_first and (not was_polars_input) and isinstance(train_df, pd.DataFrame):
        try:
            _text_cand_cols = [
                c for c in train_df.columns
                if train_df[c].dtype.kind in "OUSb" or isinstance(train_df[c].dtype, pd.CategoricalDtype)
            ]
            _n_unique: dict[str, int] = {}
            _non_null: dict[str, int] = {}
            for _c in _text_cand_cols:
                _s = train_df[_c]
                _n_unique[_c] = int(_s.nunique(dropna=True))
                _non_null[_c] = int(_s.notna().sum())
            # Embedding-shape sniff: object-dtype cells holding ndarray / list (e.g. sentence-transformer vectors) cannot
            # be auto-detected from dtype alone. Probe the first 8 non-null cells per object column at snapshot time so the
            # downstream consumer can route them to embedding_features without touching the (potentially-mutated) source.
            _embedding_object_cols: list[str] = []
            for _c in train_df.columns:
                if str(train_df[_c].dtype).startswith("object"):
                    try:
                        _first = next((v for v in train_df[_c].head(8) if v is not None), None)
                    except Exception:
                        _first = None
                    if _first is not None and (
                        hasattr(_first, "shape")
                        or (hasattr(_first, "__len__") and not isinstance(_first, (str, bytes)))
                    ):
                        _embedding_object_cols.append(_c)
            # Wave 54 (2026-05-20): pandas allows duplicate column names; the prior
            # {c: ...} comprehension silently collapsed dupes to one entry, so the
            # downstream schema-hash would mis-flag a "matching" schema and drop
            # auto-detect coverage for the duplicate columns. Refuse explicitly.
            _cols_list = list(train_df.columns)
            if len(set(_cols_list)) != len(_cols_list):
                from collections import Counter as _Counter
                _dupes = [_c for _c, _n in _Counter(_cols_list).items() if _n > 1]
                raise ValueError(
                    f"train_df has {len(_dupes)} duplicate column name(s) "
                    f"({_dupes[:5]}); deduplicate before fit() to keep schema-hash honest."
                )
            train_df_pandas_pre_meta = {
                "columns": _cols_list,
                "dtypes": {c: str(train_df[c].dtype) for c in _cols_list},
                "n_unique": _n_unique,
                "non_null": _non_null,
                "embedding_object_cols": _embedding_object_cols,
                "shape": tuple(train_df.shape),
            }
        except Exception:
            train_df_pandas_pre_meta = None
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
    # PySR symbolic regression (inside apply_preprocessing_extensions) needs a
    # 1-D y_train. Multi-target pipelines pass a target_by_type dict; pick the
    # first regression target as the supervised signal for symbolic feature
    # discovery. Classification-only setups: PySR is regression-only, falls
    # back to None and the function logs a warning.
    _y_train_for_ext = None
    if (
        preprocessing_extensions is not None
        and getattr(preprocessing_extensions, "pysr_enabled", False)
        and target_by_type is not None
    ):
        try:
            # target_by_type structure varies by extractor:
            #   (a) Dict[TargetTypes, Dict[str, ndarray]]  - nested
            #   (b) Dict[TargetTypes, ndarray]             - flat (single target)
            #   (c) Dict[str, ndarray]                     - "regression" -> arr
            # Iterate items() and match by str-cast to dodge any
            # StrEnum-identity-vs-string-equality quirks.
            _reg_targets = None
            if hasattr(target_by_type, "items"):
                for _k, _v in target_by_type.items():
                    if str(_k).lower().endswith("regression"):
                        _reg_targets = _v
                        break
            if _reg_targets is not None and not isinstance(_reg_targets, dict):
                # Case (b)/(c): _reg_targets is already a 1-D array-like.
                _vals_direct = _reg_targets
                _reg_targets = {"_default": _vals_direct}
            if _reg_targets:
                _first_name = next(iter(_reg_targets))
                _vals = _reg_targets[_first_name]
                # Audit D P2-5 (2026-05-18): polars/pandas → numpy is NEEDED here; PySR's
                # ``PySRRegressor.fit(X, y)`` consumes a numpy array, and the ndim / [:, 0]
                # downstream operations also assume numpy semantics. Conversion cannot be
                # pushed further.
                if hasattr(_vals, "to_numpy"):
                    _y_train_for_ext = _vals.to_numpy()
                else:
                    _y_train_for_ext = np.asarray(_vals)
                if _y_train_for_ext is not None and _y_train_for_ext.ndim > 1:
                    # Multi-output regression target -> first column for PySR.
                    # PySR is single-target by design; we surface the chosen target name so a
                    # multi-head regression doesn't silently pick the wrong head as the symbolic-FE signal.
                    if verbose:
                        logger.info(
                            "PySR symbolic FE: multi-output regression target detected (n=%d columns); "
                            "using first column '%s' as the supervised signal.",
                            _y_train_for_ext.shape[1], _first_name,
                        )
                    _y_train_for_ext = _y_train_for_ext[:, 0]
                # target_by_type carries the PRE-split full target; slice to train_idx
                # so PySR's symbolic FE only sees train-set y. Without this we hit a
                # length mismatch (e.g. full ~5M rows vs train ~4M) and PySR is silently
                # skipped.
                if (
                    _y_train_for_ext is not None
                    and train_idx is not None
                    and hasattr(train_df, "shape")
                    and len(_y_train_for_ext) != train_df.shape[0]
                ):
                    try:
                        _idx_arr = np.asarray(train_idx)
                        if len(_idx_arr) == train_df.shape[0] and int(_idx_arr.max()) < len(_y_train_for_ext):
                            _y_train_for_ext = _y_train_for_ext[_idx_arr]
                    except (TypeError, ValueError, IndexError):
                        pass
                if hasattr(train_df, "shape") and _y_train_for_ext is not None and len(_y_train_for_ext) != train_df.shape[0]:
                    if verbose:
                        logger.warning(
                            "PySR y_train length mismatch (target=%d, train rows=%d); skipping symbolic FE.",
                            len(_y_train_for_ext), train_df.shape[0],
                        )
                    _y_train_for_ext = None
        except Exception as _exc:
            if verbose:
                _diag = "n/a"
                try:
                    _diag = f"keys={list(target_by_type.keys()) if hasattr(target_by_type, 'keys') else type(target_by_type).__name__}"
                except Exception:
                    pass
                logger.warning(
                    "Could not extract y_train for PySR FE: %s: %s (target_by_type %s)",
                    type(_exc).__name__, _exc, _diag,
                )
            _y_train_for_ext = None
    t0_ext = timer()
    # Snapshot the train_df_polars_pre column set so we can detect which new
    # columns the extensions produced and back-merge them into the polars-pre
    # frames. fix audit row FE-P1-3.
    _pre_polars_columns_snapshot = (
        list(train_df_polars_pre.columns) if isinstance(train_df_polars_pre, pl.DataFrame) else None
    )
    # Capture PySR's equation -> column-name map so predict can replay symbolic features against the same content-hashed column names that training emitted.
    _pysr_equations_out: dict = {}
    train_df, val_df, test_df, extensions_pipeline = apply_preprocessing_extensions(
        train_df, val_df, test_df, preprocessing_extensions, verbose=verbose, y_train=_y_train_for_ext,
        out_pysr_equations=_pysr_equations_out,
    )
    if _pysr_equations_out:
        metadata["pysr_equations"] = dict(_pysr_equations_out)
    if verbose and preprocessing_extensions is not None:
        logger.info("  apply_preprocessing_extensions done in %s", _elapsed_str(t0_ext))
    if extensions_pipeline is not None:
        cat_features = []
        # Polars-fastpath consumers (CB / XGB polars-native path) only see the
        # polars-pre frames; copy the extension-produced new columns onto them
        # so models downstream see consistent feature sets. We use a pandas
        # bridge for the new columns only (existing polars-pre columns are kept
        # as-is to preserve native dtypes / categorical metadata).
        try:
            if (
                isinstance(train_df, pd.DataFrame)
                and _pre_polars_columns_snapshot is not None
                and was_polars_input
            ):
                _new_cols = [c for c in train_df.columns if c not in set(_pre_polars_columns_snapshot)]
                if _new_cols:
                    # Explicit (label, pandas-frame, polars-pre-frame) triples so the
                    # back-merge no longer fishes the polars-side frame out of locals()
                    # by string lookup. Captures the current binding at iteration time;
                    # rebinding below updates the same local name afterward.
                    _polars_pre_by_label = {
                        "train": train_df_polars_pre,
                        "val": val_df_polars_pre,
                        "test": test_df_polars_pre,
                    }
                    for _label, _pd_df in (
                        ("train", train_df),
                        ("val", val_df),
                        ("test", test_df),
                    ):
                        _pl_df = _polars_pre_by_label.get(_label)
                        if not isinstance(_pl_df, pl.DataFrame) or not isinstance(_pd_df, pd.DataFrame):
                            continue
                        if _pd_df.shape[0] != _pl_df.shape[0]:
                            # Row counts must match; otherwise the join could mis-align silently.
                            if verbose:
                                logger.warning(
                                    "polars-pre %s frame row mismatch for extension columns "
                                    "(pd=%d, pl=%d); skipping back-merge for this split.",
                                    _label, _pd_df.shape[0], _pl_df.shape[0],
                                )
                            continue
                        # Only the new columns we want to merge.
                        _new_df_pd = _pd_df[[c for c in _new_cols if c in _pd_df.columns]]
                        if _new_df_pd.shape[1] == 0:
                            continue
                        try:
                            _new_pl = pl.from_pandas(_new_df_pd)
                            _merged = _pl_df.hstack(_new_pl)
                            if _label == "train":
                                train_df_polars_pre = _merged
                            elif _label == "val":
                                val_df_polars_pre = _merged
                            else:
                                test_df_polars_pre = _merged
                        except Exception as _exc:
                            if verbose:
                                logger.warning(
                                    "Failed to back-merge extension columns into polars-pre %s frame: %s",
                                    _label, _exc,
                                )
        except Exception as _exc:
            if verbose:
                logger.warning(
                    "Polars-pre extension back-merge skipped (%s); polars-fastpath models will not see extension columns.",
                    _exc,
                )

    metadata["pipeline"] = pipeline
    metadata["extensions_pipeline"] = extensions_pipeline
    metadata["cat_features"] = cat_features
    _post_cols = train_df.columns.tolist() if isinstance(train_df, pd.DataFrame) else list(train_df.columns)
    # SKEW-COL-ORDER: write the explicit "post_pipeline_columns" name AND the legacy "columns" alias.
    # Predict-side validators prefer the explicit name; old metadata files still resolve via "columns".
    metadata["post_pipeline_columns"] = list(_post_cols)
    metadata["columns"] = _post_cols

    if verbose:
        logger.info("  Pipeline done -- train: %s, cat_features: %s", _df_shape_str(train_df), cat_features or '(none)')
        if was_polars_input and cat_features_polars and list(cat_features_polars) != list(cat_features or []):
            logger.info("  Pre-pipeline Polars cat_features: %s", cat_features_polars)
        logger.info("  PHASE 3 total: %s", _elapsed_str(t0_phase3))

    return FitPipelineResult(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        pipeline=pipeline,
        extensions_pipeline=extensions_pipeline,
        cat_features=cat_features,
        cat_features_polars=cat_features_polars,
        was_polars_input=was_polars_input,
        all_models_polars_native=all_models_polars_native,
        polars_pipeline_applied=polars_pipeline_applied,
        train_df_polars_pre=train_df_polars_pre,
        val_df_polars_pre=val_df_polars_pre,
        test_df_polars_pre=test_df_polars_pre,
        pipeline_config=pipeline_config,
        preprocessing_extensions=preprocessing_extensions,
        train_df_pandas_pre_meta=train_df_pandas_pre_meta,
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
) -> "TrainValTestSplitResult":
    """Train/val/test splitting with auto-stratification + group-aware splitting.

    Mutates ``metadata`` in-place with split sizes + per-split details.
    """
    if verbose:
        log_phase("PHASE 2: Train/Val/Test Splitting")

    t0_phase2 = timer()
    if verbose:
        logger.info(f"Making train_val_test split...")
    # Auto-stratify by target when no timestamps are present (without stratification,
    # rare-imbalance shuffles produce all-class-0 val slices). Three regimes:
    #   (a) single classification target  -> stratify on its ndarray directly
    #   (b) multiple classification targets (e.g. several binary heads) -> stratify on
    #       a composite key built from the row-tuple, encoded as an int class id.
    #       Gated on combined-cardinality <= MAX_COMPOSITE_CARDINALITY so the
    #       sklearn StratifiedShuffleSplit doesn't reject for sparse classes.
    #   (c) multilabel target (N, K)      -> if iterative-stratification is installed,
    #       pass its ndarray through; otherwise fall back to first-label stratification
    #       as a best-effort over the all-classes-fully-balanced corner case.
    _MAX_COMPOSITE_CARDINALITY = 200  # caps the (b) regime at ~200 distinct row-tuples
    _stratify_y = None
    if timestamps is None and isinstance(target_by_type, dict):
        _classification_targets = []
        _multilabel_target = None
        for _tt, _named in target_by_type.items():
            _tt_name = getattr(_tt, "name", str(_tt)).upper()
            if "MULTILABEL" in _tt_name:
                # Multilabel arrives as (N, K) ndarray under one key; capture and stop.
                if isinstance(_named, dict):
                    _ml_vals = next(iter(_named.values()), None)
                else:
                    _ml_vals = _named
                _multilabel_target = _ml_vals
                continue
            if "CLASS" in _tt_name and isinstance(_named, dict):
                for _tn, _tv in _named.items():
                    if _tv is not None:
                        _classification_targets.append(_tv)
        if _multilabel_target is not None:
            try:
                _ml_arr = np.asarray(_multilabel_target)
                if _ml_arr.ndim == 2 and _ml_arr.shape[1] >= 1:
                    # Prefer the proper iterative-stratification path when available.
                    try:
                        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit  # noqa: F401
                        _stratify_y = _ml_arr
                    except ImportError:
                        # Best-effort fallback: stratify on the first label column. Better
                        # than nothing when one of the K labels is the rare class.
                        _first = _ml_arr[:, 0]
                        _u, _c = np.unique(_first, return_counts=True)
                        if len(_u) >= 2 and _c.min() >= 2:
                            _stratify_y = _first
                        else:
                            logger.warning(
                                "Auto-stratify: multilabel first-label fallback has %d "
                                "unique values with min-count=%d (need >=2 + min-count>=2); "
                                "stratification disabled. val/test slices may be class-degenerate.",
                                len(_u), int(_c.min()) if len(_c) else 0,
                            )
            except Exception as _strat_err:
                logger.warning(
                    "Auto-stratify: multilabel build failed (%s: %s); shuffled-only splits.",
                    type(_strat_err).__name__, _strat_err,
                )
                _stratify_y = None
        elif len(_classification_targets) == 1:
            try:
                _arr = np.asarray(_classification_targets[0])
                if _arr.ndim == 1:
                    _u, _c = np.unique(_arr, return_counts=True)
                    if len(_u) >= 2 and _c.min() >= 2:
                        _stratify_y = _arr
                    else:
                        # Surface so the rare-imbalance scenario (single-class slice OR
                        # rare-class with one sample) isn't misdiagnosed as random-seed
                        # flakiness when val ends up all-class-0. Pre-fix this branch
                        # silently flipped to shuffled-only.
                        logger.warning(
                            "Auto-stratify: single classification target has %d unique "
                            "classes with min-count=%d (need >=2 + min-count>=2); "
                            "stratification disabled. val/test slices may be class-degenerate.",
                            len(_u), int(_c.min()) if len(_c) else 0,
                        )
            except Exception as _strat_err:
                logger.warning(
                    "Auto-stratify: single-target build failed (%s: %s); shuffled-only splits.",
                    type(_strat_err).__name__, _strat_err,
                )
                _stratify_y = None
        elif len(_classification_targets) > 1:
            try:
                _arrs = [np.asarray(_t) for _t in _classification_targets]
                _n = len(_arrs[0])
                if all(_a.ndim == 1 and len(_a) == _n for _a in _arrs):
                    # Composite key: each row maps to an integer class id from
                    # (val_t0, val_t1, ..., val_tK) tuple. np.unique on stacked (N, K)
                    # returns_inverse for the encoding in one pass.
                    _stack = np.stack(_arrs, axis=1)
                    _, _composite_ids = np.unique(_stack, axis=0, return_inverse=True)
                    _u, _c = np.unique(_composite_ids, return_counts=True)
                    if 2 <= len(_u) <= _MAX_COMPOSITE_CARDINALITY and _c.min() >= 2:
                        _stratify_y = _composite_ids
                    elif len(_u) < 2 or _c.min() < 2:
                        logger.warning(
                            "Auto-stratify: composite key has %d distinct row-tuples with "
                            "min-count=%d (need >=2 + min-count>=2); stratification disabled. "
                            "val/test slices may be class-degenerate.",
                            len(_u), int(_c.min()) if len(_c) else 0,
                        )
                    elif len(_u) > _MAX_COMPOSITE_CARDINALITY:
                        # Surface the silent fallback to shuffled-only splits so operators
                        # know auto-stratification was abandoned on multi-head targets that
                        # exceed the composite-cardinality cap; otherwise they re-discover
                        # the all-class-0-val-slice bug under heavy class imbalance.
                        logger.warning(
                            "Auto-stratify: composite key has %d distinct row-tuples > "
                            "_MAX_COMPOSITE_CARDINALITY=%d; falling back to UNstratified "
                            "shuffle splits. Rare-class imbalance may produce all-one-class "
                            "val/test slices. Reduce the number of classification heads or "
                            "pre-compute a stratify_y manually to restore stratification.",
                            len(_u), _MAX_COMPOSITE_CARDINALITY,
                        )
            except Exception:
                _stratify_y = None
    # Group-aware splitting opt-in: when the extractor produced ``group_ids`` and
    # ``split_config.use_groups`` is set, route through GroupShuffleSplit.
    _groups = group_ids if (split_config.use_groups and group_ids is not None and len(group_ids) > 0) else None
    with phase("split_data"):
        # ``calib_size`` is a TrainingSplitConfig field for opt-in
        # post-calibration; ``make_train_test_split`` does not accept it
        # (the calibration slice is carved by downstream code, not the
        # splitter). Exclude alongside ``use_groups`` (which is
        # consumed above to derive ``_groups``).
        train_idx, val_idx, test_idx, train_details, val_details, test_details = make_train_test_split(
            df=df,
            timestamps=timestamps,
            stratify_y=_stratify_y,
            groups=_groups,
            **split_config.model_dump(exclude={"use_groups", "calib_size"}),
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

    return TrainValTestSplitResult(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_details=train_details,
        val_details=val_details,
        test_details=test_details,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        fairness_subgroups=fairness_subgroups,
        fairness_features=fairness_features,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        test_sequences=test_sequences,
        baseline_rss_mb=baseline_rss_mb,
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

    # Capture FTE-emitted columns (source -> [derived]) so the downstream pipeline phase can skip re-decomposing already-handled datetime sources. Persist to metadata for predict-path informational use; FTE itself re-derives on predict input via its own ``transform`` call.
    _fte_emitted = getattr(features_and_targets_extractor, "ftextractor_emitted_columns", None)
    if isinstance(_fte_emitted, dict) and _fte_emitted:
        ctx.metadata["ftextractor_emitted_columns"] = {k: list(v) for k, v in _fte_emitted.items()}

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
        # Wave 46 (2026-05-20): raw ctx.model_name plumbed into os.path.join is a path-
        # traversal vector ("../../evil" escapes models dir; an absolute "/foo" or "C:/x"
        # eats the prefix entirely). Slugify mirrors the non-LTR sibling paths at
        # _setup_helpers.py:852 and _phase_finalize.py:71.
        from pyutilz.strings import slugify as _slugify
        _save_dir = os.path.join(_data_dir, _models_dir, _slugify(ctx.model_name))

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
