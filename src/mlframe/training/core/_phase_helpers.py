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

# Models that consume pandas Categorical via the joint train+val
# encoding. Linear / MLP / Neural strategies operate on numeric tensors
# and don't read the Categorical dtype at all; running the joint cat
# prep for them is pure waste (c0095 iter195 found 865ms wasted on a
# linear-only multilabel combo at 200k rows). LearningToRank LGB / XGB
# / CB rankers also consume cat_features through the joint encoding,
# so they're included here.
_MODELS_NEEDING_PANDAS_CAT_PREP: frozenset[str] = frozenset({
    "cb", "lgb", "xgb", "hgb",
})


def _models_need_pandas_cat_prep(
    mlframe_models: list[str] | None,
    recurrent_models: list[str] | None,
    rfecv_models: list[str] | None,
) -> bool:
    """Return True iff at least one configured model branch consumes the
    pandas-Categorical cat_features prepared by ``prepare_dfs_for_catboost_joint``.

    All booster strategies (CB / LGB / XGB / HGB) read the Categorical
    dtype directly. Linear / MLP / Neural / Recurrent operate on numeric
    arrays after the suite's downstream encoders, so they never look at
    the Categorical dtype.

    RFECV: its inner estimator IS a booster (lgb_rfecv / cb_rfecv / xgb_rfecv
    / hgb_rfecv) -- when present we conservatively assume the joint cat
    prep is needed even if no top-level booster is configured. ``rfecv_models``
    is the active rfecv-estimator subset (after the suite's auto-filter); an
    empty list means RFECV is disabled for this run.
    """
    for src in (mlframe_models, recurrent_models, rfecv_models):
        if not src:
            continue
        for m in src:
            base = m.lower().replace("_rfecv", "")
            if base in _MODELS_NEEDING_PANDAS_CAT_PREP:
                return True
    return False


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
    elif ml_strategy == "multi_target_regression":
        # F-34 (2026-05-31): keep (N, K) regression targets as-is, route
        # them to TargetTypes.MULTI_TARGET_REGRESSION. Strategies that
        # support native multi-target (CatBoost MultiRMSE, XGBoost
        # multi_output_tree, MLP K-head, sklearn Linear/Ridge/RF) fit
        # one model per target_name producing (N, K) predictions; others
        # are wrapped by sklearn.multioutput.MultiOutputRegressor at
        # build time. This is the OPPOSITE of "per_target": instead of
        # exploding K columns into K independent 1-D fits, we keep them
        # joint so the trunk / boosting ensemble exploits target
        # correlations. Picked per dataset basis: per_target is safer
        # for uncorrelated targets, multi_target_regression captures
        # correlated targets in a single shared model.
        _kept = dict(new_target_by_type[TargetTypes.REGRESSION])
        _routed = dict(new_target_by_type.get(TargetTypes.MULTI_TARGET_REGRESSION, {}))
        _routed_names: list[str] = []
        for _tn, _tv in list(new_target_by_type[TargetTypes.REGRESSION].items()):
            _arr = np.asarray(_tv)
            if _arr.ndim == 2 and _arr.shape[1] >= 2:
                _routed[_tn] = _arr  # keep the full (N, K) frame
                _kept.pop(_tn, None)
                _routed_names.append(_tn)
                logger.info(
                    "[CompositeTargetDiscovery] multi-target regression: kept '%s' "
                    "as (N, %d) under TargetTypes.MULTI_TARGET_REGRESSION (joint fit).",
                    _tn, _arr.shape[1],
                )
        new_target_by_type[TargetTypes.REGRESSION] = _kept
        if _routed:
            new_target_by_type[TargetTypes.MULTI_TARGET_REGRESSION] = _routed
        if _routed_names:
            metadata.setdefault("multi_target_regression_routing", {})[
                str(TargetTypes.MULTI_TARGET_REGRESSION)
            ] = _routed_names
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
        # Sizing for downstream GPU / RAM heuristics. Two-tier strategy:
        # 1. If ``was_polars_input`` already populated ``train_df_size_bytes_cached`` via
        #    ``estimated_size()`` + cat-heavy inflation (lines ~473-483 above), KEEP that value.
        #    On a 4M-row x 25-col object-heavy frame ``memory_usage(deep=True)`` scans every cell
        #    and takes ~17.6s (measured 2026-05-24); the polars estimate plus the 1.5x cat-heavy
        #    factor is accurate enough for GPU-sizing heuristics that already over-allocate.
        # 2. Otherwise fall back to ``memory_usage(deep=False)`` -- shallow scan returns in <1ms
        #    by reading buffer-block sizes per column. The object-dtype undercount that
        #    ``deep=True`` was guarding against (5GB vs 0.72GB on the test fixture) is acceptable
        #    given how much wall-time the deep scan costs; downstream consumers can request the
        #    exact value via a follow-up ``deep=True`` call if needed (none currently do).
        _t0_memsize = timer()
        try:
            if isinstance(train_df_pd, pd.DataFrame) and train_df_size_bytes_cached is None:
                train_df_size_bytes_cached = float(
                    train_df_pd.memory_usage(deep=False, index=False).sum()
                )
            if (val_df_pd is not None and isinstance(val_df_pd, pd.DataFrame)
                    and val_df_size_bytes_cached is None):
                val_df_size_bytes_cached = float(
                    val_df_pd.memory_usage(deep=False, index=False).sum()
                )
        except Exception:
            pass
        if verbose:
            _memsize_elapsed = timer() - _t0_memsize
            if _memsize_elapsed > 1.0:
                logger.info(
                    "  trainset_features_stats memory_usage: %.1fs "
                    "(train=%.1fMB, val=%.1fMB) -- runs once per suite",
                    _memsize_elapsed,
                    (train_df_size_bytes_cached or 0) / 1e6,
                    (val_df_size_bytes_cached or 0) / 1e6,
                )

    # OPT-1 bench-attempt-rejected (2026-05-23): gated joint cat prep on
    # mlframe_models containing CB/HGB/LGB/XGB/*_rfecv. Verified on c0095
    # (linear-only multilabel 200k): gate fires correctly (prep call
    # eliminated, saving ~865ms cumtime) BUT overall wall went 25.05s ->
    # 32.48s. sklearn.multioutput.fit went 18.7s -> 23.4s (+4.7s); the
    # joint Categorical encoding apparently provides a downstream win for
    # sklearn linear pipelines that exceeds the prep cost. NET REGRESSION.
    # Keeping the always-on path; `_models_need_pandas_cat_prep` helper
    # stays as a tool for future targeted opts.
    if cat_features and not defer_pandas_conv:
        if verbose:
            logger.info("Preparing %d categorical features for CatBoost: %s", len(cat_features), cat_features)
        # Joint train+val union for stable codes across splits. Test never
        # contributes to the union (held-out must look unseen); OOV values
        # land as null codes.
        # Observability: bracket with t0/t1 logs so the
        # cat-feature-prep step's wall-time is visible (silent gap on
        # 4M-row frames previously).
        _t0_cb_prep = timer()
        prepare_dfs_for_catboost_joint(
            train_df=train_df_pd, val_df=val_df_pd, test_df=test_df_pd,
            cat_features=cat_features,
        )
        if verbose:
            logger.info(
                "  prepare_dfs_for_catboost_joint: %.1fs (cat_features=%d)",
                timer() - _t0_cb_prep, len(cat_features),
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


# Wave 105 (2026-05-21): _phase_auto_detect_feature_types, _phase_fit_pipeline,
# and _phase_train_val_test_split moved to sibling _phase_helpers_fit_split.py.
# Re-exported below so existing callers
# (`from ._phase_helpers import _phase_fit_pipeline`, etc.) keep working.
from ._phase_helpers_fit_split import (  # noqa: F401, E402
    _phase_auto_detect_feature_types,
    _phase_fit_pipeline,
    _phase_train_val_test_split,
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
