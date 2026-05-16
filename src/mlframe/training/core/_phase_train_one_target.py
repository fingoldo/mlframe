"""_train_one_target - per-target training entry point."""
from __future__ import annotations

import hashlib
import inspect
import logging
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

from sklearn.base import clone

from pyutilz.strings import slugify
from pyutilz.system import tqdmu_lazy_start

from mlframe.models.ensembling import score_ensemble
from .._ram_helpers import estimate_df_size_mb, get_process_rss_mb, maybe_clean_ram_and_gpu
from ..phases import phase
from ..models import is_neural_model
from ..strategies import get_strategy
from ..train_eval import process_model, select_target
from ..utils import compute_model_input_fingerprint, filter_existing, get_pandas_view_of_polars_df, log_ram_usage
from ._misc_helpers import _build_tier_dfs, _compute_neural_max_time, _elapsed_str, _filter_polars_cat_features_by_dtype, _maybe_clear_shim_cache, _prep_polars_df, _split_preds_probs
from ._phase_diagnostics import run_per_target_diagnostics
from ._phase_dummy_baselines import run_dummy_baselines
from ._phase_temporal_audit import _format_temporal_audit_report, _plot_target_over_time
from ._setup_helpers import _build_common_params_for_target, _build_pre_pipelines, _build_process_model_kwargs, _setup_model_directories, _should_skip_catboost_metamodel
from ..strategies import PipelineCache

logger = logging.getLogger(__name__)


def _compute_pipeline_cache_key(
    strategy_cache_key: str,
    pre_pipeline_name: str | None,
    feature_tier,
    supports_polars: bool,
    cat_features,
    text_features,
    embedding_features,
) -> str:
    """Build the PipelineCache lookup key for a (strategy, pre_pipeline, tier, kind, features) combo.

    The features digest folds (cat, text, embedding) lists through blake2b so cache HIT invalidates
    when the user reshapes those lists between sessions yet stays stable across list ordering;
    without it, a tier frame prepared for one (cat/text/embedding) split could be served to a later
    session that toggled a column's role. Sorting before serialization gives a deterministic byte
    stream: frozenset.__repr__ iterates in hash-seeded order (PYTHONHASHSEED) and would change the
    digest across processes for the same membership.
    """
    _tier_suffix = f"_tier{feature_tier}"
    _kind_suffix = f"_kind{'pl' if supports_polars else 'pd'}"
    _feats_repr = repr((
        tuple(sorted(cat_features or ())),
        tuple(sorted(text_features or ())),
        tuple(sorted(embedding_features or ())),
    ))
    _feats_suffix = f"_feats{hashlib.blake2b(_feats_repr.encode(), digest_size=8).hexdigest()}"
    if pre_pipeline_name:
        return f"{strategy_cache_key}_{pre_pipeline_name}{_tier_suffix}{_kind_suffix}{_feats_suffix}"
    return f"{strategy_cache_key}{_tier_suffix}{_kind_suffix}{_feats_suffix}"


# XGB DMatrix / LGB Dataset reuse cache attribute names: forwarded across sklearn.clone() in both
# directions (template -> clone before fit; clone -> template after fit) so the weight-schema loop
# reuses the heavy binned dataset via set_label / set_weight instead of rebuilding.
_DATASET_REUSE_CACHE_ATTRS = (
    "_cached_train_dmatrix",
    "_cached_train_key",
    "_cached_val_dmatrix",
    "_cached_val_key",
    "_cached_train_dataset",
    "_cached_val_dataset",
)


def _forward_dataset_reuse_cache(src, dst, attrs=_DATASET_REUSE_CACHE_ATTRS, *, skip_none: bool = False):
    """Copy each present attr from ``src`` onto ``dst``.

    CODE-LOW-7: both the template -> clone forward and the clone -> template back transfer used to
    inline the same loop with slightly different ``if _val is not None`` guard. Centralised here so
    additions to ``_DATASET_REUSE_CACHE_ATTRS`` flow to both call sites automatically.

    ``skip_none=True`` matches the back-transfer's behaviour: only carry non-None caches up to the
    template, otherwise a clone that did not populate the cache would NULL out the template's prior
    value and defeat the reuse.
    """
    for _attr in attrs:
        if not hasattr(src, _attr):
            continue
        _val = getattr(src, _attr)
        if skip_none and _val is None:
            continue
        try:
            setattr(dst, _attr, _val)
        except Exception as _attr_err:
            logger.debug("Could not transfer %s from %r to %r: %s", _attr, type(src).__name__, type(dst).__name__, _attr_err)


# Heuristic: if reclaim is under this share of the dropped-frame footprint, something is still pinning the buffers.
_POLARS_RELEASE_MIN_RECLAIM_FRACTION = 0.05


# ============================================================================================
# Suite-scoped feature-side cache helpers. The per-target inner loop in _train_one_target
# stages tier-DFs / pl.Enum maps / prepared polars frames / fingerprints into ctx.artifacts
# so the NEXT target's call reads them off ctx instead of rebuilding (CB Pool / XGB DMatrix /
# LGB Dataset all rebuild via id(train_df) keys and the train_df pointer is pinned by ctx).
# Entries store REFERENCES, never clones - a 100GB frame is shared between the cache slot and
# ctx.train_df_polars. Polars-tier entries are dropped when polars frames are released
# (``_release_ctx_polars_frames``) since their pinned references would otherwise defeat the
# release. Dataset-reuse cache (XGB DMatrix / LGB Dataset) is keyed by mlframe_model_name and
# bridges the per-target rebuild of models_params: the binned dataset built on target 1 gets
# re-attached onto target 2's freshly-built model template via _DATASET_REUSE_CACHE_ATTRS.
# ============================================================================================

_FEATURE_SIDE_CACHE_KEY = "feature_side_cache"
_DATASET_REUSE_CACHE_KEY = "dataset_reuse_cache"


def _ensure_ctx_artifacts(ctx) -> dict:
    """Return ctx.artifacts as a dict, materialising it if the dataclass default left it as None.

    ``ctx.artifacts`` is declared ``dict = field(default_factory=dict)`` in _training_context.py
    so normal construction produces an empty dict, BUT older test fixtures and direct field
    assignments can land ``None`` on the slot. Calling .setdefault() then AttributeErrors before
    the helper has a chance to install its key.
    """
    artifacts = ctx.artifacts
    if artifacts is None:
        artifacts = {}
        ctx.artifacts = artifacts
    return artifacts


def _get_feature_side_cache(ctx) -> dict:
    """Return the (creating-if-needed) suite-scoped feature-side cache off ctx.artifacts."""
    return _ensure_ctx_artifacts(ctx).setdefault(_FEATURE_SIDE_CACHE_KEY, {})


def _get_dataset_reuse_cache(ctx) -> dict:
    """Return the (creating-if-needed) suite-scoped dataset-reuse cache off ctx.artifacts.

    Keyed by ``mlframe_model_name``; each entry is a dict of ``_DATASET_REUSE_CACHE_ATTRS`` ->
    value captured from the prior target's fitted model template before _maybe_clear_shim_cache
    nuked it. The per-target restore happens on the FRESH ``models_params[name]["model"]``
    template right before clone(), so set_label / set_weight on the same train_df pointer reuses
    the heavy binned dataset across targets without rebuild.
    """
    return _ensure_ctx_artifacts(ctx).setdefault(_DATASET_REUSE_CACHE_KEY, {})


def _invalidate_polars_feature_side_cache(ctx) -> None:
    """Drop every polars-tier entry from ctx.artifacts['feature_side_cache'].

    Called from ``_release_ctx_polars_frames`` (the only place where ctx polars frames go to
    None) so the next target's loop doesn't read back stale pointers into freed frames. Pandas-
    tier entries (``supports_polars=False``) are preserved - they live in their own keys and
    point at frames that are NOT being released here.
    """
    cache = (ctx.artifacts or {}).get(_FEATURE_SIDE_CACHE_KEY)
    if not cache:
        return
    # Cache shape: cache[pp_name] -> {"tier_dfs": {sub_key -> dict}, "prepared_frames":
    # {sub_key -> dict}, "tier_enum_map": {sub_key -> map}}. Sub-keys are tuples and we
    # drop only the polars-tier ones; the "tier_enum_map" group is polars-only by
    # construction so it can be cleared whole.
    for _pp_name, _pp_payload in list(cache.items()):
        if not isinstance(_pp_payload, dict):
            continue
        for _group in ("tier_dfs", "prepared_frames"):
            _group_map = _pp_payload.get(_group)
            if not isinstance(_group_map, dict):
                continue
            # tier_dfs sub-key is (tier_tuple, kind) where kind is "pl" / "pd"; prepared_frames
            # sub-key is (tier_tuple, supports_polars, strategy_class, cb_text_pass). Polars
            # marker: kind=="pl" OR supports_polars==True (positional element 1).
            _polars_sub_keys = []
            for _sub_key in list(_group_map.keys()):
                if not isinstance(_sub_key, tuple) or len(_sub_key) < 2:
                    continue
                _kind = _sub_key[1]
                if _kind == "pl" or _kind is True:
                    _polars_sub_keys.append(_sub_key)
            for _k in _polars_sub_keys:
                _group_map.pop(_k, None)
        # tier_enum_map is polars-only by construction (the per-target loop only writes to
        # it on polars_fastpath_active); a polars frame release means all entries are stale.
        _enum_map = _pp_payload.get("tier_enum_map")
        if isinstance(_enum_map, dict):
            _enum_map.clear()


def _capture_dataset_reuse_cache(
    ctx,
    mlframe_model_name: str,
    model_template,
) -> None:
    """Snapshot ``_DATASET_REUSE_CACHE_ATTRS`` off ``model_template`` into ctx.artifacts.

    Runs BEFORE ``_maybe_clear_shim_cache`` so the next target gets the live binned dataset
    (XGB DMatrix / LGB Dataset) rather than the post-clear None. Skips entries whose value is
    None - those entries would defeat the next target's cache-hit check (``is not None``).
    """
    if model_template is None:
        return
    captured = {}
    for _attr in _DATASET_REUSE_CACHE_ATTRS:
        if not hasattr(model_template, _attr):
            continue
        _val = getattr(model_template, _attr)
        if _val is None:
            continue
        captured[_attr] = _val
    if captured:
        _get_dataset_reuse_cache(ctx)[mlframe_model_name] = captured


def _restore_dataset_reuse_cache(
    ctx,
    mlframe_model_name: str,
    model_template,
) -> None:
    """Re-attach ``_DATASET_REUSE_CACHE_ATTRS`` from ctx.artifacts onto ``model_template``.

    The per-target rebuild of ``models_params`` produces a fresh estimator without the cache
    attributes; this restore wires the previous target's binned dataset back on so the next
    forward-transfer-into-clone() carries it forward, and the shim's signature_of(X) check
    detects the same X (ctx-pinned across targets) and triggers the set_label / set_weight
    swap instead of a fresh build. No-op when there is no prior capture, or when target 1
    has not run yet for this model.
    """
    if model_template is None:
        return
    captured = (ctx.artifacts or {}).get(_DATASET_REUSE_CACHE_KEY, {}).get(mlframe_model_name)
    if not captured:
        return
    for _attr, _val in captured.items():
        try:
            setattr(model_template, _attr, _val)
        except Exception as _attr_err:
            logger.debug(
                "Could not restore %s on %s template: %s",
                _attr, mlframe_model_name, _attr_err,
            )


def _release_ctx_polars_frames(
    ctx,
    baseline_rss_mb: float,
    df_size_mb: float,
    *,
    verbose: bool,
    reason: str,
) -> float:
    """Drop ctx.{train,val,test}_df_polars strong refs, then trigger maybe_clean_ram_and_gpu and verify reclaim.

    The naked ``del train_df_polars`` at each call site only released the local alias inside
    ``_train_one_target``; the ctx attributes (assigned at lines 123-125 from ctx.*_df_polars) kept the
    real strong reference alive, so ``maybe_clean_ram_and_gpu`` had nothing to reclaim and the log line
    claiming a release was misleading. Centralised here so both call sites stay in sync and the post-release
    sanity check (RSS drop vs estimated frame footprint) flags any future regression where a new strong
    ref to the same frames is introduced upstream without being scrubbed here.
    """
    expected_mb = 0.0
    for _attr in ("train_df_polars", "val_df_polars", "test_df_polars"):
        _frame = getattr(ctx, _attr, None)
        if _frame is None:
            continue
        try:
            _sz = estimate_df_size_mb(_frame)
        except Exception:
            _sz = 0.0
        if _sz and _sz != float("inf"):
            expected_mb += float(_sz)
    rss_before_mb = get_process_rss_mb()
    ctx.train_df_polars = None
    ctx.val_df_polars = None
    ctx.test_df_polars = None
    # Drop polars-tier entries from the suite-scoped feature-side cache so they don't pin the
    # frames we just released. Pandas-tier entries are preserved - they point at separate
    # frames not touched by this release.
    _invalidate_polars_feature_side_cache(ctx)
    new_baseline = maybe_clean_ram_and_gpu(baseline_rss_mb, df_size_mb, verbose=verbose, reason=reason)
    if expected_mb > 0.0:
        rss_after_mb = get_process_rss_mb()
        delta_mb = rss_before_mb - rss_after_mb
        if delta_mb < _POLARS_RELEASE_MIN_RECLAIM_FRACTION * expected_mb:
            logger.warning(
                "ctx polars frames released but RSS dropped only %.1f MB; expected at least %.1f MB - check for lingering refs",
                delta_mb,
                expected_mb,
            )
    return new_baseline


def _maybe_run_feature_handling_apply(
    ctx,
    *,
    cur_target_name: str,
    train_df,
    val_df,
    test_df,
    current_train_target,
    sample_weight=None,
):
    """Run feature_handling_apply once per target when ctx carries a FeatureHandlingConfig; else no-op.

    Returns the FeatureHandlingResult on success or None when disabled / failed. Fitted state is also
    stashed under ctx.artifacts["feature_handling_fitted"][cur_target_name] so a future predict-side
    wave can replay handlers without re-fitting. ctx.artifacts is the only ctx slot we may write to
    here -- TrainingContext uses slots=True so adding a new attribute would AttributeError, and the
    SCOPE constraint forbids touching _training_context.py in this wave.

    sample_weight is accepted for forward compatibility: feature_handling_apply does not yet take it
    (validated against apply.py 2026-05-16). The keyword is plumbed through so a later apply.py
    extension picks it up without a second wire-in change here. NOTE: the underlying handlers do
    consume sample_weight via LeakageSafeEncoder -- once apply.py grows the kwarg, drop the silent
    discard below.

    model_kind comes from ctx.sorted_mlframe_models[0] -- the first concrete kind drives FHC
    validation; the resulting fitted state is model-agnostic for the handlers wired in v1 (TF-IDF,
    target-encoder, custom), so one call seeds the FeatureCache for every model that follows.
    """
    fhc = getattr(ctx, "feature_handling_config", None)
    if fhc is None:
        return None
    try:
        from mlframe.training.feature_handling import feature_handling_apply  # local: avoid suite-import cost when FHC is off
    except ImportError:  # pragma: no cover
        return None

    sorted_models = getattr(ctx, "sorted_mlframe_models", None) or getattr(ctx, "mlframe_models", None) or []
    if not sorted_models:
        return None
    model_kind = sorted_models[0]

    try:
        result = feature_handling_apply(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            train_target=current_train_target,
            fhc=fhc,
            model_kind=model_kind,
        )
    except ValueError as fhc_err:
        # Surface configuration errors with the kwarg name so users grep the right place; chain so the
        # original validation traceback is preserved.
        raise ValueError(
            f"feature_handling_config rejected for model_kind={model_kind!r} on target "
            f"{cur_target_name!r}: {fhc_err}"
        ) from fhc_err
    except Exception as fhc_err:
        logger.warning(
            "feature_handling_apply failed for target %r (model_kind=%s): %s; continuing without FHC enrichment for this target.",
            cur_target_name, model_kind, fhc_err,
        )
        return None

    # ctx.artifacts is a plain dict on the dataclass, so we can nest a sub-dict here without slots issues.
    fitted_store = ctx.artifacts.setdefault("feature_handling_fitted", {})
    fitted_store[cur_target_name] = result

    # TODO(wave-N): downstream consumption -- the assembled matrices in `result.train/val/test` are
    # currently fit-and-stash-only. Phase F (CB embedding_features) / phase G (TabularInputEncoder)
    # will route these into the model.fit() path. Until then the call exists to seed the per-suite
    # FeatureCache and exercise the validate_against_models guard so misconfig is caught at fit time.
    return result


def _train_one_target(ctx, target_type, targets, cur_target_name, cur_target_values):
    """Train all models for one (target_type, target_name) pair."""
    model_name = ctx.model_name
    target_name = ctx.target_name
    split_config = ctx.split_config
    hyperparams_config = ctx.hyperparams_config
    behavior_config = ctx.behavior_config
    reporting_config = ctx.reporting_config
    feature_selection_config = ctx.feature_selection_config
    baseline_diagnostics_config = ctx.baseline_diagnostics_config
    dummy_baselines_config = ctx.dummy_baselines_config
    quantile_regression_config = ctx.quantile_regression_config
    verbose = ctx.verbose
    linear_model_config = ctx.linear_model_config
    data_dir = ctx.data_dir
    models_dir = ctx.models_dir
    save_charts = ctx.save_charts
    outlier_detector = ctx.outlier_detector
    use_mrmr_fs = ctx.use_mrmr_fs
    use_ordinary_models = ctx.use_ordinary_models
    use_mlframe_ensembles = ctx.use_mlframe_ensembles
    mrmr_kwargs = ctx.mrmr_kwargs
    rfecv_models = ctx.rfecv_models
    multilabel_dispatch_config = ctx.multilabel_dispatch_config
    custom_pre_pipelines = ctx.custom_pre_pipelines
    common_params_dict = ctx.common_params_dict
    mlframe_models = ctx.mlframe_models
    metadata = ctx.metadata
    target_by_type = ctx.target_by_type
    group_ids = ctx.group_ids
    timestamps = ctx.timestamps
    sample_weights = ctx.sample_weights
    baseline_rss_mb = ctx.baseline_rss_mb
    df_size_mb = ctx.df_size_mb
    train_idx = ctx.train_idx
    test_idx = ctx.test_idx
    train_details = ctx.train_details
    val_details = ctx.val_details
    test_details = ctx.test_details
    fairness_subgroups = ctx.fairness_subgroups
    pipeline = ctx.pipeline
    polars_pipeline_applied = ctx.polars_pipeline_applied
    cat_features = ctx.cat_features
    text_features = ctx.text_features
    embedding_features = ctx.embedding_features
    _dropped_high_card_data = ctx._dropped_high_card_data
    train_df_pd = ctx.train_df_pd
    val_df_pd = ctx.val_df_pd
    test_df_pd = ctx.test_df_pd
    train_df_polars = ctx.train_df_polars
    val_df_polars = ctx.val_df_polars
    test_df_polars = ctx.test_df_polars
    filtered_train_df = ctx.filtered_train_df
    filtered_val_df = ctx.filtered_val_df
    filtered_train_idx = ctx.filtered_train_idx
    filtered_val_idx = ctx.filtered_val_idx
    train_od_idx = ctx.train_od_idx
    val_od_idx = ctx.val_od_idx
    category_encoder = ctx.category_encoder
    imputer = ctx.imputer
    scaler = ctx.scaler
    trainset_features_stats = ctx.trainset_features_stats
    defer_pandas_conv = ctx.defer_pandas_conv
    train_df_size_bytes_cached = ctx.train_df_size_bytes_cached
    val_df_size_bytes_cached = ctx.val_df_size_bytes_cached
    _all_target_audits = ctx._all_target_audits
    _non_neural_train_times = ctx._non_neural_train_times
    models = ctx.models
    slug_to_original_target_type = ctx.slug_to_original_target_type
    slug_to_original_target_name = ctx.slug_to_original_target_name
    # Identity assignment is intentional: keep the slug key registered even when it equals the original name,
    # so downstream lookups via slug never KeyError on round-trip identity targets.
    slug_to_original_target_name[slugify(cur_target_name)] = cur_target_name
    # Initialised pre-conditional so a later reference doesn't NameError when mlframe_models is empty.
    rfecv_models_params = {}
    if mlframe_models:
        plot_file, model_file = _setup_model_directories(
            target_name=target_name,
            model_name=model_name,
            target_type=target_type,
            cur_target_name=cur_target_name,
            data_dir=data_dir,
            models_dir=models_dir,
            save_charts=save_charts,
        )

        _train_idx = filtered_train_idx if filtered_train_idx is not None else train_idx
        current_train_target = (
            cur_target_values[_train_idx]
            if isinstance(cur_target_values, (np.ndarray, pl.Series))
            else cur_target_values.iloc[_train_idx]
        )
        current_val_target = None
        if filtered_val_idx is not None:
            current_val_target = (
                cur_target_values[filtered_val_idx]
                if isinstance(cur_target_values, (np.ndarray, pl.Series))
                else cur_target_values.iloc[filtered_val_idx]
            )
        # test_idx is intentionally raw (not OD-filtered) - test must never be filtered by outlier detector.
        current_test_target = None
        if test_idx is not None:
            current_test_target = (
                cur_target_values[test_idx]
                if isinstance(cur_target_values, (np.ndarray, pl.Series))
                else cur_target_values.iloc[test_idx]
            )

        # Feature-handling wire-in: opt-in via ctx.feature_handling_config. Sits after the per-target
        # OD-filtered frames + targets are bound (this is the "post-FS / pre-final-pipeline" seam for
        # the inner pre_pipelines x models loops below) and before per-target diagnostics so any
        # FHC-detected text columns surface in the same log block. No-op when fhc is None, so the
        # default code path is unchanged. polars-fastpath frames are preferred when present; the
        # underlying handlers detect polars vs pandas via _extract_column_values. A blanket
        # polars->pandas conversion here would defeat the suite's polars fastpath -- left to apply.py
        # to keep frame container as-given.
        _fhc_train_df = train_df_polars if train_df_polars is not None else filtered_train_df
        _fhc_val_df = val_df_polars if val_df_polars is not None else filtered_val_df
        _fhc_test_df = test_df_polars if test_df_polars is not None else test_df_pd
        _maybe_run_feature_handling_apply(
            ctx,
            cur_target_name=cur_target_name,
            train_df=_fhc_train_df,
            val_df=_fhc_val_df,
            test_df=_fhc_test_df,
            current_train_target=current_train_target,
            sample_weight=sample_weights,
        )

        metadata = run_per_target_diagnostics(
            target_type=target_type,
            cur_target_name=cur_target_name,
            current_train_target=current_train_target,
            current_val_target=current_val_target,
            current_test_target=current_test_target,
            filtered_train_df=filtered_train_df,
            baseline_diagnostics_config=baseline_diagnostics_config,
            cat_features=cat_features,
            metadata=metadata,
        )

        metadata = run_dummy_baselines(
            target_type=target_type,
            cur_target_name=cur_target_name,
            target_name=target_name,
            model_name=model_name,
            current_train_target=current_train_target,
            current_val_target=current_val_target,
            current_test_target=current_test_target,
            filtered_train_df=filtered_train_df,
            filtered_val_df=filtered_val_df,
            test_df_pd=test_df_pd,
            filtered_train_idx=filtered_train_idx,
            filtered_val_idx=filtered_val_idx,
            test_idx=test_idx,
            timestamps=timestamps,
            cat_features=cat_features,
            dummy_baselines_config=dummy_baselines_config,
            quantile_regression_config=quantile_regression_config,
            reporting_config=reporting_config,
            _dropped_high_card_data=_dropped_high_card_data,
            train_od_idx=train_od_idx,
            val_od_idx=val_od_idx,
            plot_file=plot_file,
            metadata=metadata,
            target_by_type=target_by_type,
            _split_preds_probs=_split_preds_probs,
        )

        # Audits are precomputed once for all targets via the batch API; this lookup is the per-target render.
        _audit = _all_target_audits.get(target_type, {}).get(cur_target_name)
        if _audit is not None:
            try:
                logger.info(_format_temporal_audit_report(_audit))
                if (getattr(behavior_config, "target_temporal_audit_save_plot", True)
                        and plot_file):
                    _plot_path = f"{plot_file}_target_temporal_audit.png"
                    _plot_target_over_time(_audit, save_path=_plot_path)
                metadata.setdefault("target_temporal_audit", {}) \
                    .setdefault(str(target_type), {})[cur_target_name] = _audit.to_dict()
            except Exception as _audit_err:
                logger.warning(
                    "target_temporal_audit (per-target render) failed for "
                    "target='%s': %s. Training continues.",
                    cur_target_name, _audit_err,
                )

        if verbose:
            logger.info(f"select_target...")

        t0_select_target = timer()
        od_common_params, current_behavior_config = _build_common_params_for_target(
            common_params_dict=common_params_dict,
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

        # Test set is never OD-filtered. train_df_size_bytes_cached is the pre-conversion Polars-side size
        # passed through so configure_training_params can skip a 3-min pandas memory_usage(deep=...) scan
        # on high-cardinality object columns; the OD-shrinkage approximation only feeds a GPU-RAM heuristic.
        common_params, models_params, rfecv_models_params, cpu_configs, gpu_configs = select_target(
            model_name=f"{target_name} {model_name} {cur_target_name}",
            target=cur_target_values,
            target_type=target_type,
            df=None,
            train_df=filtered_train_df,
            val_df=filtered_val_df,
            test_df=test_df_pd,
            train_idx=filtered_train_idx,
            val_idx=filtered_val_idx,
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
            train_df_size_bytes=train_df_size_bytes_cached,
            val_df_size_bytes=val_df_size_bytes_cached,
            multilabel_dispatch_config=multilabel_dispatch_config,
        )

    if verbose:
        logger.info("  select_target done in %s", _elapsed_str(t0_select_target))
        log_ram_usage()

    pre_pipelines, pre_pipeline_names = _build_pre_pipelines(
        use_ordinary_models=use_ordinary_models,
        rfecv_models=rfecv_models,
        rfecv_models_params=rfecv_models_params,
        use_mrmr_fs=use_mrmr_fs,
        mrmr_kwargs=mrmr_kwargs,
        custom_pre_pipelines=custom_pre_pipelines,
        rfecv_leakage_corr_threshold=feature_selection_config.rfecv_leakage_corr_threshold,
        rfecv_mbh_adaptive_threshold=feature_selection_config.rfecv_mbh_adaptive_threshold,
    )

    # Custom transformers run AFTER preprocessing, so the preprocessing output is shared across
    # pre_pipelines of the same model-type bucket; one cache instance covers the whole sweep.
    pipeline_cache = PipelineCache()

    for pre_pipeline, pre_pipeline_name in tqdmu_lazy_start(zip(pre_pipelines, pre_pipeline_names), desc="pre_pipeline", total=len(pre_pipelines)):
        # CatBoost + RFECV metamodel_func combination breaks sklearn.clone().
        if _should_skip_catboost_metamodel(pre_pipeline_name.strip(), target_type, behavior_config):
            continue

        # Skip identity-equivalent pre_pipelines: marker survives across targets, so a selector
        # that was a no-op on a prior target gets skipped here before any model trains.
        # Honour ``feature_selection_config.skip_identity_equivalent_pre_pipelines``: when False
        # the caller asked to retrain even on identity-equivalent pre_pipelines (e.g. for
        # ensembling-diversity-via-RNG-seed scenarios), so this early-exit must not fire.
        _pp_name_stripped = pre_pipeline_name.strip()
        if (
            _pp_name_stripped
            and feature_selection_config.skip_identity_equivalent_pre_pipelines
            and getattr(pre_pipeline, "_mlframe_identity_equivalent", False)
        ):
            logger.info(
                "[Dedup] Skipping pre_pipeline '%s' -- "
                "identity-equivalent to ordinary (cached from "
                "prior target/iteration); models already covered.",
                _pp_name_stripped,
            )
            continue
        ens_models = [] if use_mlframe_ensembles else None
        orig_pre_pipeline = pre_pipeline

        if sample_weights:
            weight_schemas = sample_weights
            if "uniform" in sample_weights:
                logger.info("Using %d weighting schema(s) from extractor: %s", len(weight_schemas), list(weight_schemas.keys()))
            else:
                logger.info("Using %d weighting schema(s) from extractor: %s. Note: uniform weighting not included.", len(weight_schemas), list(weight_schemas.keys()))
        else:
            weight_schemas = {"uniform": None}
            logger.info("No weighting schemas from extractor, defaulting to uniform weighting.")

        # Backward val placement + recency weighting cancel each other's drift-proxy intent
        # (val older than train, training biased to newest rows). Warn so the user picks one.
        _val_placement = getattr(split_config, "val_placement", "forward")
        if _val_placement == "backward":
            _non_uniform = [k for k in weight_schemas.keys() if k != "uniform"]
            if _non_uniform:
                logger.warning(
                    "  val_placement='backward' is combined with %d non-"
                    "uniform weighting schema(s) %s. Backward val is "
                    "designed to approximate DEPLOYMENT error under "
                    "drift by mirroring the val->train gap against the "
                    "train->prod gap, while recency-style weights bias "
                    "training toward the newest rows. Together they "
                    "optimise 'fit newest, validate on oldest' -- which "
                    "contradicts the drift-proxy intent of backward. "
                    "Consider disabling use_recency_weighting on the "
                    "extractor (runs will fall back to uniform only) "
                    "or switching back to val_placement='forward'.",
                    len(_non_uniform), _non_uniform,
                )

        # Models sorted by feature tier (richest first) so text/embedding columns are dropped once per tier.
        # Strategy lookup keyed by id() because estimators / tuples are not hashable, and identity-distinct
        # instances must stay distinct in the map. Pre-computed once per suite by setup_configuration;
        # reading off ctx here avoids the O(targets * pre_pipelines * models) re-evaluation that used to
        # rebuild this map per inner-loop iteration.
        strategy_by_model = ctx.strategy_by_model
        sorted_models = ctx.sorted_mlframe_models
        # Suite-scoped feature-side cache: tier_dfs / pl.Enum map / prepared polars frames carry
        # ACROSS targets (target-independent transforms) so only y / sample_weight differ inside
        # the inner loop. Both inner caches are scoped to the current ``pre_pipeline_name`` since
        # different pre_pipelines may keep different columns (MRMR / RFECV vs ordinary), and the
        # tier-DFs / Enum maps depend on the column set after pre-pipeline column trimming.
        _suite_feature_cache = _get_feature_side_cache(ctx)
        _per_pp_cache = _suite_feature_cache.setdefault(pre_pipeline_name, {})
        tier_dfs_cache: dict[tuple, dict[str, Any]] = _per_pp_cache.setdefault("tier_dfs", {})
        # Leak-free pl.Enum map built from train+val UNION only (test EXCLUDED to avoid label-time leakage).
        # Depends only on (feature_tier, strategy class) - target-independent so it carries cross-target.
        tier_enum_map_cache: dict[tuple, dict[str, Any] | None] = _per_pp_cache.setdefault("tier_enum_map", {})
        # Prepared polars frames + xgb_category_map per (tier, supports_polars, strategy_class).
        # Target-independent because _prep_polars_df / build_polars_enum_map do not touch y; the
        # text-features fill_null pass below is also target-independent. Carry cross-target.
        prepared_frames_cache: dict[tuple, dict[str, Any]] = _per_pp_cache.setdefault("prepared_frames", {})
        prev_tier = None

        # Neural max_time defaults to P95 of non-neural train times so MLP can't run 2h while boosters take 5min.
        # CODE-LOW-4: per-target reset is INTENTIONAL -- each target's neural budget is computed only from the
        # same target's non-neural runs, so an unusually fast/slow earlier target cannot widen or starve the
        # current target's neural budget. We rebind both the local AND ctx._non_neural_train_times to the
        # SAME fresh list so the writeback at end-of-function is a no-op (the dict the caller sees is the
        # one we just mutated) and downstream readers of ctx._non_neural_train_times observe the per-target
        # contents in-flight, not the previous target's tail.
        _non_neural_train_times = []
        ctx._non_neural_train_times = _non_neural_train_times

        _total_models_in_run = len(sorted_models)
        _model_idx_in_run = 0
        _break_model_loop = False
        for mlframe_model_name in tqdmu_lazy_start(sorted_models, desc="mlframe model"):
            if _should_skip_catboost_metamodel(mlframe_model_name, target_type, behavior_config):
                continue
            _model_idx_in_run += 1
            if verbose:
                # Per-model RSS sample is intentional: localising OOM-blame to a specific
                # model+target+pre_pipeline tuple in the verbose-suite log saves hours of post-mortem
                # log-correlation. The ~3ms/call Windows cost is dwarfed by per-model fit times.
                try:
                    import psutil as _ps
                    _ram_gb_now = _ps.Process().memory_info().rss / (1024 ** 3)
                except Exception:
                    _ram_gb_now = 0.0
                logger.info(
                    "  process_model(%s) START -- model %d/%d, RAM=%.1fGB",
                    mlframe_model_name,
                    _model_idx_in_run, _total_models_in_run,
                    _ram_gb_now,
                )

            if mlframe_model_name not in models_params:
                logger.warning(f"mlframe model {mlframe_model_name} not known, skipping...")
                continue

            # Cross-target dataset reuse: restore the prior target's _DATASET_REUSE_CACHE_ATTRS
            # snapshot onto the freshly-built model template BEFORE the weight loop's clone()
            # forward-transfer reads them. select_target() rebuilds models_params per target so
            # the cache attributes are absent on a virgin template - without this restore the
            # XGB/LGB shims would rebuild the binned dataset on target 2. The shim's
            # signature_of(X) check then matches against the same ctx-pinned train_df pointer
            # and triggers set_label / set_weight in place rather than a fresh build.
            _restore_dataset_reuse_cache(
                ctx, mlframe_model_name, models_params[mlframe_model_name]["model"],
            )

            strategy = strategy_by_model[id(mlframe_model_name)]

            # Drop pre-pipeline Polars originals as soon as we hit the first non-Polars strategy. The
            # post-iteration release fires only on tier transitions, but same-tier siblings (e.g. XGB and
            # LGB share tier=(False,False)) would keep Polars frames alive into a lazy pandas conversion,
            # doubling peak RAM. Releasing upfront halves peak in mixed suites.
            if (
                not strategy.supports_polars
                and train_df_polars is not None
            ):
                # Drop locals AND ctx attributes -- ctx still pins the strong ref to the same frames
                # assigned via ctx.*_df_polars at function entry, so a bare ``del`` of the locals would
                # leave maybe_clean_ram_and_gpu with nothing to reclaim and turn the log line into a lie.
                del train_df_polars, val_df_polars, test_df_polars
                train_df_polars = val_df_polars = test_df_polars = None
                # Drop polars-tier entries only - pandas-tier entries hang on the SAME cache dicts
                # (these locals now reference suite-scoped dicts in _per_pp_cache) and must survive
                # the release. ``_invalidate_polars_feature_side_cache(ctx)`` runs further down the
                # _release_ctx_polars_frames path and does the same for the prepared_frames sub-
                # cache; the tier_dfs / tier_enum_map dicts that pre-date this hoist are scrubbed
                # here so a same-target pandas-tier sibling reads a clean enum-map slot.
                for _pl_only_key in [_k for _k in tier_dfs_cache if isinstance(_k, tuple) and len(_k) >= 2 and _k[1] == "pl"]:
                    tier_dfs_cache.pop(_pl_only_key, None)
                tier_enum_map_cache.clear()  # All entries are polars-only (populated only on the polars fastpath).
                baseline_rss_mb = _release_ctx_polars_frames(
                    ctx,
                    baseline_rss_mb,
                    df_size_mb,
                    verbose=verbose,
                    reason="non-polars-native strategy entry",
                )
                if verbose:
                    logger.info(
                        "  Released pre-pipeline Polars originals before %s (non-polars-native strategy).",
                        mlframe_model_name,
                    )

            # Clone the base_pipeline per model so each iteration gets a fresh, un-fitted selector. Sharing a
            # fitted MRMR/RFECV across strategies caused `_is_fitted` to misreport True for a partially-fit
            # pipeline (selector fitted but encoder/imputer/scaler not), tripping imputer.transform on a
            # feature-names mismatch.
            _base_for_strategy = orig_pre_pipeline
            if _base_for_strategy is not None:
                try:
                    _base_for_strategy = clone(_base_for_strategy)
                except Exception:
                    # Non-BaseEstimator custom pipelines don't clone; keep the original reference.
                    pass
            pre_pipeline = strategy.build_pipeline(
                base_pipeline=_base_for_strategy,
                cat_features=cat_features,
                category_encoder=category_encoder if cat_features else None,
                imputer=imputer,
                scaler=scaler,
            )
            # Cache key = strategy.cache_key + pre_pipeline_name + feature_tier + container kind + feature-list digest.
            # feature_tier is required because CB/LGB/XGB all share cache_key="tree" but have different
            # tiers; without it, CB's text/embedding-bearing frame would be served to LGB/XGB.
            # Kind suffix prevents Polars-native (XGB) and pandas-only (LGB) consumers from sharing entries
            # within a tier, which would otherwise undo the lazy pandas conversion downstream.
            # See _compute_pipeline_cache_key for the features-digest contract (frozenset, order-invariant).
            cache_key = _compute_pipeline_cache_key(
                strategy.cache_key,
                pre_pipeline_name,
                strategy.feature_tier(),
                strategy.supports_polars,
                cat_features,
                text_features,
                embedding_features,
            )

            # Polars fastpath substitutes original Polars DataFrames for natively-Polars consumers
            # (CatBoost >= 1.2.7, HGB). Polars DFs are prepared once per model (outside the weight loop)
            # because prepare_polars_dataframe() allocates via .with_columns().
            polars_fastpath_active = train_df_polars is not None and strategy.supports_polars

            if polars_fastpath_active:
                if verbose:
                    logger.info("  Polars fastpath active for %s (strategy=%s)", mlframe_model_name, type(strategy).__name__)
                # MUST use the post-promotion `cat_features` (Phase 3.5 reassignment), NOT the stale
                # `cat_features_polars` snapshot from before auto-detect ran - the latter would still list
                # text-promoted columns and trip CB's polars-categorical fastpath on String dtypes.
                _cat_features = list(cat_features or [])

                # Cross-target reuse: cache key is (feature_tier, supports_polars=True, strategy_class,
                # cb_text_pass) where cb_text_pass tracks whether the CB-only Categorical->String text-
                # column cast must be applied (CB requires it; other CB-tier polars-native models don't).
                # All target-independent so the prepared frames carry from target 1 to target N.
                _prep_key = (
                    strategy.feature_tier(),
                    True,
                    type(strategy).__name__,
                    bool(text_features and mlframe_model_name == "cb"),
                )
                _cached_prep = prepared_frames_cache.get(_prep_key)
                if _cached_prep is not None:
                    prepared_train = _cached_prep["prepared_train"]
                    prepared_val = _cached_prep["prepared_val"]
                    prepared_test = _cached_prep["prepared_test"]
                    _xgb_category_map = _cached_prep["xgb_category_map"]
                    if verbose:
                        logger.info(
                            "  feature-side cache hit for %s (strategy=%s, pp=%s): reusing prepared polars frames across targets",
                            mlframe_model_name, type(strategy).__name__, pre_pipeline_name or "<ordinary>",
                        )
                else:
                    tier_base = {
                        "train_df": train_df_polars,
                        "val_df": val_df_polars,
                        "test_df": test_df_polars,
                    }
                    tier_polars = _build_tier_dfs(
                        tier_base, strategy, text_features, embedding_features, tier_dfs_cache, verbose=verbose,
                    )

                    # Enum map: leak-free, train+val union only; cached by (tier, strategy class).
                    _enum_cache_key = (strategy.feature_tier(), type(strategy).__name__)
                    if _enum_cache_key in tier_enum_map_cache:
                        _xgb_category_map = tier_enum_map_cache[_enum_cache_key]
                    elif hasattr(strategy, "build_polars_enum_map"):
                        try:
                            _xgb_category_map = strategy.build_polars_enum_map(
                                tier_polars["train_df"],
                                tier_polars.get("val_df"),
                                _cat_features,
                            )
                        except Exception as _emb_exc:
                            logger.warning(
                                "build_polars_enum_map failed for %s; "
                                "falling back to per-DF Enum cast: %s",
                                type(strategy).__name__, _emb_exc,
                            )
                            _xgb_category_map = None
                        tier_enum_map_cache[_enum_cache_key] = _xgb_category_map
                    else:
                        _xgb_category_map = None
                        tier_enum_map_cache[_enum_cache_key] = None

                    prepared_train = _prep_polars_df(tier_polars["train_df"], strategy, _cat_features, _xgb_category_map)
                    prepared_val = _prep_polars_df(tier_polars.get("val_df"), strategy, _cat_features, _xgb_category_map)
                    prepared_test = _prep_polars_df(tier_polars.get("test_df"), strategy, _cat_features, _xgb_category_map)

                    # CatBoost's polars text-features path requires plain String with no nulls; cast Categorical/Enum
                    # text columns and fill_null. The dtype mismatch happens whenever auto-detect promotes a
                    # column from cat_features to text_features without changing its backing dtype.
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
                                    "  Cast %d text feature(s) from Polars Categorical to String "
                                    "for CatBoost: %s",
                                    len(needs_cast), needs_cast,
                                )

                    # Null-in-Categorical fill is applied upstream once on train_df_polars/val/test (search:
                    # `_polars_fill_null_in_categorical`, marker "__MISSING__"); no per-model fill needed.

                    # Store REFERENCES only (no clones / no copies): a 100GB train_df_polars is shared
                    # with ctx.train_df_polars; the prepared variant is a polars LazyFrame-evaluation
                    # result that's already eager but immutable in our path. Carrying across targets
                    # costs ~one pointer per slot - never duplicates feature data.
                    prepared_frames_cache[_prep_key] = {
                        "prepared_train": prepared_train,
                        "prepared_val": prepared_val,
                        "prepared_test": prepared_test,
                        "xgb_category_map": _xgb_category_map,
                    }

            else:

                # Lazy pandas conversion for non-Polars-native strategies. The upfront _convert_dfs_to_pandas
                # is skipped when all blockers are non-native; per-strategy conversion happens here, which
                # preserves RAM when CB/XGB can run natively on polars. Two trigger cases get distinct log
                # messages: (a) strategy genuinely non-Polars-native; (b) strategy IS native but polars
                # originals were released earlier in the run.
                # CONV-MED-5: cache the polars->pandas view by id() of the source frame on ctx so two
                # non-Polars-native strategies sharing the same source polars frame pay one conversion
                # total, not one per strategy.
                _logged_lazy_conv = False
                _view_cache = ctx._pandas_view_cache
                for df_key in ("train_df", "val_df", "test_df"):
                    df_ = common_params.get(df_key)
                    if isinstance(df_, pl.DataFrame):
                        if not _logged_lazy_conv and verbose:
                            if strategy.supports_polars:
                                _reason = (
                                    "Polars originals released "
                                    "(common_params still carries "
                                    "polars frames; converting to "
                                    "pandas for inner predict path)"
                                )
                            else:
                                _reason = (
                                    f"non-Polars-native strategy "
                                    f"{type(strategy).__name__}"
                                )
                            logger.info(
                                "  Lazy pandas conversion for %s -- %s",
                                mlframe_model_name, _reason,
                            )
                            _logged_lazy_conv = True
                        _src_id = id(df_)
                        _pd_view = _view_cache.get(_src_id)
                        if _pd_view is None:
                            _pd_view = get_pandas_view_of_polars_df(df_)
                            _view_cache[_src_id] = _pd_view
                        common_params[df_key] = _pd_view

                # Defense-in-depth: after lazy conversion, every common_params DF must be non-polars.
                # Surfacing here (rather than at trainer.fit time) makes the cross-iteration leakage cause
                # visible with full strategy/common_params context.
                for df_key in ("train_df", "val_df", "test_df"):
                    df_ = common_params.get(df_key)
                    if isinstance(df_, pl.DataFrame):
                        raise RuntimeError(
                            f"Lazy pandas conversion produced incomplete "
                            f"state for non-Polars-native strategy "
                            f"{type(strategy).__name__} ({mlframe_model_name}): "
                            f"common_params[{df_key!r}] is still pl.DataFrame "
                            f"(shape={df_.shape}, id={id(df_)}). The lazy-"
                            f"conversion hook iterated over train/val/test but "
                            f"this key escaped. Likely cause: a ``common_params`` "
                            f"override between lazy-conversion and here, or "
                            f"pipeline_cache cross-stream leakage (see core.py "
                            f"kind-suffix in cache_key)."
                        )

                tier_pandas = _build_tier_dfs(
                    {"train_df": common_params.get("train_df"), "val_df": common_params.get("val_df"), "test_df": common_params.get("test_df")},
                    strategy, text_features, embedding_features, tier_dfs_cache, verbose=verbose,
                )

            # CODE-P1-10: compute input-schema fingerprint ONCE per (model, pre_pipeline) outside the
            # weight loop. The fingerprinted train_df is the same across all weight schemas (only
            # sample_weight changes inside the weight loop), so the previous per-iteration call was
            # pure waste. Cache key is purely feature-side (strategy+tier+kind+pp_name) - dropping
            # ``target_type`` / ``cur_target_name`` from the key was the per-target hoist: the
            # schema hash depends on column names/dtypes, NOT on y, so target N reuses target 1's
            # fingerprint without recomputation. Audit-checked vs compute_model_input_fingerprint:
            # signature takes train_df + cat/text/embedding_features only, no target.
            _fp_train_df_pre = prepared_train if polars_fastpath_active else tier_pandas["train_df"]
            _fp_cache_key = (
                id(strategy),
                strategy.feature_tier(),
                strategy.supports_polars,
                pre_pipeline_name,
            )
            if _fp_cache_key in ctx._model_input_fingerprint_cache:
                _schema_hash, _input_schema = ctx._model_input_fingerprint_cache[_fp_cache_key]
            else:
                _schema_hash, _input_schema = compute_model_input_fingerprint(
                    _fp_train_df_pre,
                    cat_features=cat_features,
                    text_features=text_features,
                    embedding_features=embedding_features,
                )
                ctx._model_input_fingerprint_cache[_fp_cache_key] = (_schema_hash, _input_schema)

            for weight_name, weight_values in tqdmu_lazy_start(weight_schemas.items(), desc="weighting schema"):
                model_name_with_weight = common_params["model_name"]
                model_file_name=f"{mlframe_model_name}"
                if weight_name != "uniform":
                    model_name_with_weight += f" w={weight_name}"
                    model_file_name +=f"_{weight_name}"

                # Isolation copy: per-(model, weight) inner mutations (sample_weight, plot_file
                # decoration, lazy pandas conversion, fastpath frame swap) must not bleed into
                # the outer ``common_params`` template that the next iteration consumes. The
                # 4-deep nesting (target_type x target x pre_pipeline x model x weight) has been
                # verified across the suite -- removing the copy regresses the cross-weight
                # contamination tests. Do NOT inline.
                current_common_params = common_params.copy()
                current_common_params["sample_weight"] = weight_values

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
                if getattr(behavior_config, "model_file_hash_suffix", True):
                    model_file_name += f"__sch_{_schema_hash}"

                if weight_name != "uniform" and current_common_params.get("plot_file"):
                    current_common_params["plot_file"] = current_common_params["plot_file"] + weight_name + "_"

                cached_dfs = pipeline_cache.get(cache_key)

                # INTENTIONAL: clone() lives INSIDE the weight loop. Each weight schema produces a
                # different trained model stored separately in models[type][target]; without per-iteration
                # cloning all in-memory entries would alias to the same last-trained sklearn object and
                # only the .dump snapshots would be correct. Do NOT move clone() outside the loop.
                original_model = models_params[mlframe_model_name]["model"]
                try:
                    cloned_model = clone(original_model)
                except RuntimeError:
                    # CatBoost wraps custom eval_metric objects internally; sklearn's identity check fails.
                    # Direct constructor call with get_params() produces an equivalent unfitted instance.
                    cloned_model = type(original_model)(**original_model.get_params())
                except TypeError:
                    # NGBoost: get_params() exposes attributes the constructor doesn't accept.
                    _cls = type(original_model)
                    _sig_params = set(inspect.signature(_cls.__init__).parameters) - {"self"}
                    _raw = original_model.get_params(deep=False)
                    cloned_model = _cls(**{k: v for k, v in _raw.items() if k in _sig_params})
                # sklearn.clone() strips non-param attributes; re-assert mlframe sticky flags so the
                # calibration directive and the polars-fastpath-broken marker survive each iteration.
                if getattr(original_model, "_mlframe_posthoc_calibrate", False):
                    try:
                        cloned_model._mlframe_posthoc_calibrate = True
                    except Exception as _attr_err:
                        logger.debug("Could not set _mlframe_posthoc_calibrate on clone: %s", _attr_err)
                if getattr(original_model, "_mlframe_polars_fastpath_broken", False):
                    try:
                        cloned_model._mlframe_polars_fastpath_broken = True
                    except Exception as _attr_err:
                        logger.debug("Could not set _mlframe_polars_fastpath_broken on clone: %s", _attr_err)
                # Hand the XGB DMatrix / LGB Dataset reuse caches forward across clone() so the
                # weight-schema loop (uniform -> recency on the same train_df) reuses the heavy binned
                # dataset in place via set_label / set_weight instead of rebuilding.
                _forward_dataset_reuse_cache(original_model, cloned_model)
                # Isolation copy: each weight iteration installs its own cloned_model and may
                # patch fit_params (CatBoost text/embedding fastpath); without copying we would
                # mutate the suite-level models_params template and the next target would inherit
                # this iteration's overrides.
                current_model_params = models_params[mlframe_model_name].copy()
                current_model_params["model"] = cloned_model

                # CatBoost is the only Polars-native consumer that accepts cat_features / text_features /
                # embedding_features at fit time; XGB and HGB auto-detect via enable_categorical=True.
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
                    # Per-strategy decision on whether preprocessing for this strategy is already done.
                    # Two sufficient conditions:
                    #   (1) the suite-level polars-ds pipeline ran AND this strategy consumes polars natively;
                    #   (2) the polars fastpath is active for this strategy (its frame is the polars native
                    #       one, so sklearn encoder/scaler/imputer would be redundant and crash anyway).
                    # Note: requires_encoding=True is NOT a re-run trigger (HGB declares it for pandas-fallback
                    # only; on the polars fastpath HGB consumes pl.Categorical natively). Only non-Polars
                    # strategies fall through to their own pre_pipeline run in trainer.py.
                    polars_pipeline_applied=(
                        (polars_pipeline_applied and strategy.supports_polars)
                        or polars_fastpath_active
                    ),
                    mlframe_model_name=mlframe_model_name,
                    metadata_columns=metadata.get("columns"),
                )

                _is_neural = is_neural_model(mlframe_model_name)
                _timeout = _compute_neural_max_time(_non_neural_train_times) if _is_neural else None
                if _timeout is not None:
                    _max_time_dict, _p95, _n = _timeout
                    # Reach into Pipeline(StandardScaler, TTR(PytorchLightningRegressor(...))) to find trainer_params.
                    _neural_model = current_model_params.get("model")
                    if _neural_model is not None:
                        _inner = getattr(_neural_model, "regressor", None)
                        if _inner is None and hasattr(_neural_model, "named_steps"):
                            for _step in _neural_model.named_steps.values():
                                if hasattr(_step, "regressor"):
                                    _inner = _step.regressor
                                    break
                        if _inner is not None and hasattr(_inner, "trainer_params"):
                            _inner.trainer_params["max_time"] = _max_time_dict
                            if verbose:
                                logger.info(
                                    "  [NeuralTimeout] %s max_time=%dh%02dm%02ds "
                                    "(P95 of %d prior non-neural train times: %.0fs)",
                                    mlframe_model_name,
                                    _max_time_dict["hours"], _max_time_dict["minutes"], _max_time_dict["seconds"],
                                    _n, _p95,
                                )

                t0_model = timer()
                try:
                    with phase("process_model", model=mlframe_model_name, weight=weight_name):
                        trainset_features_stats, pre_pipeline, train_df_transformed, val_df_transformed, test_df_transformed = process_model(
                            **process_model_kwargs
                        )
                except Exception as model_err:
                    # Skip-and-continue is opt-in. KeyboardInterrupt is intentionally not caught here;
                    # native SIGSEGV that kills the process won't be caught either.
                    if not behavior_config.continue_on_model_failure:
                        raise
                    logger.error(
                        f"  process_model({mlframe_model_name}, w={weight_name}) FAILED after "
                        f"{_elapsed_str(t0_model)} -- {type(model_err).__name__}: {model_err}. "
                        f"continue_on_model_failure=True -> skipping and moving on.",
                        exc_info=True,
                    )
                    metadata.setdefault("failed_models", []).append({
                        "model": mlframe_model_name,
                        "weighting": weight_name,
                        "error_type": type(model_err).__name__,
                        "error_message": str(model_err),
                    })
                    continue  # next weight_name in the inner loop
                if verbose:
                    logger.info("  process_model(%s, w=%s) done -- %s", mlframe_model_name, weight_name, _elapsed_str(t0_model))
                if not _is_neural and t0_model is not None:
                    _non_neural_train_times.append(timer() - t0_model)

                # After the first model trains, if the pre_pipeline is identity-equivalent (kept all
                # columns) AND the ordinary branch is in the suite, the remaining models would see
                # identical data - skip them.
                if (
                    _model_idx_in_run == 1
                    and _pp_name_stripped
                    and use_ordinary_models
                    and feature_selection_config.skip_identity_equivalent_pre_pipelines
                    and getattr(
                        pre_pipeline, "_mlframe_identity_equivalent", False
                    )
                ):
                    _skip_remaining = _total_models_in_run - 1
                    if _skip_remaining > 0:
                        logger.info(
                            "[Dedup] pre_pipeline '%s' is "
                            "identity-equivalent to ordinary (kept "
                            "all %d columns); skipping remaining "
                            "%d model(s) for this target.",
                            _pp_name_stripped,
                            train_df_transformed.shape[1]
                            if train_df_transformed is not None
                            else 0,
                            _skip_remaining,
                        )
                    _break_model_loop = True
                    break  # exit weight_schema loop

                # Hand the dataset-reuse cache from cloned_model back to the template so the next
                # weight-schema iteration's clone() carries it forward (symmetric to the forward-transfer
                # block above). Without this the cache would be born and die in a single iteration.
                _forward_dataset_reuse_cache(cloned_model, original_model, skip_none=True)

                # Persist this model's input-schema fingerprint in metadata so load-time can verify it
                # against the serving frame. Multi-output extensions (target_type / n_classes /
                # multilabel_strategy + schema_version) let load_mlframe_suite dispatch correctly;
                # legacy artifacts without these fields fall back to binary inference.
                _record = {
                    "schema_hash": _schema_hash,
                    "input_schema": _input_schema,
                    "mlframe_model": mlframe_model_name,
                    "weight_name": weight_name,
                    "target_type": str(target_type) if target_type is not None else None,
                    "schema_version": 2,  # 1=legacy, 2=multi-output-aware
                }
                train_y = (
                    cur_target_values[_train_idx]
                    if isinstance(cur_target_values, (np.ndarray, pl.Series))
                    else cur_target_values.iloc[_train_idx]
                )
                try:
                    from ..configs import TargetTypes
                    if target_type == TargetTypes.MULTILABEL_CLASSIFICATION:
                        _record["n_classes"] = (
                            int(train_y.shape[1])
                            if hasattr(train_y, "shape") and train_y.ndim == 2
                            else None
                        )
                        _record["multilabel_strategy"] = "native" if (
                            hasattr(strategy, "supports_native_multilabel") and strategy.supports_native_multilabel
                        ) else "wrapper"
                    elif target_type == TargetTypes.MULTICLASS_CLASSIFICATION:
                        _record["n_classes"] = (
                            int(len(np.unique(np.asarray(train_y))))
                            if hasattr(train_y, "shape") else None
                        )
                        _record["multilabel_strategy"] = None
                    else:
                        _record["n_classes"] = None
                        _record["multilabel_strategy"] = None
                except Exception as _intro_err:
                    # Never fail the metadata write because of an introspection error on optional fields.
                    # Surface as warning since load_mlframe_suite dispatches on n_classes/multilabel_strategy.
                    logger.warning("n_classes/multilabel_strategy introspection failed for %s: %s", mlframe_model_name, _intro_err)
                metadata.setdefault("model_schemas", {})[model_file_name] = _record

                if cached_dfs is None:
                    pipeline_cache.set(cache_key, train_df_transformed, val_df_transformed, test_df_transformed)

            # Preserve a fitted feature-selector across same-bucket tree iterations. Tree strategies return
            # just the base_pipeline from build_pipeline(); non-tree strategies wrap it in a full Pipeline
            # (encoder/imputer/scaler), which we do NOT want to reuse as the base for other model types.
            if cache_key.startswith("tree"):
                orig_pre_pipeline = pre_pipeline

            if _break_model_loop:
                break

            # Release dataset-reuse caches at strategy-iter end. Both shims park the heavy binned dataset
            # on ``_cached_train_*`` / ``_cached_val_*`` as a weight-schema-loop scratchpad; nothing
            # downstream reads them (.predict goes through _Booster, ensemble uses pre-computed probs,
            # save strips via __getstate__). Releasing here frees ~30% of peak RAM between strategies.
            # Capture the binned-dataset references off the template BEFORE clearing so the next
            # target's _restore_dataset_reuse_cache can re-attach them. Without this snapshot the
            # clear below frees the dataset and the cross-target hoist degrades to a no-op (same
            # behaviour as before the hoist). Storing references only - the binned dataset is
            # shared with whatever held it before; the clear merely drops the template's pointer.
            _capture_dataset_reuse_cache(ctx, mlframe_model_name, original_model)
            _maybe_clear_shim_cache(original_model)
            # ens_models snapshots may also hold the cache by reference (forward-transfer at clone() copied
            # the reference rather than moving it); release on each so the binned dataset can be freed.
            if ens_models:
                for _ens_ns in ens_models:
                    _maybe_clear_shim_cache(getattr(_ens_ns, "model", None))

            # On a tier transition into a non-Polars strategy, release the pre-pipeline Polars originals.
            cur_tier = strategy.feature_tier()
            if prev_tier is not None and cur_tier != prev_tier and not strategy.supports_polars:
                if train_df_polars is not None:
                    # Same rationale as the entry-site release: locals AND ctx attributes must both drop their refs.
                    del train_df_polars, val_df_polars, test_df_polars
                    train_df_polars = val_df_polars = test_df_polars = None
                    # Selective drop: see same-shape comment at the non-polars-native entry site
                    # above for rationale. These cache references are suite-scoped now, so a blanket
                    # .clear() would also wipe pandas-tier entries that survived the polars release.
                    for _pl_only_key in [_k for _k in tier_dfs_cache if isinstance(_k, tuple) and len(_k) >= 2 and _k[1] == "pl"]:
                        tier_dfs_cache.pop(_pl_only_key, None)
                    tier_enum_map_cache.clear()
                    baseline_rss_mb = _release_ctx_polars_frames(ctx, baseline_rss_mb, df_size_mb, verbose=verbose, reason="tier transition")
                    if verbose:
                        logger.info("  Released pre-pipeline Polars originals (tier transition)")
            prev_tier = cur_tier

        if ens_models and len(ens_models) > 1:
            if verbose:
                logger.info(f"evaluating simple ensembles...")
            ens_n_features = train_df_transformed.shape[1] if train_df_transformed is not None else None
            # Name the ensemble by its members so log grep shows which models actually participated;
            # cap to 4 to keep headers readable. short_model_tag strips internal shim suffixes
            # (WithDMatrixReuse / WithDatasetReuse) so the tag is the bare family name.
            from .._format import short_model_tag as _short_tag_fn
            _member_tags = [_short_tag_fn(getattr(m, "model", m)) for m in ens_models]
            if len(_member_tags) <= 4:
                _members_label = "[" + "+".join(_member_tags) + "]"
            else:
                _members_label = f"[N={len(_member_tags)}]"
            # confidence_ensemble_quantile=0.0 disables the Conf Ensemble output entirely.
            _conf_q = float(getattr(behavior_config, "confidence_ensemble_quantile", 0.1))
            _ensembles = score_ensemble(
                models_and_predictions=ens_models,
                ensemble_name=f"{pre_pipeline_name}{_members_label} ",
                n_features=ens_n_features,
                uncertainty_quantile=_conf_q,
                **common_params,
            )
            # Persist the ensemble outputs so finalize_suite can serialise them and downstream
            # consumers (predict, reporting) see them. Pre-fix this return value was bound to a
            # local that nothing read, silently discarding every ensemble model the suite built.
            if _ensembles:
                ctx.ensembles.setdefault(target_type, {})[cur_target_name] = _ensembles
                # Mirror into the per-target model list (same slot the per-family training loop
                # uses) so any code iterating ``models[target_type][target_name]`` picks the
                # ensembles up without needing a separate dispatch.
                _target_models = models.setdefault(target_type, {}).setdefault(cur_target_name, [])
                for _ens_method, _ens_result in _ensembles.items():
                    _target_models.append(_ens_result)

    ctx.models = models
    ctx.metadata = metadata
    ctx.trainset_features_stats = trainset_features_stats
    # CODE-LOW-2 + CODE-LOW-4: slug_to_original_target_{type,name} and _non_neural_train_times
    # are mutable containers we already rebound on ctx (the slugs are bound by reference at the top
    # of this function and mutated in place; _non_neural_train_times is rebound to a fresh list each
    # target with a matching ``ctx._non_neural_train_times = _non_neural_train_times`` at that point).
    # The earlier writeback of these three was a no-op.
    ctx.train_df_polars = train_df_polars
    ctx.val_df_polars = val_df_polars
    ctx.test_df_polars = test_df_polars
    ctx.train_df_pd = train_df_pd
    ctx.val_df_pd = val_df_pd
    ctx.test_df_pd = test_df_pd
    ctx.filtered_train_df = filtered_train_df
    ctx.filtered_val_df = filtered_val_df
    ctx.pipeline = pipeline
    ctx.defer_pandas_conv = defer_pandas_conv
    ctx.baseline_rss_mb = baseline_rss_mb
    ctx.train_df_size_bytes_cached = train_df_size_bytes_cached
    ctx.val_df_size_bytes_cached = val_df_size_bytes_cached
