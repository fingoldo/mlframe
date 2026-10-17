"""
Composite-target discovery (default OFF, auto-enabled per suite when a regression target shows a
heavy-tail/skewed-target pathology and the caller left ``enabled`` at its default -- see
``_maybe_auto_enable_discovery``; an explicit ``enabled=False`` is always respected).

Runs after outlier detection, before per-target training. Each discovered spec
becomes one entry in ``target_by_type[regression]``; specs are stored on
``metadata["composite_target_specs"]`` for downstream inversion at predict time.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..baselines import BaselineDiagnostics
from ..configs import TargetTypes
from ._ar_skip import (  # noqa: F401  (_extreme_ar_discovery_skip / _recompute_lag1_ar_per_group stay importable from this module)
    _RERANK_SKIP_RATIO_BOUNDED_DEFAULT,
    _extreme_ar_discovery_skip,
    _recompute_lag1_ar_per_group,
    _zoo_is_bounded_only,
)
from .utils import (
    _build_full_column_from_splits,
    _defensive_copy_and_expand_multilabel_regression,
    _init_composite_discovery_metadata,
)
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

from ._target_slots import insert_composite_targets
from ._phase_composite_discovery_helpers import (  # noqa: F401  -- carved helpers
    _render_composite_discovery_diagnostics,
    _build_disc_df_for_target,
    _discovery_config_signature,
)

from ._phase_composite_discovery_gates import (  # noqa: F401  (re-exported)
    _AUTO_ENABLE_DISCOVERY_PATHOLOGY_PREFIXES,
    _DEFAULT_MIN_HONEST_GAIN_Z,
    _drop_below_honest_gain_floor,
    _relative_gain_se,
    _target_pathologies_for_auto_enable,
    _maybe_auto_enable_discovery,
    _maybe_narrow_to_unary_transforms,
    _per_target_discovery_config,
    _drop_specs_whose_bases_the_suite_cannot_materialise,
    _discovery_cache_lookup,
    discovery_inputs_digest,
    rank_pending_composites,
    select_composites_to_train,
)
from mlframe.training.composite.transforms.shared import call_transform
from ._phase_composite_mix_offset import _add_mix_offset_targets
from ._phase_composite_discovery_target import DiscoveryTargetCtx, discover_one_target


def composite_t_full(transform: Any, spec: Any, y_arr: np.ndarray, base_full: np.ndarray, filtered_train_idx: Any) -> "tuple[np.ndarray, np.ndarray]":
    """``(T for every row, T before imputation)`` of one composite spec, with its train-fitted parameters.

    A row where ``y`` is valid but the base violates the transform's domain gets the train median of T, so val/test
    rows still have a value to score. A row where ``y`` itself is missing keeps a missing T: filling it made up a
    composite label the per-target loop would then train on.
    """
    valid = transform.domain_check(y_arr, base_full)
    t_full = np.full(y_arr.shape[0], np.nan, dtype=np.float64)
    if valid.any():
        # Multi-base specs pass the row-filtered 2-D base slice; single-base stays 1-D.
        base_for_forward = base_full[valid, :] if base_full.ndim == 2 else base_full[valid]
        t_full[valid] = call_transform(transform, "forward", y_arr[valid], base_for_forward, spec.fitted_params)
    t_raw = t_full.copy()  # un-imputed: charts and the dedup must not see the fill value
    if not np.all(np.isfinite(t_full)):
        t_train = t_full[filtered_train_idx]
        t_train = t_train[np.isfinite(t_train)]
        if t_train.size > 0:
            t_full[~np.isfinite(t_full)] = float(np.median(t_train))
        t_full[~np.isfinite(np.asarray(y_arr, dtype=np.float64))] = np.nan
    return t_full, t_raw


def run_composite_target_discovery(
    *,
    composite_target_discovery_config: Any,
    target_by_type: dict,
    mlframe_models: list[str] | None,
    metadata: dict,
    filtered_train_df: Any,
    filtered_train_idx: Any,
    train_df_pd: Any,
    val_df_pd: Any,
    test_df_pd: Any,
    train_idx: Any,
    val_idx: Any,
    test_idx: Any,
    baseline_diagnostics_config: Any,
    cat_features: list[str] | None,
    verbose: bool,
    discovery_cache_dir: Any = None,
    group_ids: Any = None,
    split_config: Any = None,
    data_dir: Any = None,
    save_charts: bool = False,
    precomputed_specs: dict | None = None,
    mix_offset_inputs: dict | None = None,
) -> tuple[dict, dict]:  # unused-ok: verbose: uniform suite-phase signature; logging goes through logger levels
    """Run composite-target discovery for regression targets.

    Defensive: target_by_type is shallow-copied before adding entries so the
    FTE-returned original is never mutated in-place.

    ``mix_offset_inputs`` (optional, default None = off): ``{"X": full frame, "train_idx": training rows of the full frame, "selector": MRMR with fe_target_mix_enable, "gate_regressor": the model the held-out decision is made for}``.
    When given, a regression target whose feature-base mix the gate accepts gets a companion target ``<name>__mixoff`` with values
    ``y - beta * u(X)``, built from the training rows only (``build_mix_offset_target``).

    Returns updated (target_by_type, metadata).
    """
    if mix_offset_inputs is not None and TargetTypes.REGRESSION in target_by_type:
        target_by_type = _add_mix_offset_targets(target_by_type, metadata, mix_offset_inputs)
    _disc_feature_cols: Any = None
    if TargetTypes.REGRESSION in target_by_type:
        composite_target_discovery_config = _maybe_auto_enable_discovery(
            composite_target_discovery_config, target_by_type=target_by_type, train_idx=train_idx, metadata=metadata,
        )
        # The auto-enabled copy is local to this phase, while post-processing is handed the caller's original config and
        # would read enabled=False there: the cross-target ensemble and the lag failsafe would be skipped for exactly the
        # heavy-tail targets auto-enable exists for. Publish the effective decision so those gates see it.
        metadata["composite_discovery_effective_enabled"] = bool(composite_target_discovery_config.enabled)

    _gpu_families, _kept_spec_total = _init_composite_discovery_metadata(
        composite_target_discovery_config=composite_target_discovery_config,
        target_by_type=target_by_type,
        mlframe_models=mlframe_models,
        metadata=metadata,
    )
    # Caller-supplied specs are an explicit request to train those targets, so they run the replay even when discovery
    # itself is off: the reuse path exists precisely to skip the search, not the training.
    if not ((composite_target_discovery_config.enabled or precomputed_specs) and TargetTypes.REGRESSION in target_by_type):
        return target_by_type, metadata

    target_by_type = _defensive_copy_and_expand_multilabel_regression(
        target_by_type=target_by_type,
        composite_target_discovery_config=composite_target_discovery_config,
        metadata=metadata,
    )

    try:
        _disc_feature_cols = list(filtered_train_df.columns)
    except Exception as e:
        logger.debug("reading filtered_train_df.columns failed, using train_df_pd.columns: %s", e)
        _disc_feature_cols = list(train_df_pd.columns)

    # filtered_train_df is row-aligned to filtered_train_idx, so discovery sees a contiguous range.
    _disc_train_idx = np.arange(len(filtered_train_df))

    _auto_skip = bool(getattr(
        composite_target_discovery_config,
        "auto_skip_on_baseline_optimal", False,
    ))
    _existing_diags = metadata.get("baseline_diagnostics", {})

    # Extreme-AR + group-aware short-circuit: when the target is
    # dominated by an AR lag and the split is group-aware, the
    # residual T = y - alpha * lag has near-zero signal on unseen
    # groups; every trained model on T overfits per-group patterns
    # and produces predictions worse than the trivial median(T)
    # dummy in y-scale on the test split. Discovery + training of
    # such specs is pure waste (observed in prod: 3 composite
    # specs shipped, all trained models on residuals failed
    # dummy-floor gate with R2<0 and pred_std 3-5x target_std,
    # multi-minute wall-time per target wasted).
    _extreme_ar_skip = bool(getattr(
        composite_target_discovery_config,
        "extreme_ar_group_aware_skip", True,
    ))
    _extreme_ar_threshold = float(getattr(
        composite_target_discovery_config,
        "extreme_ar_threshold", 0.99,
    ))
    _td_report = metadata.get("target_distribution_report", {}) or {}
    _td_diag = _td_report.get("diagnostics", {}) or {}
    _td_knobs = _td_report.get("knob_overrides", {}) or {}
    _lag1_ar = _td_diag.get("lag1_autocorr_per_group")
    _pending_composite = _run_composite_targ_split_cfg_overrides_td(_td_knobs, split_config, group_ids, mlframe_models, composite_target_discovery_config, filtered_train_idx, target_by_type, metadata, _lag1_ar, _td_report, _extreme_ar_skip, _extreme_ar_threshold, _td_diag, _auto_skip, _existing_diags, baseline_diagnostics_config, filtered_train_df, cat_features, _disc_feature_cols, precomputed_specs, discovery_cache_dir, val_df_pd, val_idx, _disc_train_idx, train_df_pd, test_df_pd, train_idx, test_idx, save_charts, data_dir)

    # Global selection: every base target's discovery has now run, so every candidate's honest-holdout
    # quality score is comparable at once. Keep the best-scoring specs across the WHOLE run (not an equal
    # share per target) up to max_total_composite_targets; None keeps every discovered spec (old behaviour).
    _kept_composite = select_composites_to_train(_pending_composite, composite_target_discovery_config, metadata)
    _kept_composite = insert_composite_targets(target_by_type, _kept_composite, metadata)

    n_specs_total = sum(len(v) for tt_specs in metadata["composite_target_specs"].values() for v in tt_specs.values())
    # Composite-feature-stacking stub: surface the discovered specs so the
    # downstream FE pipeline (caller-specific) can opt in via
    # ``composite_oof_predictions`` / ``composite_predictions_as_feature``.
    # Wiring the columns into a generic FE pipeline is non-trivial because
    # the consumer is project-specific, so we ship a metadata marker + warn.
    if getattr(
        composite_target_discovery_config,
        "composite_feature_stacking_enabled", False,
    ) and n_specs_total > 0:
        metadata.setdefault("composite_feature_stacking", {})["enabled"] = True
        metadata["composite_feature_stacking"]["available_specs"] = [
            {"target_type": _tt_str, "target_name": _tname,
             "spec_name": _spec.get("name") if isinstance(_spec, dict) else getattr(_spec, "name", "?"),
             "base_column": _spec.get("base_column") if isinstance(_spec, dict) else getattr(_spec, "base_column", "?"),
             "transform_name": _spec.get("transform_name") if isinstance(_spec, dict) else getattr(_spec, "transform_name", "?")}
            for _tt_str, _by_t in metadata["composite_target_specs"].items()
            for _tname, _spec_list in _by_t.items()
            for _spec in (_spec_list or [])
        ]
        logger.warning(
            "[CompositeFeatureStacking] enabled by config; %d composite "
            "spec(s) surfaced under metadata['composite_feature_stacking']. "
            "Caller must wire ``composite_oof_predictions`` / "
            "``composite_predictions_as_feature`` into the downstream FE "
            "pipeline -- the generic suite does not auto-attach.",
            len(metadata["composite_feature_stacking"]["available_specs"]),
        )
    n_specs_trained = len(_kept_composite)
    _run_composite_targ_specs_trained(n_specs_trained, _gpu_families)

    return target_by_type, metadata


def _run_composite_targ_split_cfg_overrides_td(_td_knobs, split_config, group_ids, mlframe_models, composite_target_discovery_config, filtered_train_idx, target_by_type, metadata, _lag1_ar, _td_report, _extreme_ar_skip, _extreme_ar_threshold, _td_diag, _auto_skip, _existing_diags, baseline_diagnostics_config, filtered_train_df, cat_features, _disc_feature_cols, precomputed_specs, discovery_cache_dir, val_df_pd, val_idx, _disc_train_idx, train_df_pd, test_df_pd, train_idx, test_idx, save_charts, data_dir):
    """Build the per-target discovery context and run discovery for every regression target; returns the buffered pending composites."""
    _split_cfg_overrides = _td_knobs.get("split_config", {}) or {}
    _group_aware_recommended = bool(_split_cfg_overrides.get("prefer_group_aware", False))

    # The ACTUAL production splitter is group-aware iff ``split_config.use_groups`` is set AND the suite produced ``group_ids`` (see
    # _phase_helpers_fit_split.py:288, the single place the real splitter routes through GroupShuffleSplit / StratifiedGroupKFold). This is
    # distinct from ``_group_aware_recommended`` above, which is only the target-distribution analyzer's HINT (knob_overrides). The tiny-CV
    # rerank must follow the real splitter: a user who configured a group-aware split WITHOUT the analyzer recommending it previously got a
    # plain-KFold rerank that promoted per-group memorisers whose trained models then failed the production group-aware test (the documented
    # prod failure). Gate the rerank on EITHER signal so neither a recommendation-only nor a config-only group-aware setup is missed.
    # ``setup_configuration`` normalises split_config to a ``TrainingSplitConfig`` object, but accept a raw dict too so a caller that
    # bypasses the suite boundary (tests, custom drivers) is still honoured rather than silently treated as group-naive.
    if isinstance(split_config, dict):
        _split_cfg_use_groups = bool(split_config.get("use_groups", False))
    else:
        _split_cfg_use_groups = bool(getattr(split_config, "use_groups", False))
    _splitter_group_aware = _split_cfg_use_groups and group_ids is not None
    _group_aware_active = _group_aware_recommended or _splitter_group_aware

    # Extreme-AR skip safety gate: only auto-skip discovery when the model zoo cannot extrapolate unboundedly on unseen
    # groups (tree ensembles / Ridge). A neural or plain-linear model could be rescued by a residualised target, so its
    # presence keeps discovery running even on a strongly-AR target.
    _bounded_only_zoo = _zoo_is_bounded_only(mlframe_models)

    # Part B (zoo-aware rerank-skip): on a bounded-only zoo the "raw already near-perfect -> composite has no headroom"
    # rerank-skip is safe to re-enable (no unbounded model needs a residual target). Default the ratio to a conservative
    # value when the user left it at 0; unbounded zoos keep the user's value (0 = never skip). One model_copy reused for
    # every target so the shared config object is never mutated.
    _disc_cfg_base = composite_target_discovery_config
    _disc_cfg_base = _run_composite_targ_every_target_shared_config(_bounded_only_zoo, composite_target_discovery_config, _disc_cfg_base)

    # Suite-constant group_ids: coerce once + materialise the filtered slice + length cap; the per-target loop below otherwise pays a fresh
    # ``np.asarray`` + ``np.max`` per regression target inside the tiny-rerank wiring block (group_ids is invariant across targets).
    _grp_arr_hoisted: np.ndarray | None = None
    _grp_filtered_slice: np.ndarray | None = None
    _grp_max_required: int = 0
    _grp_filtered_slice = _run_composite_targ_group_ids_none_group(group_ids, _group_aware_active, filtered_train_idx, _grp_filtered_slice)

    # Candidates accepted by EVERY base target's own discovery, buffered rather than written straight into
    # target_by_type: max_total_composite_targets is a budget for the whole run, and picking the best-scoring
    # specs across targets (instead of an equal per-target share) requires seeing all of them first. See the
    # config field's own docstring for the production motivation (47 composite targets from 5 base targets,
    # no overall ceiling).
    _pending_composite: list[dict] = []
    ctx = DiscoveryTargetCtx(
        metadata=metadata, group_ids=group_ids, mlframe_models=mlframe_models, composite_target_discovery_config=composite_target_discovery_config,
        filtered_train_idx=filtered_train_idx, filtered_train_df=filtered_train_df, lag1_ar=_lag1_ar, td_report=_td_report, td_diag=_td_diag,
        extreme_ar_skip=_extreme_ar_skip, extreme_ar_threshold=_extreme_ar_threshold, auto_skip=_auto_skip, existing_diags=_existing_diags,
        baseline_diagnostics_config=baseline_diagnostics_config, cat_features=cat_features, disc_feature_cols=_disc_feature_cols,
        precomputed_specs=precomputed_specs, discovery_cache_dir=discovery_cache_dir, val_df_pd=val_df_pd, val_idx=val_idx,
        disc_train_idx=_disc_train_idx, train_df_pd=train_df_pd, test_df_pd=test_df_pd, train_idx=train_idx, test_idx=test_idx,
        save_charts=save_charts, data_dir=data_dir, group_aware_recommended=_group_aware_recommended, split_cfg_use_groups=_split_cfg_use_groups,
        splitter_group_aware=_splitter_group_aware, group_aware_active=_group_aware_active, bounded_only_zoo=_bounded_only_zoo,
        disc_cfg_base=_disc_cfg_base, grp_filtered_slice=_grp_filtered_slice, pending_composite=_pending_composite,
    )

    for _tt_disc, _named_disc in list(target_by_type.items()):
        if _tt_disc != TargetTypes.REGRESSION:
            continue
        for _tname_disc, _tvals_disc in list(_named_disc.items()):
            discover_one_target(ctx, _tt_disc, _tname_disc, _tvals_disc)
    return _pending_composite


def _run_composite_targ_every_target_shared_config(_bounded_only_zoo, composite_target_discovery_config, _disc_cfg_base):
    """Block of run_composite_target_discovery starting at ``if _bounded_only_zoo and float(getattr(composite_target_discovery_conf``."""
    if _bounded_only_zoo and float(getattr(composite_target_discovery_config, "composite_skip_when_raw_dominates_ratio", 0.0)) <= 0.0:
        try:
            _disc_cfg_base = composite_target_discovery_config.model_copy(
                update={"composite_skip_when_raw_dominates_ratio": _RERANK_SKIP_RATIO_BOUNDED_DEFAULT},
            )
            logger.info(
                "[CompositeTargetDiscovery] bounded-only zoo: rerank raw-dominance skip ratio "
                "defaulted to %.3f (raw R^2 > 0.9996 -> composite has no headroom).",
                _RERANK_SKIP_RATIO_BOUNDED_DEFAULT,
            )
        except Exception as e:
            logger.debug("re-rank-skip-ratio auto-default computation failed, using the caller's config as-is: %s", e)
            _disc_cfg_base = composite_target_discovery_config
    return _disc_cfg_base


def _run_composite_targ_group_ids_none_group(group_ids, _group_aware_active, filtered_train_idx, _grp_filtered_slice):
    """Block of run_composite_target_discovery starting at ``if group_ids is not None and _group_aware_active and filtered_train_id``."""
    if group_ids is not None and _group_aware_active and filtered_train_idx is not None:
        try:
            _grp_arr_hoisted = np.asarray(group_ids)
            _grp_max_required = int(np.max(filtered_train_idx) + 1)
            if _grp_arr_hoisted.shape[0] >= _grp_max_required:
                _grp_filtered_slice = _grp_arr_hoisted[filtered_train_idx]
        except (TypeError, ValueError, IndexError) as _hoist_err:
            logger.debug(
                "[CompositeTargetDiscovery] group_ids hoist failed (%s); " "tiny-rerank will fall back to KFold for every target.",
                _hoist_err,
            )
    return _grp_filtered_slice


def _run_composite_targ_auto_skip_use_hint(_auto_skip, _use_hint, _existing_diags, _tt_disc, _tname_disc, baseline_diagnostics_config, filtered_train_idx, _y_arr, filtered_train_df, cat_features, metadata, _diag):
    """Block of run_composite_target_discovery starting at ``if _auto_skip or _use_hint:``."""
    if _auto_skip or _use_hint:
        _diag = _existing_diags.get(str(_tt_disc), {}).get(_tname_disc)
        if _diag is None:
            try:
                _bd_inline = BaselineDiagnostics(baseline_diagnostics_config)
                _y_train_for_diag = _y_arr[filtered_train_idx] if filtered_train_idx is not None else _y_arr
                _diag_report = _bd_inline.fit_and_report(
                    train_df=filtered_train_df,
                    train_target=_y_train_for_diag,
                    feature_cols=list(filtered_train_df.columns),
                    target_type=str(_tt_disc),
                    target_name=_tname_disc,
                    cat_features=cat_features,
                )
                _diag = _diag_report.to_dict()
                metadata.setdefault("baseline_diagnostics", {}).setdefault(str(_tt_disc), {})[_tname_disc] = _diag
            except Exception as _bd_err:
                logger.info(
                    "[CompositeTargetDiscovery] inline diagnostic precompute " "failed for '%s': %s; discovery proceeds without auto-skip / hint.",
                    _tname_disc,
                    _bd_err,
                )
                _diag = None
    return _diag


def _run_composite_targ_source_truth_did_skip(group_ids, filtered_train_idx, _grp_train):
    """Block of run_composite_target_discovery starting at ``if group_ids is not None:``."""
    if group_ids is not None:
        try:
            _grp_full_pc = np.asarray(group_ids).reshape(-1)
            if _grp_full_pc.shape[0] > int(np.max(filtered_train_idx)):
                _grp_train = _grp_full_pc[filtered_train_idx]
        except (TypeError, ValueError, IndexError):
            _grp_train = None
    return _grp_train


def _replay_cached_payload_into_metadata(metadata: dict, target_type, target_name: str, payload: dict) -> None:
    """Copy a discovery payload's slots into ``metadata`` as fresh containers so in-place metadata edits cannot reach the cache entry."""
    tt = str(target_type)
    metadata["composite_target_specs"].setdefault(tt, {})[target_name] = list(payload.get("specs_export") or [])
    metadata["composite_target_failures"].setdefault(tt, {})[target_name] = list(payload.get("failures") or [])
    metadata.setdefault("composite_target_filter_drops", {}).setdefault(tt, {})[target_name] = dict(payload.get("filter_drops") or {})


def _run_composite_targ_pre_specs(_pre_specs, _tname_disc, discovery_cache_dir, _grp_filtered_slice, _use_hint, _hint_strengths, _disc_df, _disc_cfg, val_df_pd, _y_arr, val_idx, _disc_feature_cols, _disc_signature, _disc_cache, _disc_cache_key):
    """Block of run_composite_target_discovery starting at ``if _pre_specs:``."""
    if _pre_specs:
        _cached_payload = {"specs_export": list(_pre_specs), "failures": [], "filter_drops": {}}
        logger.info(
            "[CompositeTargetDiscovery] replaying %d caller-supplied spec(s) for target='%s'; skipping discovery.",
            len(_cached_payload["specs_export"]), _tname_disc,
        )
    elif discovery_cache_dir is not None:
        _digest = discovery_inputs_digest(group_ids=_grp_filtered_slice, hint_strengths=_hint_strengths if _use_hint else None, disc_df=_disc_df,
                                          time_column=getattr(_disc_cfg, "time_column", None), val_df=val_df_pd, y_full=_y_arr, val_idx=val_idx)
        _disc_cache, _disc_cache_key, _cached_payload = _discovery_cache_lookup(_disc_cfg, _disc_df, _tname_disc, _disc_feature_cols, discovery_cache_dir, _digest,
                                                                                 signature_out=_disc_signature)
    else:
        _cached_payload = None
    return _cached_payload, _disc_cache, _disc_cache_key


def _run_composite_targ_cached_specs_reference_otherwise(_cached_specs):
    """Block of run_composite_target_discovery starting at ``try:``."""
    try:
        from mlframe.training.composite.discovery.shared import reregister_auto_chain_transforms
        _rereg = reregister_auto_chain_transforms([getattr(s, "transform_name", "") for s in _cached_specs])
        if _rereg:
            logger.info(
                "[CompositeTargetDiscovery] cache replay re-registered %d auto-chain transform(s): %s",
                len(_rereg), sorted(_rereg),
            )
    except Exception as _rereg_err:  # best-effort: a cache-replay re-registration miss just means the transform re-registers itself lazily on next use
        log_throttle(
            logger, "composite_discovery_cache_replay_rereg_failed", logging.WARNING,
            "[CompositeTargetDiscovery] cache replay auto-chain re-registration failed: %s", _rereg_err,
        )


def _run_composite_targ_future_past_temporal_data(_tcol, _disc_df, _time_ordering):
    """Block of run_composite_target_discovery starting at ``if _tcol:``."""
    if _tcol:
        try:
            if hasattr(_disc_df, "columns") and _tcol in _disc_df.columns:
                if hasattr(_disc_df, "get_column"):  # polars
                    _time_ordering = _disc_df.get_column(_tcol).to_numpy()
                else:  # pandas
                    _time_ordering = _disc_df[_tcol].to_numpy()
        except Exception as _tc_err:  # best-effort: falls back to base-monotonicity time detection below
            log_throttle(
                logger, "composite_discovery_time_column_extract_failed", logging.WARNING,
                "[CompositeTargetDiscovery] time_column='%s' "
                "could not be extracted (%s); discovery falls "
                "back to base-monotonicity time detection.",
                _tcol, _tc_err,
            )
            _time_ordering = None
    return _time_ordering


def _run_composite_targ_discovery_df_train_only(val_df_pd, val_idx, _y_arr, _disc_val_df, _disc_val_y):
    """Block of run_composite_target_discovery starting at ``try:``."""
    try:
        if val_df_pd is not None and val_idx is not None:
            _vy = np.asarray(_y_arr)[val_idx]
            if hasattr(val_df_pd, "__len__") and len(val_df_pd) == len(_vy):
                _disc_val_df = val_df_pd
                _disc_val_y = _vy
    except Exception as e:  # -- val gate is best-effort; fall back to train-group holdout
        logger.debug("val-gate construction failed, falling back to train-group holdout: %s", e)
        _disc_val_df, _disc_val_y = None, None
    return _disc_val_df, _disc_val_y


def _run_composite_targ_use_stacked_residual(_use_stacked_residual, _disc_instance, _disc_df, _tname_disc, _disc_feature_cols, _disc_train_idx, _disc_cfg, _stacked_time_aware, _time_ordering, _disc_val_df, _disc_val_y, _use_stacked):
    """Block of run_composite_target_discovery starting at ``if _use_stacked_residual:``."""
    if _use_stacked_residual:
        _disc = _disc_instance.fit_stacked_on_residual(
            df=_disc_df,
            target_col=_tname_disc,
            feature_cols=_disc_feature_cols,
            train_idx=_disc_train_idx,
            n_oof_folds=int(getattr(
                _disc_cfg, "stacked_n_oof_folds", 3,
            )),
            residual_aggregation=str(getattr(
                _disc_cfg, "stacked_residual_aggregation", "mean",
            )),
            max_pass1_specs_to_aggregate=int(getattr(
                _disc_cfg,
                "stacked_residual_max_pass1_specs_to_aggregate",
                3,
            )),
            time_aware=_stacked_time_aware, time_ordering=_time_ordering, val_df=_disc_val_df, val_y=_disc_val_y,
        )
    elif _use_stacked:
        _disc = _disc_instance.fit_stacked(
            df=_disc_df,
            target_col=_tname_disc,
            feature_cols=_disc_feature_cols,
            train_idx=_disc_train_idx,
            n_oof_folds=int(getattr(
                _disc_cfg, "stacked_n_oof_folds", 3,
            )),
            max_pass1_specs_to_stack=int(getattr(
                _disc_cfg, "stacked_max_pass1_specs", 3,
            )),
            time_aware=_stacked_time_aware, time_ordering=_time_ordering, val_df=_disc_val_df, val_y=_disc_val_y,
        )
    else:
        _disc = _disc_instance.fit(
            df=_disc_df,
            target_col=_tname_disc,
            feature_cols=_disc_feature_cols,
            train_idx=_disc_train_idx,
            time_ordering=_time_ordering,
            val_df=_disc_val_df,
            val_y=_disc_val_y,
        )
    return _disc


def _run_composite_targ_short_circuits(_disc_cache, _disc_cache_key, _disc, metadata, _tt_disc, _tname_disc):
    """Block of run_composite_target_discovery starting at ``if _disc_cache is not None and _disc_cache_key is not None:``."""
    if _disc_cache is not None and _disc_cache_key is not None:
        try:
            _disc_cache.set(
                _disc_cache_key,
                {
                    "specs_export": _disc.export_specs(),
                    "failures": [r for r in _disc.report() if r.get("rejected")],
                    "filter_drops": _disc.filter_drops(),
                },
            )
            metadata.setdefault("composite_target_cache", {}).setdefault(str(_tt_disc), {})[_tname_disc] = {
                "hit": False,
                "key": _disc_cache_key,
            }
        except Exception as _cache_err:
            logger.info(
                "[CompositeTargetDiscovery] cache set failed for " "target='%s' (%s); ignored.",
                _tname_disc,
                _cache_err,
            )


def _run_composite_targ_metadata_composite_precheck_verdict(_ceiling_verdict, _disc):
    """Block of run_composite_target_discovery starting at ``if _ceiling_verdict is not None:``."""
    if _ceiling_verdict is not None:
        try:
            _disc.composite_precheck_verdict_ = _ceiling_verdict
        except Exception as e:  # -- exotic/read-only _disc; the verdict already lives in metadata
            logger.debug("swallowed exception in _phase_composite_discovery.py: %s", e)
            pass


def _run_composite_targ_nan_rows_domain_violations(_disc, train_df_pd, val_df_pd, test_df_pd, train_idx, val_idx, test_idx, _y_arr, filtered_train_idx, _t_by_spec_for_charts, _pending_composite, _tt_disc, _tname_disc):
    """Block of run_composite_target_discovery starting at ``for _spec in _disc.specs_:``."""
    from mlframe.training.composite import get_transform as _get_transform_local

    for _spec in _disc.specs_:
        _transform = _get_transform_local(_spec.transform_name)
        # Multi-base specs (extra_base_columns non-empty) need a
        # (n_total, 1+K) matrix stacking the primary base column
        # with each extra base. linear_residual_multi.forward
        # consumes that 2-D matrix; passing only the primary
        # column raises ValueError("base has N columns but
        # fitted alphas has M entries") in the forward call.
        _extra_bases = tuple(getattr(_spec, "extra_base_columns", ()) or ())
        _base_primary = _build_full_column_from_splits(
            _spec.base_column,
            train_df_pd, val_df_pd, test_df_pd,
            train_idx, val_idx, test_idx,
            n_total=_y_arr.shape[0],
        )
        if _extra_bases:
            _base_cols = [_base_primary]
            _base_cols.extend(
                _build_full_column_from_splits(
                    _eb_name,
                    train_df_pd, val_df_pd, test_df_pd,
                    train_idx, val_idx, test_idx,
                    n_total=_y_arr.shape[0],
                )
                for _eb_name in _extra_bases
            )
            _base_full = np.column_stack(_base_cols)
        else:
            _base_full = _base_primary
        _ct_t_full, _t_by_spec_for_charts[_spec.name] = composite_t_full(_transform, _spec, _y_arr, _base_full, filtered_train_idx)
        # Not written to target_by_type yet -- buffered so the max_total_composite_targets budget
        # (spent across ALL base targets, not per-target) can pick the best-scoring specs seen so
        # far across the WHOLE run once every target's own discovery has finished. See the
        # end-of-function flush for the actual target_by_type write + log line.
        _raw_rmse = getattr(_spec, "honest_holdout_raw_rmse", None)
        _rmse_gain = getattr(_spec, "honest_holdout_rmse_gain", None)
        _rel_gain: float
        if _raw_rmse is not None and _rmse_gain is not None and _raw_rmse > 0:
            _rel_gain = float(_rmse_gain) / float(_raw_rmse)
            _gain_is_rmse = True
        else:
            _gain_is_rmse = False
            # Honest-holdout RMSE re-score didn't run for this spec (disabled / too few holdout rows) --
            # fall back to the honest MI-gain (still holdout-measured, just not RMSE-scaled) so the spec
            # is still globally rankable rather than silently excluded from the budget entirely.
            _honest_mi = getattr(_spec, "honest_holdout_gain", None)
            _rel_gain = float(_honest_mi) if _honest_mi is not None else float(_spec.mi_gain)
        _pending_composite.append({
            "tt": _tt_disc, "target": _tname_disc, "name": _spec.name, "values": _ct_t_full, "gain": _rel_gain,
            "rmse_gain": _gain_is_rmse, "gain_se": _relative_gain_se(_spec, _raw_rmse) if _gain_is_rmse else None,
        })


def _run_composite_targ_ranked_spec_export_all(save_charts, data_dir, _t_by_spec_for_charts, metadata, _tt_disc, _tname_disc, _y_arr, filtered_train_idx):
    """Block of run_composite_target_discovery starting at ``if save_charts and data_dir and _t_by_spec_for_charts:``."""
    if save_charts and data_dir and _t_by_spec_for_charts:
        try:
            _chart_specs = metadata.get("composite_target_specs", {}).get(str(_tt_disc), {}).get(_tname_disc, [])
            _saved_charts = _render_composite_discovery_diagnostics(
                data_dir=data_dir,
                raw_target_name=_tname_disc,
                y_full=_y_arr, train_idx=filtered_train_idx,
                t_by_spec=_t_by_spec_for_charts,
                specs_export=list(_chart_specs or []),
            )
            if _saved_charts:
                metadata.setdefault("composite_target_diagnostic_charts", {}).setdefault(str(_tt_disc), {})[_tname_disc] = _saved_charts
        except Exception as _chart_err:
            logger.info(
                "[CompositeTargetDiscovery] diagnostic chart render failed for target='%s': %s; training continues.",
                _tname_disc, _chart_err,
            )


def _run_composite_targ_specs_trained(n_specs_trained, _gpu_families):
    """Block of run_composite_target_discovery starting at ``if n_specs_trained > 0:``."""
    if n_specs_trained > 0:
        logger.info(
            "[CompositeTargetDiscovery] %d composite target(s) added to "
            "target_by_type. They will be trained alongside raw targets in the "
            "per-target loop.", n_specs_trained,
        )
        if _gpu_families:
            logger.warning(
                "[CompositeTargetDiscovery] composite mode + GPU training detected "
                "(%s) AND %d composite spec(s) shipped. GPU non-determinism is "
                "amplified by the K=%d extra fits; ensemble weights may drift across "
                "runs even with random_state fixed. Set deterministic=True / "
                "single_precision_histogram=True / force_row_wise=True on the inner "
                "estimators if reproducibility matters.",
                ", ".join(_gpu_families), n_specs_trained, n_specs_trained,
            )
