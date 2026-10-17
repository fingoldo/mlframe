"""Per-target body of the composite-target discovery loop (extreme-AR / auto-skip / row-align / achievable-ceiling gates, cache replay, fresh fit).

Carved from ``_phase_composite_discovery`` so the loop driver there stays small; it keeps the original logger name so caplog filters by it still match.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..composite import CompositeTargetDiscovery
from ..composite.cache import DiscoveryCache
from ._achievable_ceiling import run_achievable_ceiling_precheck
from ._ar_skip import (
    _extreme_ar_discovery_skip,
    _extreme_ar_skip_blocked_by_missing_info,
    _recompute_lag1_ar_per_group,
)
from ._phase_composite_discovery_gates import (
    _drop_specs_whose_bases_the_suite_cannot_materialise,
    _per_target_discovery_config,
)
from ._phase_composite_discovery_helpers import _build_disc_df_for_target
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger("mlframe.training.core._phase_composite_discovery")


class _LazyPcd:
    """Resolve ``_phase_composite_discovery`` attributes at call time: a module-level import would close a cycle (the driver imports this module)."""

    def __getattr__(self, name: str):
        """Forward the attribute lookup to ``_phase_composite_discovery``."""
        from . import _phase_composite_discovery as _mod

        return getattr(_mod, name)


_pcd = _LazyPcd()


@dataclass
class DiscoveryTargetCtx:
    """Suite-level inputs and derived settings shared by every per-target discovery step."""

    metadata: dict
    group_ids: Any
    mlframe_models: Any
    composite_target_discovery_config: Any
    filtered_train_idx: Any
    filtered_train_df: Any
    lag1_ar: Any
    td_report: dict
    td_diag: dict
    extreme_ar_skip: bool
    extreme_ar_threshold: float
    auto_skip: bool
    existing_diags: Any
    baseline_diagnostics_config: Any
    cat_features: Any
    disc_feature_cols: list
    precomputed_specs: Any
    discovery_cache_dir: Any
    val_df_pd: Any
    val_idx: Any
    disc_train_idx: np.ndarray
    train_df_pd: Any
    test_df_pd: Any
    train_idx: Any
    test_idx: Any
    save_charts: Any
    data_dir: Any
    group_aware_recommended: bool
    split_cfg_use_groups: bool
    splitter_group_aware: bool
    group_aware_active: bool
    bounded_only_zoo: bool
    disc_cfg_base: Any
    grp_filtered_slice: np.ndarray | None
    pending_composite: list[dict]


def skip_extreme_ar(
    ctx: DiscoveryTargetCtx,
    _tt_disc: Any,
    _tname_disc: Any,
    _y_arr: np.ndarray,
) -> bool:
    """True when the extreme-AR group-aware gate fires for this target (the rejection is recorded in ``metadata``); logs why it did not fire otherwise."""
    group_ids = ctx.group_ids
    mlframe_models = ctx.mlframe_models
    filtered_train_idx = ctx.filtered_train_idx
    metadata = ctx.metadata
    _lag1_ar = ctx.lag1_ar
    _td_report = ctx.td_report
    _extreme_ar_skip = ctx.extreme_ar_skip
    _extreme_ar_threshold = ctx.extreme_ar_threshold
    _td_diag = ctx.td_diag
    _group_aware_recommended = ctx.group_aware_recommended
    _split_cfg_use_groups = ctx.split_cfg_use_groups
    _splitter_group_aware = ctx.splitter_group_aware
    _group_aware_active = ctx.group_aware_active
    _bounded_only_zoo = ctx.bounded_only_zoo
    # Robustness for the extreme-AR skip (the report can lack the skip keys at discovery time, silently
    # disabling it -- observed across TV6-9 on a 0.9999-AR target). Three fallbacks, each target-specific:
    #  - lag1: recompute the target's own per-group AR(1) from the data when the report didn't carry it;
    #  - group-aware: discovery only RECEIVES group_ids on a group-aware split, so its presence is sufficient;
    #  - picked-target: when we recomputed lag1 for THIS target the value is target-specific, so the
    #    "is this the analyzer's picked target" guard (which only matters for the report's picked-target lag1)
    #    no longer applies.
    _lag1_eff = _lag1_ar
    _recomputed_lag1 = False
    if _lag1_eff is None and group_ids is not None and filtered_train_idx is not None:
        _lag1_eff = _recompute_lag1_ar_per_group(_y_arr, group_ids, filtered_train_idx)
        _recomputed_lag1 = _lag1_eff is not None
    _grp_active_eff = bool(_group_aware_active or group_ids is not None)
    _is_picked_eff = (_td_report.get("picked_target_name") == _tname_disc) or _recomputed_lag1
    if _extreme_ar_discovery_skip(
        skip_enabled=_extreme_ar_skip,
        group_aware_active=_grp_active_eff,
        bounded_only_zoo=_bounded_only_zoo,
        lag1_ar=_lag1_eff,
        is_picked_target=_is_picked_eff,
        threshold=_extreme_ar_threshold,
    ):
        log_throttle(
            logger, "composite_discovery_extreme_ar_skip_fired", logging.WARNING,
            "[CompositeTargetDiscovery] extreme-AR + group-aware "
            "skip fired for target='%s' (lag1_autocorr_per_group=%.4f%s >= %.2f, group-aware split active, "
            "zoo is bounded-only). Residual targets have near-zero signal on unseen groups and "
            "every trained model on T overfits per-group patterns "
            "to ship predictions worse than the median(T) dummy "
            "in y-scale. Discovery skipped; lag_predict in "
            "CT_ENSEMBLE pool will carry the AR signal. Disable "
            "via composite_target_discovery_config."
            "extreme_ar_group_aware_skip=False.",
            _tname_disc, float(_lag1_eff) if _lag1_eff is not None else float("nan"), (" [recomputed]" if _recomputed_lag1 else ""), _extreme_ar_threshold,
        )
        metadata["composite_target_failures"].setdefault(
            str(_tt_disc), {})[_tname_disc] = [{
                "name": _tname_disc, "kept": False, "rejected": True,
                "reason": (
                    f"extreme_ar_group_aware_skip=True + "
                    f"lag1_autocorr_per_group={(float(_lag1_eff) if _lag1_eff is not None else float('nan')):.4f}"
                    f"{' (recomputed)' if _recomputed_lag1 else ''} "
                    f">= threshold {_extreme_ar_threshold}"
                ),
            }]
        return True
    elif _extreme_ar_skip:
        # The skip is enabled but did NOT fire. On a strongly-AR group-aware target this is a missed
        # optimisation (discovery + composite-train wall wasted). Log EVERY input so the blocking precondition
        # is visible in one shot -- in particular whether the target_distribution_report (which carries
        # lag1_autocorr_per_group / picked_target_name / prefer_group_aware) actually reached discovery: an
        # empty ``_td_report`` makes lag1=None + recommended=False, silently disabling the skip.
        # WARNING only when the skip was applicable but blocked by missing information (see the helper).
        _skip_blocked_by_missing_info = _extreme_ar_skip_blocked_by_missing_info(
            group_aware_active=_grp_active_eff, bounded_only_zoo=_bounded_only_zoo, lag1_ar=_lag1_eff,
            is_picked_target=_is_picked_eff, threshold=_extreme_ar_threshold,
        )
        log_throttle(
            logger,
            "composite_discovery_extreme_ar_skip_not_fired" if _skip_blocked_by_missing_info else "composite_discovery_extreme_ar_skip_not_applicable",
            logging.WARNING if _skip_blocked_by_missing_info else logging.DEBUG,
            "[CompositeTargetDiscovery] extreme-AR skip did NOT fire for target=%r: skip_enabled=%s "
            "lag1_report=%r lag1_eff=%r recomputed=%s (threshold=%.2f) group_aware_active=%s (recommended=%s "
            "splitter=%s use_groups=%s gid=%s) bounded_only_zoo=%s zoo=%s picked_target_name=%r is_picked_eff=%s "
            "td_report_keys=%s td_diag_keys=%s. Discovery will run.",
            _tname_disc, _extreme_ar_skip, _lag1_ar, _lag1_eff, _recomputed_lag1, _extreme_ar_threshold,
            _grp_active_eff, _group_aware_recommended, _splitter_group_aware,
            _split_cfg_use_groups, (group_ids is not None), _bounded_only_zoo, list(mlframe_models or []),
            _td_report.get("picked_target_name"), _is_picked_eff,
            sorted(_td_report.keys()), sorted(_td_diag.keys()),
        )
    return False


def baseline_diag_gate(
    ctx: DiscoveryTargetCtx,
    _tt_disc: Any,
    _tname_disc: Any,
    _y_arr: np.ndarray,
) -> tuple[Any, bool, bool]:
    """Return ``(diag, use_hint, skip)``: the target's BaselineDiagnostics (or None) and whether auto-skip rejects the target."""
    composite_target_discovery_config = ctx.composite_target_discovery_config
    filtered_train_idx = ctx.filtered_train_idx
    metadata = ctx.metadata
    _auto_skip = ctx.auto_skip
    _existing_diags = ctx.existing_diags
    baseline_diagnostics_config = ctx.baseline_diagnostics_config
    filtered_train_df = ctx.filtered_train_df
    cat_features = ctx.cat_features
    # BaselineDiagnostics normally runs INSIDE the per-target loop; precompute here when auto_skip
    # or hint is enabled and cache the result so the per-target loop reuses it.
    _use_hint = bool(getattr(
        composite_target_discovery_config,
        "use_baseline_diagnostics_hint", True,
    ))
    _diag = None
    _diag = _pcd._run_composite_targ_auto_skip_use_hint(_auto_skip, _use_hint, _existing_diags, _tt_disc, _tname_disc, baseline_diagnostics_config, filtered_train_idx, _y_arr, filtered_train_df, cat_features, metadata, _diag)
    if _auto_skip:
        if _diag is not None and _diag.get("composite_recommendation") == "unlikely_to_help":
            logger.info(
                "[CompositeTargetDiscovery] auto-skip target='%s': " "BaselineDiagnostics recommendation='unlikely_to_help' (reason: %s).",
                _tname_disc,
                _diag.get("composite_recommendation_reason", "")[:120],
            )
            metadata["composite_target_failures"].setdefault(
                str(_tt_disc), {})[_tname_disc] = [{
                    "name": _tname_disc, "kept": False, "rejected": True,
                    "reason": "auto_skip_on_baseline_optimal=True + "
                              "diagnostic='unlikely_to_help'",
                }]
            return _diag, _use_hint, True
    return _diag, _use_hint, False


def aligned_train_y(
    ctx: DiscoveryTargetCtx,
    _tt_disc: Any,
    _tname_disc: Any,
    _y_arr: np.ndarray,
) -> np.ndarray | None:
    """``y`` restricted to the filtered train rows, or None (with the rejection recorded) when the rows cannot be aligned."""
    filtered_train_idx = ctx.filtered_train_idx
    metadata = ctx.metadata
    filtered_train_df = ctx.filtered_train_df
    if filtered_train_idx is None:
        # _y_arr[None] yields shape (1, n), so the row-align guard below would print a misleading "y[1] vs df[N]"; surface the real cause instead.
        log_throttle(
            logger, "composite_discovery_missing_filtered_train_idx", logging.WARNING,
            "[CompositeTargetDiscovery] filtered_train_idx missing; " "skipping discovery for target='%s'.",
            _tname_disc,
        )
        metadata["composite_target_failures"].setdefault(
            str(_tt_disc), {})[_tname_disc] = [{
                "name": _tname_disc, "kept": False, "rejected": True,
                "reason": "filtered_train_idx is None (cannot align y to train rows)",
            }]
        return None

    _y_train_aligned: np.ndarray = _y_arr[filtered_train_idx]
    if len(_y_train_aligned) != len(filtered_train_df):
        log_throttle(
            logger, "composite_discovery_row_align_mismatch", logging.WARNING,
            "[CompositeTargetDiscovery] target='%s' row-align mismatch "
            "(y[%d] vs filtered_train_df[%d]); skipping discovery.",
            _tname_disc, len(_y_train_aligned), len(filtered_train_df),
        )
        metadata["composite_target_failures"].setdefault(str(_tt_disc), {})[_tname_disc] = [
            {
                "name": _tname_disc,
                "kept": False,
                "rejected": True,
                "reason": (f"row-align mismatch y[{len(_y_train_aligned)}] " f"vs filtered_train_df[{len(filtered_train_df)}]"),
            }
        ]
        return None
    return _y_train_aligned


def ceiling_gate(
    ctx: DiscoveryTargetCtx,
    _tt_disc: Any,
    _tname_disc: Any,
    _y_train_aligned: np.ndarray,
) -> tuple[Any, bool]:
    """Run the measured achievable-ceiling precheck; returns ``(verdict, skip)`` and records a skip decision in ``metadata``."""
    group_ids = ctx.group_ids
    composite_target_discovery_config = ctx.composite_target_discovery_config
    filtered_train_idx = ctx.filtered_train_idx
    metadata = ctx.metadata
    filtered_train_df = ctx.filtered_train_df
    _disc_feature_cols = ctx.disc_feature_cols
    _grp_train = None
    _grp_train = _pcd._run_composite_targ_source_truth_did_skip(group_ids, filtered_train_idx, _grp_train)
    _ceiling_verdict = None
    try:
        _ceiling_verdict = run_achievable_ceiling_precheck(
            config=composite_target_discovery_config,
            df=filtered_train_df,
            target_col=_tname_disc,
            feature_cols=_disc_feature_cols,
            y_train=_y_train_aligned,
            group_ids_train=_grp_train,
        )
    except Exception as _pc_err:  # -- a precheck failure must never abort discovery
        logger.info(
            "[CompositeTargetDiscovery] achievable-ceiling precheck raised for target='%s' (%s); discovery proceeds.",
            _tname_disc, _pc_err,
        )
        _ceiling_verdict = None
    if _ceiling_verdict is not None:
        metadata.setdefault("composite_precheck_verdict", {}).setdefault(str(_tt_disc), {})[_tname_disc] = _ceiling_verdict
        if _ceiling_verdict.get("decision") == "skip":
            metadata["composite_target_failures"].setdefault(
                str(_tt_disc), {})[_tname_disc] = [{
                    "name": _tname_disc, "kept": False, "rejected": True,
                    "reason": _ceiling_verdict.get(
                        "reason",
                        "achievable-ceiling precheck: optimistic composite cannot beat min(raw, lag)",
                    ),
                }]
            return _ceiling_verdict, True
    return _ceiling_verdict, False


def replay_cached_discovery(
    ctx: DiscoveryTargetCtx,
    _tt_disc: Any,
    _tname_disc: Any,
    _cached_payload: Any,
    _disc_cache_key: str | None,
) -> Any:
    """Replay a cached (or caller-supplied) discovery payload into ``metadata`` and return the duck-typed stand-in carrying its ``specs_``."""
    metadata = ctx.metadata
    # Replay the cached output into metadata without refitting.
    # Defensive copy at the load boundary: DiscoveryCache.get() today does a fresh
    # pickle.load (safe in isolation), but the class docstring at composite_cache.py:649
    # mentions a future LRU in-memory sidecar. If that lands, the SAME list / dict
    # reference would be returned across calls; a downstream phase that ever does
    # ``.clear()`` / ``.append()`` on these values (composite_target_y_scale_metrics
    # at _phase_composite_post.py:174-177 already does ``.clear()`` on a sibling
    # metadata list) would corrupt the cache entry in place. Same shape as the CB
    # Pool id-recycle bug -- catch it BEFORE the LRU sidecar lands.
    _pcd._replay_cached_payload_into_metadata(metadata, _tt_disc, _tname_disc, _cached_payload)
    metadata.setdefault("composite_target_cache", {}).setdefault(str(_tt_disc), {})[_tname_disc] = {
        "hit": True,
        "key": _disc_cache_key,
    }
    logger.info(
        "[CompositeTargetDiscovery] cache HIT for target='%s' " "(key=%s); skipping full discovery.",
        _tname_disc,
        ("?" if _disc_cache_key is None or _disc_cache_key == "" else _disc_cache_key)[:16],
    )
    # No specs_/_disc instance available on cache hit; the
    # downstream forward-applier loop expects one. Reconstruct
    # the bare-minimum CompositeSpec list from the cached export.
    try:
        from mlframe.training.composite.spec import CompositeSpec as _Spec

        _cached_specs = [_Spec(**s) if isinstance(s, dict) else s for s in _cached_payload.get("specs_export", [])]
        # Auto-discovered chain transforms (chain_<residual>_<unary>) are registered in-process by
        # _run_auto_chain during a FRESH discovery; on a cache replay that never ran, so re-register any the
        # cached specs reference -- otherwise get_transform / predict-time inversion raises UnknownTransformError.
        _pcd._run_composite_targ_cached_specs_reference_otherwise(_cached_specs)
    except Exception as _replay_err:
        # Spec rebuild failed: without it the forward-applier adds no T columns, yet specs were already claimed in metadata above.
        # Clear the claimed specs so metadata matches the (no-column) reality; full re-discovery fallback is a larger fix.
        log_throttle(
            logger, "composite_discovery_cache_replay_spec_rebuild_failed", logging.WARNING,
            "[CompositeTargetDiscovery] cache replay spec rebuild "
            "failed for target='%s' (key=%s): %s; clearing claimed "
            "specs to avoid a no-column divergence.",
            _tname_disc, ("?" if _disc_cache_key is None or _disc_cache_key == "" else _disc_cache_key)[:16], _replay_err,
        )
        metadata["composite_target_specs"][str(_tt_disc)][_tname_disc] = []
        _cached_specs = []

    class _CacheReplay:
        """Minimal duck-typed stand-in for ``CompositeTargetDiscovery`` exposing only the cached ``specs_`` attribute, used to replay a prior run's composite-target specs without re-running discovery."""

        specs_ = _cached_specs

    _disc = _CacheReplay()
    return _disc


def fit_discovery_fresh(
    ctx: DiscoveryTargetCtx,
    _tt_disc: Any,
    _tname_disc: Any,
    _y_arr: np.ndarray,
    _disc_cfg: Any,
    _disc_df: Any,
    _use_hint: bool,
    _diag: Any,
    _hint_strengths: Any,
    _disc_signature: dict,
    _disc_cache: DiscoveryCache | None,
    _disc_cache_key: str | None,
) -> Any:
    """Fit a fresh ``CompositeTargetDiscovery`` for one target and record its specs / failures / drops in ``metadata``; None when the fit raised."""
    metadata = ctx.metadata
    _disc_feature_cols = ctx.disc_feature_cols
    val_df_pd = ctx.val_df_pd
    val_idx = ctx.val_idx
    _disc_train_idx = ctx.disc_train_idx
    train_df_pd = ctx.train_df_pd
    test_df_pd = ctx.test_df_pd
    _grp_filtered_slice = ctx.grp_filtered_slice
    try:
        _disc_instance = CompositeTargetDiscovery(_disc_cfg)
        # The cache key already hashed this frame; at the default seed it is the signature the fit records.
        if _disc_signature.get("sig") and _disc_signature.get("random_state") == 42:
            _disc_instance._fit_data_signature_seed = (id(_disc_df), _tname_disc, tuple(_disc_feature_cols), _disc_signature["sig"])
        if _use_hint and _diag is not None and _hint_strengths:
            _disc_instance._hint_strengths_pct = _hint_strengths  # type: ignore[attr-defined]  # CompositeTargetDiscovery duck-typed extension point (owned by training/composite)
        # Group-aware tiny-rerank: when the production split
        # is group-aware, the tiny-CV rerank must use
        # GroupKFold or it ranks specs by a metric that
        # doesn't reflect the production OOF distribution
        # (observed in prod: random KFold tiny-rerank
        # promoted 3 composite specs whose trained models
        # all failed dummy-floor gate on group-aware test).
        # Slice group_ids to filtered_train rows so the
        # rerank sees the same groups the discovery loop
        # samples from.
        if _grp_filtered_slice is not None:
            _disc_instance._group_ids_for_rerank = _grp_filtered_slice  # type: ignore[attr-defined]  # CompositeTargetDiscovery duck-typed extension point (owned by training/composite)
            logger.info(
                "[CompositeTargetDiscovery] target='%s' " "tiny-rerank will use GroupKFold " "(group_ids supplied, n_groups=%d on " "training rows).",
                _tname_disc,
                int(np.unique(_grp_filtered_slice).size),
            )
        # Stacked 2-pass discovery (OOF predictions
        # from pass 1 augment feature_cols for pass 2). Wraps the
        # standard ``fit()`` so the rest of the phase code path is
        # unchanged. Opt-in via config; default False keeps the
        # historical single-pass behaviour.
        #
        # Residual-target alt
        # (``fit_stacked_on_residual``). Mutually exclusive with
        # feature-stack mode -- residual wins if both flags set,
        # per its "more direct" docstring recommendation.
        _use_stacked = bool(getattr(
            _disc_cfg, "use_stacked_discovery", False,
        ))
        _use_stacked_residual = bool(getattr(
            _disc_cfg, "use_stacked_discovery_residual", False,
        ))
        if _use_stacked and _use_stacked_residual:
            log_throttle(
                logger, "composite_discovery_both_stacked_flags_set", logging.WARNING,
                "[CompositeTargetDiscovery] both "
                "use_stacked_discovery=True and "
                "use_stacked_discovery_residual=True set; "
                "residual mode wins (more direct route to "
                "residual-of-residual structure). Disable one "
                "flag to silence this warning.",
            )
        # When a time_column is configured the data is
        # temporally ordered, so the stacked OOF-prediction step
        # must use time-respecting folds (TimeSeriesSplit) instead
        # of shuffled K-fold -- otherwise the pass-2 bases / residual
        # target are built from future-contaminated OOF predictions.
        _stacked_time_aware = bool(getattr(_disc_cfg, "time_column", None))
        # Extract the chronological-order column (if the
        # user named one) so discovery sorts the screening
        # sample into a forward-walk instead of leaking
        # future->past on temporal data via shuffled K-fold.
        _time_ordering = None
        _tcol = getattr(_disc_cfg, "time_column", None)
        _time_ordering = _pcd._run_composite_targ_future_past_temporal_data(_tcol, _disc_df, _time_ordering)
        # Supply the VAL frame + val targets so the y-scale gate can validate the
        # predict-T -> invert-to-y pipeline on UNSEEN wells (val groups are disjoint
        # from train under the group-aware split) -- the regime where a residual inverse
        # extrapolates and collapses, which a train-group holdout cannot reproduce. The
        # discovery ``df`` is train-only, so the val split is passed as a separate frame.
        _disc_val_df = None
        _disc_val_y = None
        _disc_val_df, _disc_val_y = _pcd._run_composite_targ_discovery_df_train_only(val_df_pd, val_idx, _y_arr, _disc_val_df, _disc_val_y)
        _disc = _pcd._run_composite_targ_use_stacked_residual(_use_stacked_residual, _disc_instance, _disc_df, _tname_disc, _disc_feature_cols, _disc_train_idx, _disc_cfg, _stacked_time_aware, _time_ordering, _disc_val_df, _disc_val_y, _use_stacked)
    except Exception as _disc_err:  # best-effort: training continues without composite expansion for this target
        log_throttle(
            logger, "composite_discovery_fit_failed", logging.WARNING,
            "[CompositeTargetDiscovery] fit failed for target='%s': %s. " "Per-target training continues without composite expansion.",
            _tname_disc,
            _disc_err,
        )
        metadata["composite_target_failures"].setdefault(
            str(_tt_disc), {})[_tname_disc] = [{
                "name": _tname_disc, "kept": False, "rejected": True,
                "reason": f"discovery fit raised: {_disc_err}",
            }]
        return None

    _unbuildable = _drop_specs_whose_bases_the_suite_cannot_materialise(
        _disc, (train_df_pd, val_df_pd, test_df_pd), _tname_disc,
    )
    metadata["composite_target_specs"].setdefault(str(_tt_disc), {})
    metadata["composite_target_specs"][str(_tt_disc)][_tname_disc] = _disc.export_specs()
    metadata["composite_target_failures"].setdefault(str(_tt_disc), {})
    metadata["composite_target_failures"][str(_tt_disc)][_tname_disc] = [r for r in _disc.report() if r.get("rejected")] + _unbuildable
    metadata.setdefault("composite_target_filter_drops", {})
    metadata["composite_target_filter_drops"].setdefault(str(_tt_disc), {})
    metadata["composite_target_filter_drops"][str(_tt_disc)][_tname_disc] = _disc.filter_drops()

    # Populate the cache so the next call with identical inputs
    # short-circuits.
    _pcd._run_composite_targ_short_circuits(_disc_cache, _disc_cache_key, _disc, metadata, _tt_disc, _tname_disc)
    return _disc


def discover_one_target(
    ctx: DiscoveryTargetCtx,
    _tt_disc: Any,
    _tname_disc: Any,
    _tvals_disc: Any,
) -> None:
    """Discover composite specs for one regression target and append its accepted candidates to ``ctx.pending_composite``."""
    composite_target_discovery_config = ctx.composite_target_discovery_config
    filtered_train_idx = ctx.filtered_train_idx
    metadata = ctx.metadata
    filtered_train_df = ctx.filtered_train_df
    _disc_feature_cols = ctx.disc_feature_cols
    precomputed_specs = ctx.precomputed_specs
    discovery_cache_dir = ctx.discovery_cache_dir
    val_df_pd = ctx.val_df_pd
    val_idx = ctx.val_idx
    train_df_pd = ctx.train_df_pd
    test_df_pd = ctx.test_df_pd
    train_idx = ctx.train_idx
    test_idx = ctx.test_idx
    save_charts = ctx.save_charts
    data_dir = ctx.data_dir
    _disc_cfg_base = ctx.disc_cfg_base
    _grp_filtered_slice = ctx.grp_filtered_slice
    _pending_composite = ctx.pending_composite
    _y_arr = np.asarray(_tvals_disc)
    if _y_arr.ndim != 1:
        metadata["composite_target_failures"].setdefault(
            str(_tt_disc), {})[_tname_disc] = [{
                "name": _tname_disc, "kept": False, "rejected": True,
                "reason": "multilabel target unsupported (R3.18 future PR)",
            }]
        return

    if skip_extreme_ar(ctx, _tt_disc, _tname_disc, _y_arr):
        return
    _diag, _use_hint, _auto_skipped = baseline_diag_gate(ctx, _tt_disc, _tname_disc, _y_arr)
    if _auto_skipped:
        return
    _y_train_aligned = aligned_train_y(ctx, _tt_disc, _tname_disc, _y_arr)
    if _y_train_aligned is None:
        return
    _ceiling_verdict, _ceiling_skip = ceiling_gate(ctx, _tt_disc, _tname_disc, _y_train_aligned)
    if _ceiling_skip:
        return

    _disc_df = _build_disc_df_for_target(filtered_train_df, _tname_disc, _y_train_aligned)

    _disc_cfg, _hint_strengths = _per_target_discovery_config(
        _disc_cfg_base, _diag, _tname_disc, use_hint=_use_hint,
        hint_top_k=getattr(composite_target_discovery_config, "baseline_diagnostics_hint_top_k", 3),
    )

    # Disk-backed discovery cache (ship-or-strip -> ship). Key
    # carries data fingerprint + target column + config signature
    # (which embeds mlframe/sklearn/lightgbm/catboost versions for
    # cache-poisoning protection). A hit skips the full MI / Wilcoxon
    # / tiny-model rerank path that otherwise costs minutes on
    # multi-million-row frames.
    _disc_cache: DiscoveryCache | None = None
    _disc_cache_key: str | None = None
    _disc_signature: dict = {}  # the frame signature the cache key computed, handed to the fit below
    # A caller-supplied spec set (a prior run's metadata, the documented reuse path) replays exactly like a cache
    # hit: the same forward-applier builds the T columns, the same auto-chain re-registration runs, and the same
    # dedup and global cap apply. Seeding only metadata, as the old fast path did, trained nothing at all.
    _pre_specs = None
    if precomputed_specs:
        _pre_specs = (precomputed_specs.get(str(_tt_disc)) or {}).get(_tname_disc)
    _cached_payload, _disc_cache, _disc_cache_key = _pcd._run_composite_targ_pre_specs(_pre_specs, _tname_disc, discovery_cache_dir, _grp_filtered_slice, _use_hint, _hint_strengths, _disc_df, _disc_cfg, val_df_pd, _y_arr, val_idx, _disc_feature_cols, _disc_signature, _disc_cache, _disc_cache_key)

    if _cached_payload is not None:
        _disc = replay_cached_discovery(ctx, _tt_disc, _tname_disc, _cached_payload, _disc_cache_key)
    else:
        _fresh = fit_discovery_fresh(
            ctx, _tt_disc, _tname_disc, _y_arr, _disc_cfg, _disc_df, _use_hint, _diag, _hint_strengths, _disc_signature, _disc_cache, _disc_cache_key,
        )
        if _fresh is None:
            return
        _disc = _fresh

    # Record the measured achievable-ceiling verdict on the discovery instance (single source of truth also lives
    # in metadata["composite_precheck_verdict"]); both the cache-replay and fresh-fit ``_disc`` accept the attr.
    _pcd._run_composite_targ_metadata_composite_precheck_verdict(_ceiling_verdict, _disc)

    # Apply frozen (train-fitted) params to ALL rows so the per-target loop has T for val/test.
    # NaN rows (domain violations on val/test) get imputed with median(T_train).
    _t_by_spec_for_charts: dict[str, np.ndarray] = {}
    _pcd._run_composite_targ_nan_rows_domain_violations(_disc, train_df_pd, val_df_pd, test_df_pd, train_idx, val_idx, test_idx, _y_arr, filtered_train_idx, _t_by_spec_for_charts, _pending_composite, _tt_disc, _tname_disc)
    # Each shipped spec costs a full model-zoo fit: drop ones whose T is equivalent to raw y or to a better spec's T.
    from mlframe.training.core._phase_composite_discovery_dedup import prune_equivalent_composite_specs
    for _dropped_name in prune_equivalent_composite_specs(
        specs=list(_disc.specs_), t_by_name=_t_by_spec_for_charts, y_full=_y_arr, train_idx=filtered_train_idx,
        pending=_pending_composite, metadata=metadata, target_type=str(_tt_disc), target_name=_tname_disc,
    ):
        _t_by_spec_for_charts.pop(_dropped_name, None)

    # Render the winning-spec diagnostics (target-distribution + MI-gain) into the chart dir; the
    # discovery accept-path is the only point where the original y, the per-spec T column, and the
    # ranked spec export all coexist, so the wiring lives here rather than the later report path.
    _pcd._run_composite_targ_ranked_spec_export_all(save_charts, data_dir, _t_by_spec_for_charts, metadata, _tt_disc, _tname_disc, _y_arr, filtered_train_idx)
