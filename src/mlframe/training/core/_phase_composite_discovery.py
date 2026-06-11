"""
Phase 4.6: composite-target discovery (opt-in, default OFF).

Runs after outlier detection, before per-target training. Each discovered spec
becomes one entry in ``target_by_type[regression]``; specs are stored on
``metadata["composite_target_specs"]`` for downstream inversion at predict time.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

from ..baselines import BaselineDiagnostics
from ..composite import CompositeTargetDiscovery
from ..composite.cache import (
    ConfigSignatureV1,
    DiscoveryCache,
    compute_config_signature_v1,
    data_signature,
    make_discovery_cache_key,
)
from ..configs import TargetTypes
from .utils import (
    _build_full_column_from_splits,
    _defensive_copy_and_expand_multilabel_regression,
    _init_composite_discovery_metadata,
)

logger = logging.getLogger(__name__)


def _build_disc_df_for_target(filtered_train_df, target_name: str, y_train_aligned):
    """Build a per-target discovery frame that injects ``target_name`` WITHOUT mutating the caller's ``filtered_train_df``.

    Pandas ``DataFrame.copy(deep=False)`` shares the underlying BlockManager with the source; a subsequent
    ``out[target_name] = arr`` setitem can promote and mutate the SHARED block depending on the existing dtype layout,
    causing the target column to intermittently appear on the caller's ``filtered_train_df`` post-loop. The per-target
    discovery loop then accumulates leakage: target_A injected for the first iter shows up as a feature when target_B is
    processed next. ``DataFrame.assign`` always builds a fresh BlockManager so the source is guaranteed untouched, at the
    same memory cost as ``copy(deep=False)+setitem`` would have paid on the new column anyway.

    Polars ``with_columns`` is naturally immutable and returns a fresh frame, so no special handling is needed there.
    """
    if isinstance(filtered_train_df, pd.DataFrame):
        # concat(axis=1) builds a fresh BlockManager (so the caller's frame
        # is NOT mutated -- same immutability guarantee as the prior
        # ``.assign``) while attaching the target as a single consolidated
        # block. On the wide, upstream-fragmented discovery frame this also
        # silences pandas' "highly fragmented" PerformanceWarning that
        # per-column ``.assign``/insert triggers.
        target_series = pd.Series(
            y_train_aligned, index=filtered_train_df.index, name=target_name,
        )
        cols_wo_target = [c for c in filtered_train_df.columns if c != target_name]
        return pd.concat([filtered_train_df[cols_wo_target], target_series], axis=1)
    return filtered_train_df.with_columns(pl.Series(target_name, y_train_aligned))


def _discovery_config_signature(config: Any) -> ConfigSignatureV1:
    """Stable JSON-derived signature of a CompositeTargetDiscoveryConfig.

    Combined with library versions so a dependency bump invalidates
    cached specs - this is the cache-poisoning protection: a CatBoost
    upgrade changes MI bin boundaries, a polars 1->2 bump changes
    categorical codes, a numpy 2.x bump changes RNG semantics, so we
    MUST refit. The version tuple covers every library whose semantics
    can shift the discovered specs:

      * ``mlframe`` - our own version (any change is a refit signal)
      * ``sklearn`` - shared transformers; MI estimator lives here
      * ``lightgbm`` / ``catboost`` / ``xgboost`` - inner models for
        the tiny-model rerank phase
      * ``polars`` - categorical/string dtype codes that feed into
        domain checks + signatures
      * ``numpy`` - dtype promotions + RNG defaults changed in 2.x
      * ``scipy`` - Wilcoxon implementation
      * ``pandas`` - dtype dispatch on the fallback path
      * ``python`` - major.minor (3.11 -> 3.12 changes pickle proto +
        dict ordering side-effects in some serialisers)
    """
    import sys

    versions: dict[str, str] = {}
    try:
        from mlframe import __version__ as _mlv
        versions["mlframe"] = _mlv
    except Exception:
        versions["mlframe"] = "?"
    for _name in (
        "sklearn",
        "lightgbm",
        "catboost",
        "xgboost",
        "polars",
        "numpy",
        "scipy",
        "pandas",
    ):
        try:
            mod = __import__(_name)
            _ver_str = str(getattr(mod, "__version__", "?"))
            # Major.minor only -- patch bumps invalidate every cached spec even though MI /
            # Wilcoxon / boosting math is unchanged. Strip patch + any dev / rc tags.
            _parts = _ver_str.split(".")
            if len(_parts) >= 2 and _parts[0].isdigit():
                _ver_str = f"{_parts[0]}.{_parts[1].split('+')[0].split('rc')[0].split('dev')[0]}"
            versions[_name] = _ver_str
        except Exception:
            versions[_name] = "absent"
    versions["python"] = f"{sys.version_info.major}.{sys.version_info.minor}"
    return compute_config_signature_v1(config, library_versions=versions)


def run_composite_target_discovery(
    *,
    composite_target_discovery_config,
    target_by_type: dict,
    mlframe_models: list[str] | None,
    metadata: dict,
    filtered_train_df,
    filtered_train_idx,
    train_df_pd,
    val_df_pd,
    test_df_pd,
    train_idx,
    val_idx,
    test_idx,
    baseline_diagnostics_config,
    cat_features: list[str] | None,
    verbose: bool,
    discovery_cache_dir: Any = None,
    group_ids: Any = None,
    split_config: Any = None,
) -> tuple[dict, dict]:
    """Run composite-target discovery for regression targets.

    Defensive: target_by_type is shallow-copied before adding entries so the
    FTE-returned original is never mutated in-place.

    Returns updated (target_by_type, metadata).
    """
    _gpu_families, _kept_spec_total = _init_composite_discovery_metadata(
        composite_target_discovery_config=composite_target_discovery_config,
        target_by_type=target_by_type,
        mlframe_models=mlframe_models,
        metadata=metadata,
    )
    if not (composite_target_discovery_config.enabled
            and TargetTypes.REGRESSION in target_by_type):
        return target_by_type, metadata

    target_by_type = _defensive_copy_and_expand_multilabel_regression(
        target_by_type=target_by_type,
        composite_target_discovery_config=composite_target_discovery_config,
        metadata=metadata,
    )

    try:
        _disc_feature_cols = list(filtered_train_df.columns)
    except Exception:
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
    _split_cfg_overrides = _td_knobs.get("split_config", {}) or {}
    _group_aware_recommended = bool(
        _split_cfg_overrides.get("prefer_group_aware", False)
    )

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

    # Suite-constant group_ids: coerce once + materialise the filtered slice + length cap; the per-target loop below otherwise pays a fresh
    # ``np.asarray`` + ``np.max`` per regression target inside the tiny-rerank wiring block (group_ids is invariant across targets).
    _grp_arr_hoisted: np.ndarray | None = None
    _grp_filtered_slice: np.ndarray | None = None
    _grp_max_required: int = 0
    if (group_ids is not None
            and _group_aware_active
            and filtered_train_idx is not None):
        try:
            _grp_arr_hoisted = np.asarray(group_ids)
            _grp_max_required = int(np.max(filtered_train_idx) + 1)
            if _grp_arr_hoisted.shape[0] >= _grp_max_required:
                _grp_filtered_slice = _grp_arr_hoisted[filtered_train_idx]
        except (TypeError, ValueError, IndexError) as _hoist_err:
            logger.debug(
                "[CompositeTargetDiscovery] group_ids hoist failed (%s); "
                "tiny-rerank will fall back to KFold for every target.",
                _hoist_err,
            )

    for _tt_disc, _named_disc in list(target_by_type.items()):
        if _tt_disc != TargetTypes.REGRESSION:
            continue
        for _tname_disc, _tvals_disc in list(_named_disc.items()):
            _y_arr = np.asarray(_tvals_disc)
            if _y_arr.ndim != 1:
                metadata["composite_target_failures"].setdefault(
                    str(_tt_disc), {})[_tname_disc] = [{
                        "name": _tname_disc, "kept": False, "rejected": True,
                        "reason": "multilabel target unsupported (R3.18 future PR)",
                    }]
                continue

            if (_extreme_ar_skip
                    and _group_aware_recommended
                    and _lag1_ar is not None
                    and _td_report.get("picked_target_name") == _tname_disc
                    and float(_lag1_ar) >= _extreme_ar_threshold):
                logger.warning(
                    "[CompositeTargetDiscovery] extreme-AR + group-aware "
                    "skip fired for target='%s' (lag1_autocorr_per_group="
                    "%.4f >= %.2f, split prefers group-aware). Residual "
                    "targets have near-zero signal on unseen groups and "
                    "every trained model on T overfits per-group patterns "
                    "to ship predictions worse than the median(T) dummy "
                    "in y-scale. Discovery skipped; lag_predict in "
                    "CT_ENSEMBLE pool will carry the AR signal. Disable "
                    "via composite_target_discovery_config."
                    "extreme_ar_group_aware_skip=False.",
                    _tname_disc, float(_lag1_ar), _extreme_ar_threshold,
                )
                metadata["composite_target_failures"].setdefault(
                    str(_tt_disc), {})[_tname_disc] = [{
                        "name": _tname_disc, "kept": False, "rejected": True,
                        "reason": (
                            f"extreme_ar_group_aware_skip=True + "
                            f"lag1_autocorr_per_group={float(_lag1_ar):.4f} "
                            f">= threshold {_extreme_ar_threshold}"
                        ),
                    }]
                continue

            # BaselineDiagnostics normally runs INSIDE the per-target loop; precompute here when auto_skip
            # or hint is enabled and cache the result so the per-target loop reuses it.
            _use_hint = bool(getattr(
                composite_target_discovery_config,
                "use_baseline_diagnostics_hint", False,
            ))
            _diag = None
            if _auto_skip or _use_hint:
                _diag = (
                    _existing_diags.get(str(_tt_disc), {}).get(_tname_disc)
                )
                if _diag is None:
                    try:
                        _bd_inline = BaselineDiagnostics(
                            baseline_diagnostics_config
                        )
                        _y_train_for_diag = (
                            _y_arr[filtered_train_idx]
                            if filtered_train_idx is not None else _y_arr
                        )
                        _diag_report = _bd_inline.fit_and_report(
                            train_df=filtered_train_df,
                            train_target=_y_train_for_diag,
                            feature_cols=list(filtered_train_df.columns),
                            target_type=str(_tt_disc),
                            target_name=_tname_disc,
                            cat_features=cat_features,
                        )
                        _diag = _diag_report.to_dict()
                        metadata.setdefault("baseline_diagnostics", {}) \
                            .setdefault(str(_tt_disc), {})[_tname_disc] = _diag
                    except Exception as _bd_err:
                        logger.info(
                            "[CompositeTargetDiscovery] inline diagnostic precompute "
                            "failed for '%s': %s; discovery proceeds without auto-skip / hint.",
                            _tname_disc, _bd_err,
                        )
                        _diag = None
            if _auto_skip:
                if (_diag is not None
                        and _diag.get("composite_recommendation") == "unlikely_to_help"):
                    logger.info(
                        "[CompositeTargetDiscovery] auto-skip target='%s': "
                        "BaselineDiagnostics recommendation='unlikely_to_help' (reason: %s).",
                        _tname_disc,
                        _diag.get("composite_recommendation_reason", "")[:120],
                    )
                    metadata["composite_target_failures"].setdefault(
                        str(_tt_disc), {})[_tname_disc] = [{
                            "name": _tname_disc, "kept": False, "rejected": True,
                            "reason": "auto_skip_on_baseline_optimal=True + "
                                      "diagnostic='unlikely_to_help'",
                        }]
                    continue

            if filtered_train_idx is None:
                # _y_arr[None] yields shape (1, n), so the row-align guard below would print a misleading "y[1] vs df[N]"; surface the real cause instead.
                logger.warning(
                    "[CompositeTargetDiscovery] filtered_train_idx missing; "
                    "skipping discovery for target='%s'.",
                    _tname_disc,
                )
                metadata["composite_target_failures"].setdefault(
                    str(_tt_disc), {})[_tname_disc] = [{
                        "name": _tname_disc, "kept": False, "rejected": True,
                        "reason": "filtered_train_idx is None (cannot align y to train rows)",
                    }]
                continue

            _y_train_aligned = _y_arr[filtered_train_idx]
            if len(_y_train_aligned) != len(filtered_train_df):
                logger.warning(
                    "[CompositeTargetDiscovery] target='%s' row-align mismatch "
                    "(y[%d] vs filtered_train_df[%d]); skipping discovery.",
                    _tname_disc, len(_y_train_aligned), len(filtered_train_df),
                )
                metadata["composite_target_failures"].setdefault(
                    str(_tt_disc), {})[_tname_disc] = [{
                        "name": _tname_disc, "kept": False, "rejected": True,
                        "reason": (
                            f"row-align mismatch y[{len(_y_train_aligned)}] "
                            f"vs filtered_train_df[{len(filtered_train_df)}]"
                        ),
                    }]
                continue

            _disc_df = _build_disc_df_for_target(filtered_train_df, _tname_disc, _y_train_aligned)

            # If hint enabled and BD ran, derive per-target config with dominant_features_hint from ablation top-K.
            _disc_cfg = composite_target_discovery_config
            if _use_hint and _diag is not None:
                _hint_top_k = max(1, int(getattr(
                    composite_target_discovery_config,
                    "baseline_diagnostics_hint_top_k", 3,
                )))
                _ablation = _diag.get("ablation", []) or []
                _ablation_sorted = sorted(
                    _ablation,
                    key=lambda e: -float(e.get("delta_pct", 0.0)),
                )
                _hint_cols = [
                    e["feature"] for e in _ablation_sorted[:_hint_top_k]
                    if e.get("feature")
                ]
                # Strong hints get all top_k slots; weak hints fall back to half-slot cap inside discovery.
                _hint_strengths = [
                    float(e.get("delta_pct", 0.0))
                    for e in _ablation_sorted[:_hint_top_k]
                    if e.get("feature")
                ]
                if _hint_cols:
                    try:
                        _disc_cfg = composite_target_discovery_config.model_copy(
                            update={"dominant_features_hint": _hint_cols},
                        )
                        logger.info(
                            "[CompositeTargetDiscovery] target='%s' hint from "
                            "BaselineDiagnostics ablation top-%d: %s (max delta%%=%.1f)",
                            _tname_disc, len(_hint_cols), _hint_cols,
                            max(_hint_strengths) if _hint_strengths else 0.0,
                        )
                    except Exception as _clone_err:
                        logger.info(
                            "[CompositeTargetDiscovery] hint clone failed for "
                            "target='%s' (%s); proceeding with MI-only.",
                            _tname_disc, _clone_err,
                        )
            else:
                _hint_strengths = None

            # Disk-backed discovery cache (ship-or-strip -> ship). Key
            # carries data fingerprint + target column + config signature
            # (which embeds mlframe/sklearn/lightgbm/catboost versions for
            # cache-poisoning protection). A hit skips the full MI / Wilcoxon
            # / tiny-model rerank path that otherwise costs minutes on
            # multi-million-row frames.
            _disc_cache: DiscoveryCache | None = None
            _disc_cache_key: str | None = None
            if discovery_cache_dir is not None:
                try:
                    _disc_cache = DiscoveryCache(discovery_cache_dir)
                    # ``random_state=0`` is a legitimate sklearn seed and MUST
                    # reach the row-sampler verbatim. The previous ``or 42`` form
                    # silently rewrote 0->42, collapsing seed=0 and seed=42 to
                    # the same data_signature and breaking reproducibility for
                    # any caller that passed 0. ``None`` (no attribute / unset)
                    # still folds to 42 (the historical default).
                    _rs_raw = getattr(_disc_cfg, "random_state", 42)
                    _df_sig = data_signature(
                        _disc_df, _tname_disc, _disc_feature_cols,
                        random_state=int(42 if _rs_raw is None else _rs_raw),
                    )
                    _cfg_sig = _discovery_config_signature(_disc_cfg)
                    # random_state is already folded into _df_sig (seeds the row-sample) and into _cfg_sig (via the dataclass dump). Passing it again to make_discovery_cache_key would be a double-fold (DISC-RANDOM-STATE-DBL): the same data + same config but with random_state mutated would produce three independent hash mixes. We rename the kwarg here to ``_legacy_random_state_sentinel=0`` so a future reader cannot misread "random_state=0" as the actual seed in use.
                    _disc_cache_key = make_discovery_cache_key(
                        _df_sig, _tname_disc, _cfg_sig,
                        _legacy_random_state_sentinel=0,
                    )
                    _cached_payload = _disc_cache.get(_disc_cache_key)
                except Exception as _cache_err:
                    logger.info(
                        "[CompositeTargetDiscovery] cache key build failed for "
                        "target='%s' (%s); proceeding without cache.",
                        _tname_disc, _cache_err,
                    )
                    _cached_payload = None
            else:
                _cached_payload = None

            if _cached_payload is not None:
                # Replay the cached output into metadata without refitting.
                # Defensive copy at the load boundary: DiscoveryCache.get() today does a fresh
                # pickle.load (safe in isolation), but the class docstring at composite_cache.py:649
                # mentions a future LRU in-memory sidecar. If that lands, the SAME list / dict
                # reference would be returned across calls; a downstream phase that ever does
                # ``.clear()`` / ``.append()`` on these values (composite_target_y_scale_metrics
                # at _phase_composite_post.py:174-177 already does ``.clear()`` on a sibling
                # metadata list) would corrupt the cache entry in place. Same shape as the CB
                # Pool id-recycle bug from wave 7 -- catch it BEFORE the LRU sidecar lands.
                metadata["composite_target_specs"].setdefault(str(_tt_disc), {})
                metadata["composite_target_specs"][str(_tt_disc)][_tname_disc] = list(
                    _cached_payload.get("specs_export") or []
                )
                metadata["composite_target_failures"].setdefault(str(_tt_disc), {})
                metadata["composite_target_failures"][str(_tt_disc)][_tname_disc] = list(
                    _cached_payload.get("failures") or []
                )
                metadata.setdefault("composite_target_filter_drops", {})
                metadata["composite_target_filter_drops"].setdefault(str(_tt_disc), {})
                metadata["composite_target_filter_drops"][str(_tt_disc)][_tname_disc] = dict(
                    _cached_payload.get("filter_drops") or {}
                )
                metadata.setdefault("composite_target_cache", {}) \
                    .setdefault(str(_tt_disc), {})[_tname_disc] = {
                        "hit": True, "key": _disc_cache_key,
                    }
                logger.info(
                    "[CompositeTargetDiscovery] cache HIT for target='%s' "
                    "(key=%s); skipping full discovery.",
                    _tname_disc, (_disc_cache_key or "?")[:16],
                )
                # No specs_/_disc instance available on cache hit; the
                # downstream forward-applier loop expects one. Reconstruct
                # the bare-minimum CompositeSpec list from the cached export.
                try:
                    from ..composite.spec import CompositeSpec as _Spec
                    _cached_specs = [
                        _Spec(**s) if isinstance(s, dict) else s
                        for s in _cached_payload.get("specs_export", [])
                    ]
                except Exception as _replay_err:
                    # Spec rebuild failed: without it the forward-applier adds no T columns, yet specs were already claimed in metadata above.
                    # Clear the claimed specs so metadata matches the (no-column) reality; full re-discovery fallback is a larger fix.
                    logger.warning(
                        "[CompositeTargetDiscovery] cache replay spec rebuild "
                        "failed for target='%s' (key=%s): %s; clearing claimed "
                        "specs to avoid a no-column divergence.",
                        _tname_disc, (_disc_cache_key or "?")[:16], _replay_err,
                    )
                    metadata["composite_target_specs"][str(_tt_disc)][_tname_disc] = []
                    _cached_specs = []

                class _CacheReplay:
                    specs_ = _cached_specs

                _disc = _CacheReplay()
            else:
                try:
                    _disc_instance = CompositeTargetDiscovery(_disc_cfg)
                    if _use_hint and _diag is not None and _hint_strengths:
                        _disc_instance._hint_strengths_pct = _hint_strengths
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
                        _disc_instance._group_ids_for_rerank = _grp_filtered_slice
                        logger.info(
                            "[CompositeTargetDiscovery] target='%s' "
                            "tiny-rerank will use GroupKFold "
                            "(group_ids supplied, n_groups=%d on "
                            "training rows).",
                            _tname_disc,
                            int(np.unique(_grp_filtered_slice).size),
                        )
                    # Pack #3 wiring: stacked 2-pass discovery (OOF predictions
                    # from pass 1 augment feature_cols for pass 2). Wraps the
                    # standard ``fit()`` so the rest of the phase code path is
                    # unchanged. Opt-in via config; default False keeps the
                    # historical single-pass behaviour.
                    #
                    # T1#6 2026-05-18 Pack #4 wiring: residual-target alt
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
                        logger.warning(
                            "[CompositeTargetDiscovery] both "
                            "use_stacked_discovery=True and "
                            "use_stacked_discovery_residual=True set; "
                            "residual mode wins (more direct route to "
                            "residual-of-residual structure). Disable one "
                            "flag to silence this warning."
                        )
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
                        )
                    else:
                        # M6: extract the chronological-order column (if the
                        # user named one) so discovery sorts the screening
                        # sample into a forward-walk instead of leaking
                        # future->past on temporal data via shuffled K-fold.
                        _time_ordering = None
                        _tcol = getattr(_disc_cfg, "time_column", None)
                        if _tcol:
                            try:
                                if hasattr(_disc_df, "columns") and _tcol in _disc_df.columns:
                                    if hasattr(_disc_df, "get_column"):  # polars
                                        _time_ordering = _disc_df.get_column(_tcol).to_numpy()
                                    else:  # pandas
                                        _time_ordering = _disc_df[_tcol].to_numpy()
                            except Exception as _tc_err:
                                logger.warning(
                                    "[CompositeTargetDiscovery] time_column='%s' "
                                    "could not be extracted (%s); discovery falls "
                                    "back to base-monotonicity time detection.",
                                    _tcol, _tc_err,
                                )
                                _time_ordering = None
                        _disc = _disc_instance.fit(
                            df=_disc_df,
                            target_col=_tname_disc,
                            feature_cols=_disc_feature_cols,
                            train_idx=_disc_train_idx,
                            time_ordering=_time_ordering,
                        )
                except Exception as _disc_err:
                    logger.warning(
                        "[CompositeTargetDiscovery] fit failed for target='%s': %s. "
                        "Per-target training continues without composite expansion.",
                        _tname_disc, _disc_err,
                    )
                    metadata["composite_target_failures"].setdefault(
                        str(_tt_disc), {})[_tname_disc] = [{
                            "name": _tname_disc, "kept": False, "rejected": True,
                            "reason": f"discovery fit raised: {_disc_err}",
                        }]
                    continue

                metadata["composite_target_specs"].setdefault(str(_tt_disc), {})
                metadata["composite_target_specs"][str(_tt_disc)][_tname_disc] = (
                    _disc.export_specs()
                )
                metadata["composite_target_failures"].setdefault(str(_tt_disc), {})
                metadata["composite_target_failures"][str(_tt_disc)][_tname_disc] = [
                    r for r in _disc.report() if r.get("rejected")
                ]
                metadata.setdefault("composite_target_filter_drops", {})
                metadata["composite_target_filter_drops"].setdefault(str(_tt_disc), {})
                metadata["composite_target_filter_drops"][str(_tt_disc)][
                    _tname_disc] = _disc.filter_drops()

                # Populate the cache so the next call with identical inputs
                # short-circuits.
                if _disc_cache is not None and _disc_cache_key is not None:
                    try:
                        _disc_cache.set(_disc_cache_key, {
                            "specs_export": _disc.export_specs(),
                            "failures": [
                                r for r in _disc.report() if r.get("rejected")
                            ],
                            "filter_drops": _disc.filter_drops(),
                        })
                        metadata.setdefault("composite_target_cache", {}) \
                            .setdefault(str(_tt_disc), {})[_tname_disc] = {
                                "hit": False, "key": _disc_cache_key,
                            }
                    except Exception as _cache_err:
                        logger.info(
                            "[CompositeTargetDiscovery] cache set failed for "
                            "target='%s' (%s); ignored.",
                            _tname_disc, _cache_err,
                        )

            # Apply frozen (train-fitted) params to ALL rows so the per-target loop has T for val/test.
            # NaN rows (domain violations on val/test) get imputed with median(T_train).
            from ..composite import get_transform as _get_transform_local
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
                    for _eb_name in _extra_bases:
                        _base_cols.append(
                            _build_full_column_from_splits(
                                _eb_name,
                                train_df_pd, val_df_pd, test_df_pd,
                                train_idx, val_idx, test_idx,
                                n_total=_y_arr.shape[0],
                            )
                        )
                    _base_full = np.column_stack(_base_cols)
                else:
                    _base_full = _base_primary
                _valid = _transform.domain_check(_y_arr, _base_full)
                _ct_t_full = np.full(_y_arr.shape[0], np.nan, dtype=np.float64)
                if _valid.any():
                    # For multi-base, _base_full is 2-D — pass the
                    # row-filtered 2-D slice; for single-base it stays 1-D.
                    _base_for_forward = (
                        _base_full[_valid, :] if _base_full.ndim == 2
                        else _base_full[_valid]
                    )
                    _ct_t_full[_valid] = _transform.forward(
                        _y_arr[_valid], _base_for_forward, _spec.fitted_params,
                    )
                if not np.all(np.isfinite(_ct_t_full)):
                    _t_train_for_median = _ct_t_full[filtered_train_idx]
                    _t_train_for_median = _t_train_for_median[
                        np.isfinite(_t_train_for_median)
                    ]
                    if _t_train_for_median.size > 0:
                        _ct_t_full[~np.isfinite(_ct_t_full)] = float(
                            np.median(_t_train_for_median)
                        )
                target_by_type[_tt_disc][_spec.name] = _ct_t_full
                logger.info(
                    "[CompositeTargetDiscovery] added composite target '%s' "
                    "to target_by_type[%s].", _spec.name, _tt_disc,
                )

    n_specs_total = sum(
        len(v) for tt_specs in metadata["composite_target_specs"].values()
        for v in tt_specs.values()
    )
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
    if n_specs_total > 0:
        logger.info(
            "[CompositeTargetDiscovery] %d composite target(s) added to "
            "target_by_type. They will be trained alongside raw targets in the "
            "per-target loop.", n_specs_total,
        )
        if _gpu_families:
            logger.warning(
                "[CompositeTargetDiscovery] composite mode + GPU training detected "
                "(%s) AND %d composite spec(s) shipped. GPU non-determinism is "
                "amplified by the K=%d extra fits; ensemble weights may drift across "
                "runs even with random_state fixed. Set deterministic=True / "
                "single_precision_histogram=True / force_row_wise=True on the inner "
                "estimators if reproducibility matters.",
                ", ".join(_gpu_families), n_specs_total, n_specs_total,
            )

    return target_by_type, metadata
