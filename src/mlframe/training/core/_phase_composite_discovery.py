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

from ..baseline_diagnostics import BaselineDiagnostics
from ..composite import CompositeTargetDiscovery
from ..composite_cache import (
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

            _y_train_aligned = _y_arr[filtered_train_idx]
            if len(_y_train_aligned) != len(filtered_train_df):
                logger.warning(
                    "[CompositeTargetDiscovery] target='%s' row-align mismatch "
                    "(y[%d] vs filtered_train_df[%d]); skipping discovery.",
                    _tname_disc, len(_y_train_aligned), len(filtered_train_df),
                )
                continue

            if isinstance(filtered_train_df, pd.DataFrame):
                _disc_df = filtered_train_df.copy(deep=False)
                _disc_df[_tname_disc] = _y_train_aligned
            else:
                _disc_df = filtered_train_df.with_columns(
                    pl.Series(_tname_disc, _y_train_aligned)
                )

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
                    _df_sig = data_signature(
                        _disc_df, _tname_disc, _disc_feature_cols,
                        random_state=int(getattr(_disc_cfg, "random_state", 42) or 42),
                    )
                    _cfg_sig = _discovery_config_signature(_disc_cfg)
                    # random_state is already folded into _df_sig (seeds the row-sample) and into
                    # _cfg_sig (via the dataclass dump). Passing it again to make_discovery_cache_key
                    # was a double-fold (DISC-RANDOM-STATE-DBL): two semantically identical inputs
                    # both contribute to the key, so the same data + same config but with
                    # random_state mutated would produce three independent hash mixes. Drop the
                    # outer fold; pass 0 as the back-compat sentinel.
                    _disc_cache_key = make_discovery_cache_key(
                        _df_sig, _tname_disc, _cfg_sig,
                        random_state=0,
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
                metadata["composite_target_specs"].setdefault(str(_tt_disc), {})
                metadata["composite_target_specs"][str(_tt_disc)][_tname_disc] = (
                    _cached_payload.get("specs_export", [])
                )
                metadata["composite_target_failures"].setdefault(str(_tt_disc), {})
                metadata["composite_target_failures"][str(_tt_disc)][_tname_disc] = (
                    _cached_payload.get("failures", [])
                )
                metadata.setdefault("composite_target_filter_drops", {})
                metadata["composite_target_filter_drops"].setdefault(str(_tt_disc), {})
                metadata["composite_target_filter_drops"][str(_tt_disc)][_tname_disc] = (
                    _cached_payload.get("filter_drops", {})
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
                    from ..composite_spec import CompositeSpec as _Spec
                    _cached_specs = [
                        _Spec(**s) if isinstance(s, dict) else s
                        for s in _cached_payload.get("specs_export", [])
                    ]
                except Exception:
                    _cached_specs = []

                class _CacheReplay:
                    specs_ = _cached_specs

                _disc = _CacheReplay()
            else:
                try:
                    _disc_instance = CompositeTargetDiscovery(_disc_cfg)
                    if _use_hint and _diag is not None and _hint_strengths:
                        _disc_instance._hint_strengths_pct = _hint_strengths
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
                        _disc = _disc_instance.fit(
                            df=_disc_df,
                            target_col=_tname_disc,
                            feature_cols=_disc_feature_cols,
                            train_idx=_disc_train_idx,
                        )
                except Exception as _disc_err:
                    logger.warning(
                        "[CompositeTargetDiscovery] fit failed for target='%s': %s. "
                        "Per-target training continues without composite expansion.",
                        _tname_disc, _disc_err,
                    )
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
