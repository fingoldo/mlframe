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
from ..configs import TargetTypes
from .utils import (
    _build_full_column_from_splits,
    _defensive_copy_and_expand_multilabel_regression,
    _init_composite_discovery_metadata,
)

logger = logging.getLogger(__name__)


def run_composite_target_discovery(
    *,
    composite_target_discovery_config,
    target_by_type: Dict,
    mlframe_models: Optional[List[str]],
    metadata: Dict,
    filtered_train_df,
    filtered_train_idx,
    train_df_pd,
    val_df_pd,
    test_df_pd,
    train_idx,
    val_idx,
    test_idx,
    baseline_diagnostics_config,
    cat_features: Optional[List[str]],
    verbose: bool,
) -> Tuple[Dict, Dict]:
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

    # Feature columns from OD-filtered frame for discovery row alignment.
    try:
        _disc_feature_cols = list(filtered_train_df.columns)
    except Exception:
        _disc_feature_cols = list(train_df_pd.columns)

    # filtered_train_df is row-aligned to filtered_train_idx, so discovery gets
    # a contiguous range using the filtered frame directly as ``df``.
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

            # Auto-skip on baseline-optimal recommendation AND/OR ablation hint.
            # BaselineDiagnostics normally runs INSIDE the per-target loop below,
            # so on first pass the metadata key isn't populated yet. Run an inline
            # diagnostic here when either signal is enabled; result is cached so
            # the per-target loop reuses it (saves ~30-60s duplicate ablation).
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

            # Subset y to filtered_train_df rows (post split + OD).
            _y_train_aligned = _y_arr[filtered_train_idx]
            if len(_y_train_aligned) != len(filtered_train_df):
                logger.warning(
                    "[CompositeTargetDiscovery] target='%s' row-align mismatch "
                    "(y[%d] vs filtered_train_df[%d]); skipping discovery.",
                    _tname_disc, len(_y_train_aligned), len(filtered_train_df),
                )
                continue

            # Build working frame: filtered_train_df + target column.
            if isinstance(filtered_train_df, pd.DataFrame):
                _disc_df = filtered_train_df.copy(deep=False)
                _disc_df[_tname_disc] = _y_train_aligned
            else:
                _disc_df = filtered_train_df.with_columns(
                    pl.Series(_tname_disc, _y_train_aligned)
                )

            # If hint enabled and BD ran, derive per-target config with
            # ``dominant_features_hint`` from ablation top-K.
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
                # Per-hint ablation strength (delta_pct) so discovery can adapt
                # cap to hint strength: strong hints take all top_k slots, weak
                # hints fall back to half-slot cap.
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

            try:
                _disc_instance = CompositeTargetDiscovery(_disc_cfg)
                if _use_hint and _diag is not None and _hint_strengths:
                    _disc_instance._hint_strengths_pct = _hint_strengths
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

            # Persist specs, failures, and pre-MI filter drops.
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

            # Apply each discovered spec's transform to the FULL row space
            # (train+val+test). Discovery fitted params on filtered_train_idx only
            # (leakage discipline); we apply frozen params to all rows so the
            # per-target training loop has T values for val/test. NaN rows (domain
            # violations on val/test) get imputed with median(T_train).
            from ..composite import get_transform as _get_transform_local
            for _spec in _disc.specs_:
                _transform = _get_transform_local(_spec.transform_name)
                _base_full = _build_full_column_from_splits(
                    _spec.base_column,
                    train_df_pd, val_df_pd, test_df_pd,
                    train_idx, val_idx, test_idx,
                    n_total=_y_arr.shape[0],
                )
                _valid = _transform.domain_check(_y_arr, _base_full)
                _ct_t_full = np.full(_y_arr.shape[0], np.nan, dtype=np.float64)
                if _valid.any():
                    _ct_t_full[_valid] = _transform.forward(
                        _y_arr[_valid], _base_full[_valid], _spec.fitted_params,
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

    # One-liner summary so operators see impact at suite level.
    n_specs_total = sum(
        len(v) for tt_specs in metadata["composite_target_specs"].values()
        for v in tt_specs.values()
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
