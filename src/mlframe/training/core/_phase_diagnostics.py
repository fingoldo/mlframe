"""Per-target pre-training diagnostics: label-distribution drift + baseline diagnostics.

Both run before model training so operators catch selection-bias / temporal-prior-shift and get a headline-metric estimate without waiting for full model fits.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..baseline_diagnostics import BaselineDiagnostics, format_baseline_diagnostics_report
from ..drift_report import compute_label_distribution_drift, format_drift_report
from ..feature_drift_report import compute_feature_distribution_drift

logger = logging.getLogger(__name__)


def run_per_target_diagnostics(
    *,
    target_type,
    cur_target_name: str,
    current_train_target,
    current_val_target,
    current_test_target,
    filtered_train_df,
    filtered_val_df=None,
    filtered_test_df=None,
    baseline_diagnostics_config,
    cat_features: list[str] | None,
    metadata: dict,
) -> dict:
    """Run drift report + baseline diagnostics for one target."""
    _drift_report = compute_label_distribution_drift(
        train_target=current_train_target,
        val_target=current_val_target,
        test_target=current_test_target,
        target_type=str(target_type),
    )
    logger.info(format_drift_report(_drift_report, target_name=cur_target_name))
    metadata.setdefault("label_distribution_drift", {}) \
        .setdefault(str(target_type), {})[cur_target_name] = _drift_report

    # BASELINE DIAGNOSTICS FIRST -- its ablation deltas are the feature
    # importance source we feed to the feature-drift sensor below. Per the
    # 2026-05-22 design review: feature drift WITHOUT importance weighting
    # isn't a grounded harm signal (drift on irrelevant features is harmless;
    # drift on dominant features can be catastrophic). So we always compute
    # FI first, then weight the drift report accordingly.
    _bd_report_dict: dict | None = None
    try:
        # Reuse cached result if composite-discovery already computed one for this pair (~30-60s saved).
        _existing_bd = (
            metadata.get("baseline_diagnostics", {})
            .get(str(target_type), {})
            .get(cur_target_name)
        )
        if _existing_bd is not None:
            logger.info(
                "[BaselineDiagnostics] target='%s' reusing cached diagnostic "
                "from composite-discovery precompute (saved ~30-60s).", cur_target_name,
            )
            _bd_report_dict = _existing_bd
        elif baseline_diagnostics_config.enabled and (
            str(target_type) in baseline_diagnostics_config.apply_to_target_types
        ):
            _bd = BaselineDiagnostics(baseline_diagnostics_config)
            _bd_report = _bd.fit_and_report(
                train_df=filtered_train_df,
                train_target=current_train_target,
                feature_cols=list(filtered_train_df.columns),
                target_type=str(target_type),
                target_name=cur_target_name,
                cat_features=cat_features,
            )
            logger.info(format_baseline_diagnostics_report(_bd_report, target_name=cur_target_name))
            _bd_report_dict = _bd_report.to_dict()
            metadata.setdefault("baseline_diagnostics", {}) \
                .setdefault(str(target_type), {})[cur_target_name] = _bd_report_dict
    except Exception as _bd_err:
        logger.warning(
            "baseline_diagnostics failed for target='%s' (%s): %s. "
            "Training continues without diagnostics.",
            cur_target_name, target_type, _bd_err,
        )

    # 2026-05-22: feature-side drift sensor. The actionable layer downstream
    # is the K=2 ensemble catastrophic-dropout; this sensor is COMPLEMENTARY
    # observability that (a) stamps per-feature drift stats into metadata for
    # post-mortem correlation and (b) escalates to WARN when the FI-weighted
    # aggregate (drift * dominance) crosses 1.0 -- the grounded harm signal
    # the design-review of 2026-05-22 demanded.
    if filtered_val_df is not None or filtered_test_df is not None:
        try:
            # Build feature_importance from baseline_diagnostics ablation deltas.
            # Each ablation entry is {"feature": str, "delta_pct": float, "rank": int};
            # delta_pct > 0 means "metric got worse when this feature was dropped"
            # i.e. the feature was important. abs(delta_pct) is the FI proxy.
            _fi: dict[str, float] | None = None
            if _bd_report_dict and isinstance(_bd_report_dict.get("ablation"), list):
                _fi = {}
                for _ab in _bd_report_dict["ablation"]:
                    if isinstance(_ab, dict) and "feature" in _ab:
                        _fi[str(_ab["feature"])] = float(abs(_ab.get("delta_pct", 0.0)))
                if not _fi:
                    _fi = None
            # Shape-detector signal for the classification override gate.
            # ``init_score_baseline.delta_vs_raw_pct`` measures how well the
            # linear baseline (LogReg on top-FI features) captures the
            # LightGBM raw metric. When |delta| is small the target is
            # linear-shape and the classification override is safe to
            # auto-apply. Available only when baseline_diagnostics computed
            # an init_score_baseline; for regression this signal is
            # unused (regression auto-apply doesn't need it).
            _linear_shape_delta: float | None = None
            if _bd_report_dict and isinstance(_bd_report_dict.get("init_score_baseline"), dict):
                _isb = _bd_report_dict["init_score_baseline"]
                _delta = _isb.get("delta_vs_raw_pct")
                if _delta is not None:
                    try:
                        _linear_shape_delta = float(_delta)
                    except (ValueError, TypeError):
                        _linear_shape_delta = None
            _fd_report = compute_feature_distribution_drift(
                train_df=filtered_train_df,
                val_df=filtered_val_df,
                test_df=filtered_test_df,
                feature_importance=_fi,
                target_type=str(target_type),
                linear_shape_delta_vs_raw_pct=_linear_shape_delta,
            )
            metadata.setdefault("feature_distribution_drift", {}) \
                .setdefault(str(target_type), {})[cur_target_name] = _fd_report
        except Exception as _fd_err:
            logger.warning(
                "feature_distribution_drift failed for target='%s' (%s); training "
                "continues without the feature-drift sensor.",
                cur_target_name, _fd_err,
            )

    return metadata
