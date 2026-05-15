"""Per-target pre-training diagnostics: label-distribution drift + baseline diagnostics.

Both run before model training so operators catch selection-bias / temporal-prior-shift and get a headline-metric estimate without waiting for full model fits.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..baseline_diagnostics import BaselineDiagnostics, format_baseline_diagnostics_report
from ..drift_report import compute_label_distribution_drift, format_drift_report

logger = logging.getLogger(__name__)


def run_per_target_diagnostics(
    *,
    target_type,
    cur_target_name: str,
    current_train_target,
    current_val_target,
    current_test_target,
    filtered_train_df,
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

    # Stored on metadata so composite-target discovery can gate on composite_recommendation.
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
            metadata.setdefault("baseline_diagnostics", {}) \
                .setdefault(str(target_type), {})[cur_target_name] = _bd_report.to_dict()
    except Exception as _bd_err:
        logger.warning(
            "baseline_diagnostics failed for target='%s' (%s): %s. "
            "Training continues without diagnostics.",
            cur_target_name, target_type, _bd_err,
        )

    return metadata
