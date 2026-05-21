"""``_audit_from_agg`` -- temporal audit on a pre-aggregated frame.

Wave 106c (2026-05-21): split out from ``training/target_temporal_audit.py``
to keep that file below the 1k-line monolith threshold. Behaviour preserved
bit-for-bit; the helper is re-exported from ``target_temporal_audit``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Wave 106c (2026-05-21): the dataclasses + label helper live in the parent
# module; the change-point detectors live in the wave-106 sibling. Both
# parents/siblings finish their top-level loading BEFORE this module is
# pulled in via the parent's bottom-of-file re-export, so the partial-load
# imports below resolve cleanly. Single source of truth.
from .target_temporal_audit import (  # noqa: E402
    TemporalAuditResult,
    TimeBin,
    _format_bin_label,
)
from ._target_temporal_changepoint import (  # noqa: E402
    find_change_points_pelt,
    find_change_points_zscore,
    _segments_from_change_points,
)


def _audit_from_agg(
    *,
    agg: pd.DataFrame,
    target_name: str,
    target_type: str,
    timestamp_col: str,
    granularity: Granularity,
    min_bin_fraction: float,
    method: ChangePointMethod,
    pelt_model: str,
    pelt_penalty: float | None,
    pelt_min_segment_size: int,
    z_threshold: float,
    z_window: int | None,
    min_anomaly_run: int,
    drift_warn_threshold: float,
) -> TemporalAuditResult:
    """Internal: turn a pre-computed (bin_start, n_obs, target_rate)
    aggregation into a full TemporalAuditResult. Shared between
    ``audit_target_over_time`` (single) and ``audit_targets_over_time``
    (batch) so the post-aggregation pipeline lives in one place.
    """
    if agg.empty:
        return TemporalAuditResult(
            target_name=target_name, target_type=target_type,
            timestamp_col=timestamp_col, granularity=granularity,
            bins=[], change_point_indices=[], segments=[],
            warnings=["empty aggregation - no data after time-binning"],
            actionable={},
        )

    median_n = float(agg["n_obs"].median())
    threshold_n = max(1.0, min_bin_fraction * median_n)
    agg = agg.copy()
    agg["kept"] = agg["n_obs"] >= threshold_n

    bins = [
        TimeBin(
            bin_label=_format_bin_label(row.bin_start, granularity),
            bin_start=row.bin_start,
            n_obs=int(row.n_obs),
            target_rate=float(row.target_rate),
            kept=bool(row.kept),
        )
        for row in agg.itertuples(index=False)
    ]

    kept_bins = [b for b in bins if b.kept]
    if len(kept_bins) < 3:
        return TemporalAuditResult(
            target_name=target_name, target_type=target_type,
            timestamp_col=timestamp_col, granularity=granularity,
            bins=bins, change_point_indices=[], segments=[],
            warnings=[
                f"only {len(kept_bins)} non-sparse bins after the {min_bin_fraction}Г— median-n_obs filter "
                f"- too few for a temporal audit. Consider a finer granularity or a longer time span.",
            ],
            actionable={},
        )

    rates = np.array([b.target_rate for b in kept_bins])
    weights = np.array([b.n_obs for b in kept_bins], dtype=float)
    labels = [b.bin_label for b in kept_bins]
    if method == "pelt":
        boundaries = find_change_points_pelt(
            rates, weights=weights,
            model=pelt_model, penalty=pelt_penalty,
            min_segment_size=pelt_min_segment_size,
        )
    else:
        boundaries = find_change_points_zscore(
            rates, weights=weights,
            window=z_window,
            z_threshold=z_threshold, min_anomaly_run=min_anomaly_run,
        )
    segments = _segments_from_change_points(rates, weights, boundaries, labels)

    warnings: list[str] = []
    if len(segments) >= 2:
        mean_rates = [s["mean_rate"] for s in segments if s["mean_rate"] == s["mean_rate"]]
        if mean_rates and max(mean_rates) - min(mean_rates) > drift_warn_threshold:
            warnings.append(
                f"target rate is NOT stable over time: detected {len(segments)} segments "
                f"with mean rates ranging {min(mean_rates):.3f}..{max(mean_rates):.3f} "
                f"(spread {(max(mean_rates) - min(mean_rates)):.3f}, threshold {drift_warn_threshold:.2f}). "
                f"Likely causes: (a) selection-bias in your data source over time, "
                f"(b) regime change in the underlying generative process, (c) target "
                f"definition shift. See segment list below for cutoff dates."
            )
            for s in segments:
                warnings.append(
                    f"  segment {s['start_label']}..{s['end_label']} "
                    f"({s['n_bins']} bins, n_obs={s['n_obs']:_}): "
                    f"mean_rate={s['mean_rate']:.3f}"
                )

    n_dropped = sum(1 for b in bins if not b.kept)
    if n_dropped > 0:
        warnings.append(
            f"{n_dropped} bin(s) dropped from the audit (n_obs < "
            f"{int(threshold_n):_} = {min_bin_fraction}Г— median bin size). "
            "Typically the partial first / last bins of your time range; "
            "if this number is large, consider a wider granularity."
        )

    most_recent_stable: dict[str, Any] | None = None
    if segments:
        for s in reversed(segments):
            if s["n_bins"] >= 3:
                most_recent_stable = s
                break

    actionable: dict[str, Any] = {
        "n_segments": len(segments),
        "most_recent_stable_segment": most_recent_stable,
    }
    if most_recent_stable is not None and len(segments) >= 2:
        actionable["recommendation"] = (
            f"Consider restricting training to the most-recent stable segment "
            f"({most_recent_stable['start_label']}..{most_recent_stable['end_label']}, "
            f"n_obs={most_recent_stable['n_obs']:_}, mean_rate={most_recent_stable['mean_rate']:.3f}) "
            f"or pair the suite with PULearningWrapper if earlier segments are "
            f"selection-biased rather than wrong."
        )

    return TemporalAuditResult(
        target_name=target_name, target_type=target_type,
        timestamp_col=timestamp_col, granularity=granularity,
        bins=bins, change_point_indices=boundaries, segments=segments,
        warnings=warnings, actionable=actionable,
    )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
