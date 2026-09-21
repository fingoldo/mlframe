"""``_audit_from_agg`` -- temporal audit on a pre-aggregated frame.

Split out from ``training/target_temporal_audit.py`` to keep that file below the 1k-line monolith
threshold. Behaviour preserved bit-for-bit; the helper is re-exported from ``target_temporal_audit``.
"""
from __future__ import annotations

import logging
from itertools import takewhile
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from ._target_temporal_audit_aggregate import Granularity
    from .target_temporal_audit import ChangePointMethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Target types whose per-bin ``target_rate`` is a probability in [0, 1]. For those the drift threshold is a
# percentage-point difference and compares directly against the spread. Every other type carries the target's own
# units, where an absolute comparison reduces to "is this target's scale bigger than the threshold".
_PROBABILITY_RATE_TARGET_TYPES: frozenset[str] = frozenset(
    {"binary_classification", "multiclass_classification", "multilabel_classification"}
)


def _bin_ages(kept_bins: list) -> list[float]:
    """Per-bin observation window (newest bin start minus this bin's start) in float units, oldest bin largest.

    Numeric differences so a datetime64 / Timestamp axis and a plain numeric one both work; the unit cancels inside
    the maturity fit. Returns an empty list when the starts cannot be differenced, which makes the audit fall back to
    evenly spaced ages.
    """
    if not kept_bins:
        return []
    try:
        starts = np.asarray([b.bin_start for b in kept_bins])
        return [float(v) for v in (starts[-1] - starts).astype("float64")]
    except (TypeError, ValueError) as exc:
        logger.debug("bin_start values are not differenceable (%s); maturity audit falls back to even spacing", exc)
        return []


def _rate_is_probability(target_type: str) -> bool:
    """Whether ``target_rate`` for this target type is a probability, which decides absolute-vs-relative drift comparison."""
    return str(target_type) in _PROBABILITY_RATE_TARGET_TYPES


def _spread_exceeds_threshold(mean_rates: list[float], target_type: str, drift_warn_threshold: float) -> tuple[float, bool]:
    """``(relative_spread, exceeds)`` for the segment mean rates under the right scale for ``target_type``.

    Probability-valued rates compare their absolute spread against the threshold as percentage points. Unbounded rates
    compare the spread against the segment level instead: an absolute comparison there fires for any target whose units
    are larger than the threshold, which in a production run flagged a 5%-spread target against a 10% threshold.
    The level uses max|rate| rather than the mean so a segment set straddling zero cannot divide by ~0.
    """
    spread_abs = max(mean_rates) - min(mean_rates)
    if _rate_is_probability(target_type):
        return (spread_abs / abs(max(mean_rates)) if max(mean_rates) else float("nan")), spread_abs > drift_warn_threshold
    level = max(abs(r) for r in mean_rates)
    if level <= 0 or not np.isfinite(level):
        return float("nan"), False
    spread_rel = spread_abs / level
    return spread_rel, spread_rel > drift_warn_threshold


# The dataclasses + label helper live in the parent module; the change-point detectors live in a
# sibling. Both parent/sibling finish their top-level loading BEFORE this module is pulled in via the
# parent's bottom-of-file re-export, so the partial-load imports below resolve cleanly.
from .target_temporal_audit import (
    TemporalAuditResult,
    TimeBin,
    _format_bin_label,
)
from .target_maturity_audit import audit_binned_target_maturity
from ._target_temporal_changepoint import (
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
                (f"only {len(kept_bins)} non-sparse bins after the {min_bin_fraction}x median-n_obs filter "
                f"- too few for a temporal audit. Consider a finer granularity or a longer time span."),
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
        if mean_rates:
            spread_abs = max(mean_rates) - min(mean_rates)
            spread_rel, unstable = _spread_exceeds_threshold(mean_rates, target_type, drift_warn_threshold)
            if unstable:
                _scale_note = (
                    f"spread {spread_abs:.3f} > {drift_warn_threshold:.2f} (rates are probabilities, so the threshold is absolute)"
                    if _rate_is_probability(target_type)
                    else f"relative spread {spread_rel:.1%} > {drift_warn_threshold:.1%} (absolute spread {spread_abs:.3f} is in target units, "
                    f"so it is compared against the segment level rather than against the threshold directly)"
                )
                warnings.append(
                    f"target rate is NOT stable over time: detected {len(segments)} segments "
                    f"with mean rates ranging {min(mean_rates):.3f}..{max(mean_rates):.3f} "
                    f"({_scale_note}). "
                    f"Likely causes: (a) selection-bias in your data source over time, "
                    f"(b) regime change in the underlying generative process, (c) target "
                    f"definition shift, (d) a still-accruing target whose newest rows have not matured "
                    f"(check the dropped-bin note below -- monotone decline toward the present is its signature). "
                    f"See segment list below for cutoff dates."
                )
                warnings.extend(
                    f"  segment {s['start_label']}..{s['end_label']} " f"({s['n_bins']} bins, n_obs={s['n_obs']:_}): " f"mean_rate={s['mean_rate']:.3f}"
                    for s in segments
                )

    dropped_bins = [b for b in bins if not b.kept]
    n_dropped = len(dropped_bins)
    _trailing = 0
    if n_dropped > 0:
        kept_labels = [b.bin_label for b in kept_bins]
        # Name the dropped range explicitly. A stability verdict is normally read against the evaluation split, and
        # when the thin bins are the RECENT ones the audit covers a window that ends before that split does -- a
        # production run dropped 5 trailing bins and reported stability over a window ending five weeks before the
        # test period ended. A target still accruing at extract time produces exactly that bin profile.
        _dropped_labels = [b.bin_label for b in dropped_bins]
        _trailing = sum(1 for _ in takewhile(lambda b: not b.kept, reversed(bins)))
        _coverage = (
            f"the audit therefore covers {kept_labels[0]}..{kept_labels[-1]} only" if kept_labels else "no bins remain"
        )
        _recency_note = (
            f" {_trailing} of them are the MOST RECENT bin(s); any verdict below describes a window that ENDS BEFORE "
            f"your newest data, and a target still accruing at extract time thins its newest bins exactly this way."
            if _trailing
            else ""
        )
        warnings.append(
            f"{n_dropped} bin(s) dropped from the audit (n_obs < "
            f"{int(threshold_n):_} = {min_bin_fraction}x median bin size): {', '.join(_dropped_labels)}; "
            f"{_coverage}.{_recency_note} "
            "If this number is large, consider a wider granularity."
        )

    most_recent_stable: dict[str, Any] | None = None
    if segments:
        for s in reversed(segments):
            if s["n_bins"] >= 3:
                most_recent_stable = s
                break

    # Reuse the bins just computed to ask whether a decline toward the present is a still-accruing (right-censored)
    # target rather than a regime change. It matters for the recommendation below: under censoring the most-recent
    # segment is the LEAST mature one, so recommending it as the training window makes the problem worse.
    maturity = audit_binned_target_maturity(
        bin_stats=[b.target_rate for b in kept_bins],
        bin_ages=_bin_ages(kept_bins),
        bin_counts=[b.n_obs for b in kept_bins],
        target_name=target_name,
    )
    warnings.extend(maturity.warnings if maturity.verdict == "censoring_likely" else [])
    actionable: dict[str, Any] = {
        "n_segments": len(segments),
        "most_recent_stable_segment": most_recent_stable,
        "n_bins_dropped": n_dropped,
        "n_trailing_bins_dropped": _trailing,
        "audit_covers": ((kept_bins[0].bin_label, kept_bins[-1].bin_label) if kept_bins else None),
        "maturity_verdict": maturity.verdict,
        "maturity_origin_ratio": maturity.origin_ratio,
        "maturity_trend": maturity.trend,
    }
    if most_recent_stable is not None and len(segments) >= 2:
        _maturity_caveat = (
            f" CAUTION: this target's bin rates decline toward the present and extrapolate to "
            f"{maturity.origin_ratio:.0%} of their level at a zero observation window, which is what a still-accruing "
            f"(right-censored) target looks like. If that is the cause, the most-recent segment is the LEAST mature "
            f"one and restricting training to it would make the problem worse -- exclude the immature tail instead."
            if maturity.verdict == "censoring_likely"
            else ""
        )
        actionable["recommendation"] = (
            f"Consider restricting training to the most-recent stable segment "
            f"({most_recent_stable['start_label']}..{most_recent_stable['end_label']}, "
            f"n_obs={most_recent_stable['n_obs']:_}, mean_rate={most_recent_stable['mean_rate']:.3f}) "
            f"or pair the suite with PULearningWrapper if earlier segments are "
            f"selection-biased rather than wrong.{_maturity_caveat}"
        )

    return TemporalAuditResult(
        target_name=target_name, target_type=target_type,
        timestamp_col=timestamp_col, granularity=granularity,
        bins=bins, change_point_indices=boundaries, segments=segments,
        warnings=warnings, actionable=actionable,
    )
