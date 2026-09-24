"""Detect a target that is still accruing at extract time (right-censoring), and separate it from regime change.

Why this exists
---------------
A cumulative or duration-valued target ("total charge so far", "hours until hired") is not fully observed for rows
created shortly before the data was extracted: those rows have had less time to accumulate. The marginal mean then
declines toward the present, and because a temporal split puts the newest rows in TEST, the model is trained on mature
rows and scored on immature ones. The symptom is a negative test R2 that no amount of modelling can fix, because the
label itself is unfinished.

A production run showed exactly this: ``target_total_charge`` with train/val/test means of 85.65 / 57.57 / 35.07 and
p99 of 1640 / 1164 / 601, a test R2 of -0.080, and a temporal audit that responded by recommending training on the
MOST RECENT (i.e. least mature) segment.

What can and cannot be decided from the marginal
------------------------------------------------
Right-censoring and a declining regime produce the SAME declining marginal, and the bin index carries no information
that separates them -- calendar time and remaining-observation-window are perfectly collinear. Truncating the newest
bins does not separate them either: for a continuously accruing target EVERY bin is partially censored, in proportion
to its age, so the trend survives truncation just as a regime change would.

One constraint does distinguish them, and it is a property of censoring rather than of the time axis. A row observed
for zero elapsed time has accrued nothing, so under pure censoring the per-bin statistic must extrapolate to ~0 at
age 0. A declining regime carries no such constraint: its newest rows still realise a non-zero level. So the audit
fits the per-bin statistic against each bin's AGE (extract time minus bin time) and reports where that fit crosses
age 0, as a fraction of the target's own level::

    origin_ratio = fitted_value_at_age_0 / max(bin statistic)

Near 0 is the censoring signature; a substantial fraction is a level the target still reaches when barely observed,
which censoring cannot produce.

The verdict names the finding and its evidence rather than asserting a cause the marginal cannot establish: the
operator knows whether their target accrues, and that fact plus ``origin_ratio`` settles it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "MaturityAuditResult",
    "audit_target_maturity",
    "audit_binned_target_maturity",
    "DEFAULT_MATURITY_TREND_THRESHOLD",
    "DEFAULT_MATURITY_ORIGIN_RATIO_THRESHOLD",
    "DEFAULT_MATURITY_MIN_BINS",
]

DEFAULT_MATURITY_TREND_THRESHOLD: float = 0.6
"""|Spearman(bin order, bin statistic)| above which the series counts as trending toward the present. Rank-based, so
the heavy tails these targets carry cannot drive it; 0.6 over ~25 bins is a clearly monotone move, not noise."""

DEFAULT_MATURITY_ORIGIN_RATIO_THRESHOLD: float = 0.25
"""``origin_ratio`` at or below which the decline is consistent with pure right-censoring: a target extrapolating to
under a quarter of its level at a zero observation window is behaving like something that accrues from nothing."""

DEFAULT_MATURITY_MIN_BINS: int = 8
"""Fewer populated bins than this cannot support the fit; the verdict would rest on a handful of points."""


@dataclass
class MaturityAuditResult:
    """Verdict of :func:`audit_target_maturity` / :func:`audit_binned_target_maturity`.

    ``verdict`` is one of:

    * ``"stable"`` -- no monotone decline toward the present to explain;
    * ``"censoring_likely"`` -- declines toward the present AND extrapolates to ~0 at a zero observation window;
    * ``"declining_level"`` -- declines toward the present but still reaches a substantial level at a zero window,
      which censoring alone cannot produce (regime change / selection shift);
    * ``"insufficient_data"``.
    """

    verdict: str
    trend: float
    """Spearman correlation between bin order (oldest -> newest) and the per-bin statistic. Negative = declines."""
    origin_ratio: float
    """Weighted linear fit of the statistic against bin age, evaluated at age 0, divided by the largest bin statistic.
    Near 0 under right-censoring; ``nan`` when the fit is not defined (including a statistic that does not grow with
    the observation window at all, which rules censoring out)."""
    n_bins: int
    bin_stats: list[float] = field(default_factory=list)
    bin_counts: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_censoring_likely(self) -> bool:
        """Whether the evidence is consistent with a target still accruing at extract time."""
        return self.verdict == "censoring_likely"


def _average_ranks(a: np.ndarray) -> np.ndarray:
    """Ranks of ``a`` with ties receiving their group's average rank."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(a.size, dtype=np.float64)
    sorted_a = a[order]
    start = 0
    for i in range(1, a.size + 1):
        if i == a.size or sorted_a[i] != sorted_a[start]:
            if i - start > 1:
                ranks[order[start:i]] = np.mean(ranks[order[start:i]])
            start = i
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation of two equal-length vectors; NaN when either side is constant or shorter than 3."""
    if x.size < 3 or y.size != x.size:
        return float("nan")
    rx, ry = _average_ranks(x), _average_ranks(y)
    sx, sy = rx.std(), ry.std()
    if sx <= 0 or sy <= 0:
        return float("nan")
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


def _origin_ratio(stats: np.ndarray, ages: np.ndarray, weights: np.ndarray) -> float:
    """Weighted least-squares fit of ``stats`` on ``ages`` evaluated at age 0, as a fraction of ``max|stats|``.

    Ages enter in their own units and cancel out of the ratio, so the result is unit-free. A non-positive slope means
    the statistic does not grow with the observation window at all, which rules censoring out; that returns NaN rather
    than a meaningless intercept.
    """
    scale = float(np.max(np.abs(stats))) if stats.size else 0.0
    if scale <= 0 or ages.size != stats.size or ages.size < 3:
        return float("nan")
    total_w = float(weights.sum())
    w = (weights / total_w) if total_w > 0 else np.full(stats.size, 1.0 / stats.size)
    a_mean = float(np.sum(w * ages))
    s_mean = float(np.sum(w * stats))
    var_a = float(np.sum(w * (ages - a_mean) ** 2))
    if var_a <= 0:
        return float("nan")
    slope = float(np.sum(w * (ages - a_mean) * (stats - s_mean)) / var_a)
    if slope <= 0:
        return float("nan")
    return float((s_mean - slope * a_mean) / scale)


def audit_binned_target_maturity(
    *,
    bin_stats: Any,
    bin_ages: Optional[Any] = None,
    bin_counts: Optional[Sequence[int]] = None,
    trend_threshold: float = DEFAULT_MATURITY_TREND_THRESHOLD,
    origin_ratio_threshold: float = DEFAULT_MATURITY_ORIGIN_RATIO_THRESHOLD,
    min_bins: int = DEFAULT_MATURITY_MIN_BINS,
    target_name: str = "",
) -> MaturityAuditResult:
    """Verdict from per-time-bin statistics a caller has already aggregated, OLDEST BIN FIRST.

    Callers that bin the target anyway -- the temporal audit does -- get the censoring verdict for the cost of a rank
    correlation and a 1-D weighted fit, with no second pass over the rows.

    ``bin_ages`` is each bin's observation window (extract time minus bin time), oldest bin largest, in any consistent
    unit. When omitted, evenly spaced ages are assumed, which is correct for equal-width time bins.
    """
    stats = np.asarray(bin_stats, dtype=np.float64).reshape(-1)
    counts = [int(c) for c in bin_counts] if bin_counts is not None else []
    label = f" {target_name!r}" if target_name else ""
    n = stats.size
    if n < min_bins or not np.all(np.isfinite(stats)):
        return MaturityAuditResult(
            verdict="insufficient_data",
            trend=float("nan"),
            origin_ratio=float("nan"),
            n_bins=n,
            bin_stats=stats.tolist(),
            bin_counts=counts,
            warnings=[f"maturity audit for target{label} needs at least {min_bins} finite bins; got {int(np.sum(np.isfinite(stats)))}."],
        )

    positions = np.arange(n, dtype=np.float64)
    trend = _spearman(positions, stats)
    ages = positions[::-1].astype(np.float64)
    if bin_ages is not None:
        _a = np.asarray(bin_ages, dtype=np.float64).reshape(-1)
        if _a.size == n and np.all(np.isfinite(_a)):
            ages = _a
    weights = np.asarray(counts, dtype=np.float64) if len(counts) == n else np.ones(n, dtype=np.float64)
    ratio = _origin_ratio(stats, ages, weights)

    warnings: list[str] = []
    if not np.isfinite(trend) or trend > -trend_threshold:
        verdict = "stable"
    elif np.isfinite(ratio) and ratio <= origin_ratio_threshold:
        verdict = "censoring_likely"
        warnings.append(
            f"target{label} declines monotonically toward the present (Spearman(bin, stat)={trend:+.2f}) and "
            f"extrapolates to {ratio:.0%} of its level at a zero observation window. A quantity that accrues from "
            f"nothing behaves exactly this way while it is still accruing at extract time, so the newest rows carry "
            f"UNFINISHED labels. A temporal split puts those rows in TEST, which makes test metrics measure the "
            f"censoring rather than the model -- expect a negative test R2 that no modelling can remove. Confirm "
            f"whether this target accrues over time; if it does, exclude the immature tail from BOTH training and "
            f"evaluation, or model the target as censored. The most-recent rows are then the WORST available "
            f"training window, not the best."
        )
    else:
        verdict = "declining_level"
        _ratio_txt = f"{ratio:.0%}" if np.isfinite(ratio) else "undefined (the statistic does not grow with the observation window)"
        warnings.append(
            f"target{label} declines monotonically toward the present (Spearman(bin, stat)={trend:+.2f}) but "
            f"extrapolates to {_ratio_txt} of its level at a zero observation window, which right-censoring alone "
            f"cannot produce. Read it as a regime change or a selection shift in the data source rather than as an "
            f"unmatured tail."
        )
    return MaturityAuditResult(
        verdict=verdict, trend=float(trend), origin_ratio=float(ratio), n_bins=n,
        bin_stats=stats.tolist(), bin_counts=counts, warnings=warnings,
    )


def audit_target_maturity(
    *,
    timestamps: Any,
    y: Any,
    n_bins: int = 25,
    statistic: str = "mean",
    trend_threshold: float = DEFAULT_MATURITY_TREND_THRESHOLD,
    origin_ratio_threshold: float = DEFAULT_MATURITY_ORIGIN_RATIO_THRESHOLD,
    min_bins: int = DEFAULT_MATURITY_MIN_BINS,
    target_name: str = "",
) -> MaturityAuditResult:
    """Bin ``y`` over time and decide whether its decline toward the present looks like right-censoring.

    Parameters
    ----------
    timestamps
        Row timestamps, any type numpy can sort (datetime64, int, float). Used for ordering and for each bin's age.
    y
        Target values on the same rows. Non-finite rows are dropped.
    n_bins
        Equal-count bins over the time order, so a sparse early period cannot produce one-row bins of pure noise.
    statistic
        Per-bin statistic to trend: ``"mean"`` (default) or ``"p99"``. The upper quantile responds to censoring more
        sharply on a zero-inflated target, where most rows are 0 in every bin and the mean moves little.
    trend_threshold, origin_ratio_threshold, min_bins, target_name
        See the constant docstrings and :func:`audit_binned_target_maturity`.
    """
    ts = np.asarray(timestamps).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    if ts.shape[0] != yy.shape[0]:
        raise ValueError(f"audit_target_maturity: timestamps has {ts.shape[0]} rows but y has {yy.shape[0]}.")
    finite = np.isfinite(yy)
    ts, yy = ts[finite], yy[finite]
    label = f" {target_name!r}" if target_name else ""
    if ts.size < min_bins * 3:
        return MaturityAuditResult(
            verdict="insufficient_data", trend=float("nan"), origin_ratio=float("nan"), n_bins=0,
            warnings=[f"maturity audit for target{label} needs at least {min_bins * 3} finite rows; got {ts.size}."],
        )

    order = np.argsort(ts, kind="mergesort")
    yy, ts = yy[order], ts[order]
    groups = [g for g in np.array_split(np.arange(yy.size), max(1, int(n_bins))) if g.size > 0]
    if statistic == "p99":
        stats = np.array([float(np.quantile(yy[g], 0.99)) for g in groups], dtype=np.float64)
    else:
        stats = np.array([float(np.mean(yy[g])) for g in groups], dtype=np.float64)
    counts = [int(g.size) for g in groups]
    # Bin age = how far this bin sits before the newest row in the data. Taken as timestamp DIFFERENCES so a
    # datetime64 axis and a numeric one both work, and the unit cancels inside origin_ratio.
    t_extract = ts[-1]
    ages = np.array([float(np.median((t_extract - ts[g]).astype("float64"))) for g in groups], dtype=np.float64)

    return audit_binned_target_maturity(
        bin_stats=stats, bin_ages=ages, bin_counts=counts,
        trend_threshold=trend_threshold, origin_ratio_threshold=origin_ratio_threshold,
        min_bins=min_bins, target_name=target_name,
    )
