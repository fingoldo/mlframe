"""ACF/PACF-driven automatic lag-candidate selection for time-series feature engineering.

Choosing which lags to build as features is usually done by guessing a fixed grid (1, 2, 3, 7, 14, 30, ...).
A statistically principled alternative: compute the sample PACF (the DIRECT correlation at each lag, after
controlling for shorter lags -- the ACF alone double-counts propagated correlation) and only propose lags
whose PACF clears the Bartlett white-noise significance band, pruning the lag-search space before an
expensive MRMR/composite-target-discovery pass. Reuses the existing FFT-ACF / Durbin-Levinson-PACF kernels
from the diagnostics-panel machinery rather than reimplementing them.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from mlframe.reporting.charts import pacf_levinson, significance_band


def select_significant_lags(
    series: np.ndarray,
    max_lag: int = 50,
    alpha: float = 0.05,
    max_candidates: Optional[int] = None,
    groups: Optional[np.ndarray] = None,
    min_group_fraction: float = 0.0,
    min_group_size: int = 10,
) -> dict:
    """Propose a lag shortlist from a series' sample PACF, ranked by |PACF| descending.

    Parameters
    ----------
    series
        ``(n,)`` numeric series (target, or per-entity target for a panel's single instrument).
    max_lag
        Largest lag considered.
    alpha
        Only ``0.05`` (the Bartlett 95% two-sided band) is currently supported; kept as an explicit param
        for future extension rather than a silently-fixed constant.
    max_candidates
        Cap the returned shortlist to this many lags (the strongest by |PACF|); ``None`` returns every
        significant lag. Applied per-group when ``groups`` is given (before the consensus union), not to
        the final consensus list.
    groups
        Opt-in per-group mode. ``(n,)`` array of entity/group labels aligned with ``series``. When given,
        PACF-significant lags are computed independently per group (panels can have materially different
        autocorrelation structure -- one entity driven by lag 3, another by lag 7 -- and a single global
        series washes out whichever structure is the minority by variance/length) and the returned
        shortlist is the consensus union: a lag qualifies if it is significant for at least
        ``min_group_fraction`` of the groups that had enough points to be scored. When ``None`` (default),
        behavior is exactly the original single global-series computation.
    min_group_fraction
        Only used when ``groups`` is given. Fraction (0..1) of scored groups a lag must be significant in
        to make the consensus shortlist. ``0.0`` (default) is a pure union -- any group's significant lag
        qualifies.
    min_group_size
        Only used when ``groups`` is given. Groups with fewer than this many points are skipped (too few
        points for a meaningful PACF at ``max_lag``) rather than silently producing a degenerate estimate.

    Returns
    -------
    dict
        ``significant_lags`` (list[int], ascending), ``pacf_values`` (dict[int, float] for every lag 1..k
        -- for the global series when ``groups`` is ``None``, else the group-wise average PACF among groups
        where the lag was scored), ``significance_band`` (float, the +-threshold used -- for the global
        series when ``groups`` is ``None``, else the average across scored groups).
        When ``groups`` is given, also includes ``per_group`` (dict[group_label, dict] -- each group's own
        full single-series result) and ``n_groups_scored``.
    """
    if alpha != 0.05:
        raise ValueError(f"select_significant_lags: only alpha=0.05 (Bartlett 95% band) is currently supported; got {alpha!r}")

    if groups is not None:
        return _select_significant_lags_per_group(
            series=series,
            groups=groups,
            max_lag=max_lag,
            alpha=alpha,
            max_candidates=max_candidates,
            min_group_fraction=min_group_fraction,
            min_group_size=min_group_size,
        )

    pacf_vals, n_used = pacf_levinson(series, nlags=max_lag)
    band = significance_band(n_used)

    pacf_by_lag = {lag: float(pacf_vals[lag - 1]) for lag in range(1, pacf_vals.size + 1)}
    significant: List[int] = [lag for lag, v in pacf_by_lag.items() if abs(v) > band]
    significant.sort(key=lambda lag: -abs(pacf_by_lag[lag]))
    if max_candidates is not None:
        significant = significant[:max_candidates]
    significant.sort()

    return {"significant_lags": significant, "pacf_values": pacf_by_lag, "significance_band": band}


def _select_significant_lags_per_group(
    series: np.ndarray,
    groups: np.ndarray,
    max_lag: int,
    alpha: float,
    max_candidates: Optional[int],
    min_group_fraction: float,
    min_group_size: int,
) -> dict:
    """Per-group PACF significant-lag selection with a consensus-union shortlist (see ``select_significant_lags``)."""
    series = np.asarray(series)
    groups = np.asarray(groups)
    if groups.shape[0] != series.shape[0]:
        raise ValueError(f"select_significant_lags: groups length {groups.shape[0]} != series length {series.shape[0]}")
    if not 0.0 <= min_group_fraction <= 1.0:
        raise ValueError(f"select_significant_lags: min_group_fraction must be in [0, 1]; got {min_group_fraction!r}")

    per_group: Dict[object, dict] = {}
    lag_support_count: Dict[int, int] = {}
    lag_pacf_sums: Dict[int, float] = {}
    lag_scored_count: Dict[int, int] = {}
    band_sum = 0.0

    for group_label in np.unique(groups):
        group_series = series[groups == group_label]
        if group_series.shape[0] < min_group_size:
            continue
        group_result = select_significant_lags(group_series, max_lag=min(max_lag, group_series.shape[0] - 1), alpha=alpha, max_candidates=max_candidates)
        per_group[group_label] = group_result
        band_sum += group_result["significance_band"]
        for lag in group_result["significant_lags"]:
            lag_support_count[lag] = lag_support_count.get(lag, 0) + 1
        for lag, v in group_result["pacf_values"].items():
            lag_pacf_sums[lag] = lag_pacf_sums.get(lag, 0.0) + v
            lag_scored_count[lag] = lag_scored_count.get(lag, 0) + 1

    n_groups_scored = len(per_group)
    if n_groups_scored == 0:
        raise ValueError(f"select_significant_lags: no group had >= min_group_size={min_group_size} points")

    min_support = min_group_fraction * n_groups_scored
    consensus: List[int] = sorted(lag for lag, count in lag_support_count.items() if count >= min_support)
    consensus.sort(key=lambda lag: -lag_support_count[lag])
    if max_candidates is not None:
        consensus = consensus[:max_candidates]
    consensus.sort()

    # Divide each lag's PACF sum by the number of groups that actually SCORED that lag (its own
    # ``max_lag = min(max_lag, group_series.shape[0] - 1)`` can be shorter than another group's), not the
    # fixed n_groups_scored total -- for panels with heterogeneous group lengths, a higher lag scored by only
    # a few long groups was silently diluted by the full group count, contradicting this function's own
    # documented "group-wise average PACF among groups where the lag was scored" contract.
    pacf_by_lag = {lag: total / lag_scored_count[lag] for lag, total in lag_pacf_sums.items()}

    return {
        "significant_lags": consensus,
        "pacf_values": pacf_by_lag,
        "significance_band": band_sum / n_groups_scored,
        "per_group": per_group,
        "n_groups_scored": n_groups_scored,
    }


__all__ = ["select_significant_lags"]
