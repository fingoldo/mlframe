"""ACF/PACF-driven automatic lag-candidate selection for time-series feature engineering.

Choosing which lags to build as features is usually done by guessing a fixed grid (1, 2, 3, 7, 14, 30, ...).
A statistically principled alternative: compute the sample PACF (the DIRECT correlation at each lag, after
controlling for shorter lags -- the ACF alone double-counts propagated correlation) and only propose lags
whose PACF clears the Bartlett white-noise significance band, pruning the lag-search space before an
expensive MRMR/composite-target-discovery pass. Reuses the existing FFT-ACF / Durbin-Levinson-PACF kernels
from the diagnostics-panel machinery rather than reimplementing them.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from mlframe.reporting.charts import pacf_levinson, significance_band


def select_significant_lags(
    series: np.ndarray,
    max_lag: int = 50,
    alpha: float = 0.05,
    max_candidates: Optional[int] = None,
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
        significant lag.

    Returns
    -------
    dict
        ``significant_lags`` (list[int], ascending), ``pacf_values`` (dict[int, float] for every lag 1..k),
        ``significance_band`` (float, the +-threshold used).
    """
    if alpha != 0.05:
        raise ValueError(f"select_significant_lags: only alpha=0.05 (Bartlett 95% band) is currently supported; got {alpha!r}")

    pacf_vals, n_used = pacf_levinson(series, nlags=max_lag)
    band = significance_band(n_used)

    pacf_by_lag = {lag: float(pacf_vals[lag - 1]) for lag in range(1, pacf_vals.size + 1)}
    significant: List[int] = [lag for lag, v in pacf_by_lag.items() if abs(v) > band]
    significant.sort(key=lambda lag: -abs(pacf_by_lag[lag]))
    if max_candidates is not None:
        significant = significant[:max_candidates]
    significant.sort()

    return {"significant_lags": significant, "pacf_values": pacf_by_lag, "significance_band": band}


__all__ = ["select_significant_lags"]
