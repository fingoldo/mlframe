"""``drop_near_noise_univariate_auc``: cheap univariate-AUC prescreen before MRMR/DCD.

Source: 4th_santander-customer-transaction-prediction.md -- "I removed some vars from train which predictions
by long model had AUC near .5 (before grouping)." A feature whose OWN univariate AUC sits at chance carries
essentially no linear/monotone signal about the target in isolation -- dropping it before the expensive
MRMR/DCD redundancy-aware search is a cheap first-pass filter for independent-feature datasets (won't catch a
feature that's only informative in COMBINATION with others, which is exactly what MRMR itself is for; this
is a pre-filter, not a replacement).

Reuses ``preprocessing.align_feature_direction.batch_univariate_auc`` (the same vectorized rank-based AUC
computation added for the feature-direction-alignment entry) rather than reimplementing per-column AUC.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from mlframe.preprocessing.align_feature_direction import batch_univariate_auc


def drop_near_noise_univariate_auc(
    df: pd.DataFrame,
    y: np.ndarray,
    columns: Optional[Sequence[str]] = None,
    tolerance: float = 0.02,
    n_bootstrap: Optional[int] = None,
    bootstrap_frac: float = 0.8,
    bootstrap_percentile_band: Tuple[float, float] = (10.0, 90.0),
    random_state: Optional[int] = None,
) -> List[str]:
    """Return column names whose univariate AUC against ``y`` falls within ``tolerance`` of chance (0.5).

    Parameters
    ----------
    df
        Feature frame.
    y
        Binary target, same row order as ``df``.
    columns
        Columns to screen; defaults to every numeric column of ``df``.
    tolerance
        A column is flagged when ``abs(auc - 0.5) <= tolerance``.
    n_bootstrap
        Opt-in resampling-stability mode. When ``None`` (default), behavior is the original single-pass
        AUC check computed once over the full ``df``. When set to a positive int, the AUC is instead
        recomputed on ``n_bootstrap`` row subsamples (without replacement, size ``bootstrap_frac * n_rows``
        each) and a column is flagged only when its ENTIRE ``bootstrap_percentile_band`` percentile range
        of per-resample AUCs stays within ``tolerance`` of 0.5. This guards against a single unlucky
        sample making a genuinely weak-but-real feature (whose true AUC is just outside ``tolerance``)
        look like pure noise by chance -- the single-pass check has no way to distinguish "AUC is near
        0.5 because there's no signal" from "AUC is near 0.5 because this particular sample happened to
        land there".
    bootstrap_frac
        Fraction of rows sampled per resample (only used when ``n_bootstrap`` is set).
    bootstrap_percentile_band
        ``(low, high)`` percentiles (0-100) of the per-resample AUC distribution that must both fall
        within ``tolerance`` of 0.5 for a column to be flagged (only used when ``n_bootstrap`` is set).
    random_state
        Seed for the bootstrap row sampling (only used when ``n_bootstrap`` is set); ``None`` uses
        nondeterministic sampling.

    Returns
    -------
    list of str
        Column names to consider dropping as near-noise before the full selection pipeline.
    """
    cols = list(columns) if columns is not None else list(df.select_dtypes(include=[np.number]).columns)
    y_arr = np.asarray(y)
    X = df[cols].to_numpy(dtype=np.float64)

    if n_bootstrap is None:
        aucs = batch_univariate_auc(X, y_arr)
        return [col for col, auc in zip(cols, aucs) if abs(auc - 0.5) <= tolerance]

    if n_bootstrap <= 0:
        raise ValueError(f"n_bootstrap must be a positive int when provided, got {n_bootstrap}")

    n_rows = X.shape[0]
    sample_size = max(2, round(n_rows * bootstrap_frac))
    rng = np.random.default_rng(random_state)

    resampled_aucs = np.empty((n_bootstrap, len(cols)), dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.choice(n_rows, size=sample_size, replace=False)
        resampled_aucs[i, :] = batch_univariate_auc(X[idx, :], y_arr[idx])

    low_pct, high_pct = bootstrap_percentile_band
    lo = np.percentile(resampled_aucs, low_pct, axis=0)
    hi = np.percentile(resampled_aucs, high_pct, axis=0)
    consistently_near_noise = (np.abs(lo - 0.5) <= tolerance) & (np.abs(hi - 0.5) <= tolerance)
    return [col for col, flagged in zip(cols, consistently_near_noise) if flagged]


__all__ = ["drop_near_noise_univariate_auc"]
