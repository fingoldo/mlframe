"""Resampling-stability filter for a feature's "deviation from own group baseline -> target sign" property.

A feature can look informative on MI/permutation-importance yet be "jumpy": its apparent relationship to the
target (via its deviation from a per-entity/group baseline) flips sign across different subsamples of
entities/groups, meaning the signal doesn't generalize and is likely a resampling artifact rather than a real
effect. This complements plain permutation-MI/MRMR (which score on the full sample once) with a check for
whether the monotonic separation actually holds up under resampling -- the diagnostic an 8th-place Ubiquant
writeup used to discard features that were "jumpy... across many different baskets" of entities.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def monotonic_deviation_stability_filter(
    df: pd.DataFrame,
    y: np.ndarray,
    group_col: str,
    feature_cols: Optional[Sequence[str]] = None,
    n_subsamples: int = 30,
    group_fraction: float = 0.5,
    min_stable_fraction: float = 0.7,
    random_state: int = 0,
) -> pd.DataFrame:
    """Score each feature's group-baseline-deviation sign-stability across random group subsamples.

    For each feature, computes ``deviation = feature - group_mean(feature)`` (deviation from the entity's own
    baseline), then repeatedly subsamples a fraction of the groups and measures the Spearman-sign correlation
    between ``deviation`` and ``y`` within that subsample. A feature is "stable" when its correlation sign
    agrees with the full-sample sign in at least ``min_stable_fraction`` of subsamples.

    Parameters
    ----------
    df
        Feature frame, one row per (entity/group, observation).
    y
        Target aligned to ``df``.
    group_col
        Entity/group column that ``deviation`` is computed relative to.
    feature_cols
        Columns to audit; defaults to all numeric columns other than ``group_col``.
    n_subsamples
        Number of random group subsamples to draw.
    group_fraction
        Fraction of distinct groups included in each subsample.
    min_stable_fraction
        A feature's ``stable`` flag requires its sign-agreement rate across subsamples to reach this.
    random_state
        Controls the group subsampling draws.

    Returns
    -------
    pd.DataFrame
        One row per audited feature: ``feature``, ``full_sample_correlation``, ``sign_agreement_fraction``,
        ``stable`` (bool).
    """
    if feature_cols is None:
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != group_col]
    feature_cols = list(feature_cols)

    y = np.asarray(y, dtype=np.float64)
    group_means = df.groupby(group_col, sort=False)[feature_cols].transform("mean")
    deviations = df[feature_cols] - group_means

    groups = df[group_col].to_numpy()
    unique_groups = np.unique(groups)
    n_groups_per_draw = max(1, round(group_fraction * len(unique_groups)))
    rng = np.random.default_rng(random_state)

    full_sample_corr = {col: _spearman_corr(deviations[col].to_numpy(), y) for col in feature_cols}

    agreement_counts = {col: 0 for col in feature_cols}
    for _ in range(n_subsamples):
        chosen_groups = rng.choice(unique_groups, size=n_groups_per_draw, replace=False)
        row_mask = np.isin(groups, chosen_groups)
        y_sub = y[row_mask]
        for col in feature_cols:
            dev_sub = deviations[col].to_numpy()[row_mask]
            corr = _spearman_corr(dev_sub, y_sub)
            full_sign = np.sign(full_sample_corr[col])
            sub_sign = np.sign(corr)
            if full_sign == 0 or sub_sign == full_sign:
                agreement_counts[col] += 1

    rows = []
    for col in feature_cols:
        agreement_fraction = agreement_counts[col] / n_subsamples
        rows.append(
            {
                "feature": col,
                "full_sample_correlation": full_sample_corr[col],
                "sign_agreement_fraction": agreement_fraction,
                "stable": agreement_fraction >= min_stable_fraction,
            }
        )
    return pd.DataFrame(rows)


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    from scipy.stats import spearmanr

    corr, _p = spearmanr(x, y)
    return float(corr) if np.isfinite(corr) else 0.0


__all__ = ["monotonic_deviation_stability_filter"]
