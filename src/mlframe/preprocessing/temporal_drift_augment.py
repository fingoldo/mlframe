"""Temporal-drift training augmentation: "duplicate + drop last period(s)" trick.

For panel/longitudinal data (one or more rows per entity over time), a model trained only on each entity's
TRUE last-observed period never sees what an earlier, less-complete history looks like -- yet in production
the same model is queried at arbitrary points in an entity's lifecycle. This augmentation manufactures extra
training rows by dropping each entity's most recent period(s), re-standardizing period-relative features
against the truncated history only (so the synthetic row's normalization matches what that entity's stats
actually looked like at that earlier vintage), and keeping the entity's real label -- doubling (or more)
effective training rows while explicitly teaching the model to be robust to being queried mid-history.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def augment_temporal_drift(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    feature_cols: Optional[Sequence[str]] = None,
    n_drop_options: Sequence[int] = (1,),
    min_history: int = 2,
) -> pd.DataFrame:
    """Return ``df`` concatenated with synthetic "earlier vintage" rows.

    Parameters
    ----------
    df
        One row per (entity, time period); a label/other columns may be present and are carried through
        unchanged on augmented rows (the real label at that entity's TRUE last statement, per the source
        technique -- only the features are re-standardized to the truncated vintage).
    entity_col, time_col
        Grouping and ordering columns.
    feature_cols
        Columns to re-standardize (z-score) against each truncated history; defaults to all numeric columns
        other than ``entity_col``/``time_col``.
    n_drop_options
        How many trailing periods to drop per synthetic row, e.g. ``(1, 2)`` produces two augmented copies
        per eligible entity (drop-last-1 and drop-last-2).
    min_history
        An entity needs strictly more than ``n_drop`` rows AND at least this many rows remaining after the
        drop to be eligible (a single-row truncated history has no variance to standardize against).

    Returns
    -------
    pd.DataFrame
        ``df`` with augmented rows appended (index reset), plus a bool ``_temporal_drift_augmented`` column
        (``False`` on original rows) so callers can filter/weight them separately if desired.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in (entity_col, time_col)]
    feature_cols = list(feature_cols)

    ordered = df.sort_values([entity_col, time_col], kind="mergesort").reset_index(drop=True)
    entity_groups = ordered.groupby(entity_col, sort=False)

    # groupby().expanding().mean()/.std() profiled at ~13s for 100k entities x 6 rows (dominated by pandas'
    # per-window-bounds machinery, not the arithmetic) -- same "generic window/agg machinery on tiny per-
    # entity groups" cost class as the per_group_apply anti-pattern found elsewhere. groupby().cumsum() uses
    # a flat Cython segment-restart scan instead and computes the identical expanding mean/std algebraically.
    rank_within_entity = entity_groups.cumcount()
    n = (rank_within_entity + 1).astype(np.float64)
    cumsum_x = ordered.groupby(entity_col, sort=False)[feature_cols].cumsum()
    cumsum_x2 = (ordered[feature_cols] ** 2).groupby(ordered[entity_col], sort=False).cumsum()
    expanding_mean = cumsum_x.div(n, axis=0)
    # sample variance (ddof=1) of an expanding window from its running sum/sum-of-squares; undefined (NaN) at n=1.
    var = cumsum_x2.sub(cumsum_x.pow(2).div(n, axis=0)).div((n - 1).where(n > 1, np.nan), axis=0)
    expanding_std = var.clip(lower=0).pow(0.5)

    count_within_entity = entity_groups[entity_col].transform("size")

    augmented_frames = [ordered.assign(_temporal_drift_augmented=False)]
    for n_drop in n_drop_options:
        if n_drop < 1:
            raise ValueError(f"augment_temporal_drift: n_drop_options must be >= 1; got {n_drop}")
        new_last_rank = count_within_entity - n_drop - 1
        eligible = (new_last_rank >= min_history - 1) & (rank_within_entity == new_last_rank)
        if not eligible.any():
            continue
        synth = ordered.loc[eligible].copy()
        std = expanding_std.loc[eligible]
        mean = expanding_mean.loc[eligible]
        raw = ordered.loc[eligible, feature_cols]
        safe_std = std.where(std > 0, np.nan)
        standardized = (raw - mean) / safe_std
        synth[feature_cols] = standardized.where(safe_std.notna(), 0.0)
        synth["_temporal_drift_augmented"] = True
        augmented_frames.append(synth)

    return pd.concat(augmented_frames, axis=0, ignore_index=True)


__all__ = ["augment_temporal_drift"]
