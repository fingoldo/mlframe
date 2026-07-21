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
    weight_by_recency: bool = False,
    min_augmented_weight: float = 0.1,
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
    weight_by_recency
        Opt-in. When ``True``, adds a ``_sample_weight`` column: ``1.0`` on true (non-augmented) rows, and
        ``max(min_augmented_weight, retained_periods / full_periods)`` on augmented rows. Naive 50/50
        duplication treats a synthetic row truncated to 2-of-8 periods the same as one truncated to 5-of-6 --
        when entity history lengths vary a lot, the heavier-truncation copies are a much less faithful proxy
        for the true inference-time distribution and can dilute the signal from genuinely full-history rows
        if left equal-weighted. Default ``False`` leaves output (including columns) bit-identical to before.
    min_augmented_weight
        Floor applied to the recency-based weight so a very-heavily-truncated synthetic row still contributes
        some signal rather than being effectively dropped. Only used when ``weight_by_recency=True``.

    Returns
    -------
    pd.DataFrame
        ``df`` with augmented rows appended (index reset), plus a bool ``_temporal_drift_augmented`` column
        (``False`` on original rows) so callers can filter/weight them separately if desired. When
        ``weight_by_recency=True``, also includes a float ``_sample_weight`` column.

    Usage
    -----
    Real (``_temporal_drift_augmented=False``) rows are the UNMODIFIED input panel, every period at its raw
    scale -- they are deliberately left untouched (``result.loc[~result["_temporal_drift_augmented"]]``
    always equals ``df`` exactly) so a caller can pick whatever real-row representation they need. Synthetic
    rows, however, have ``feature_cols`` OVERWRITTEN with a truncated-history z-score in those SAME columns.
    Concatenating the full returned frame directly therefore mixes raw-scale and z-score-scale values in one
    column. To build a single, internally
    scale-consistent training frame, select and standardize the real rows yourself before combining, e.g.
    via :func:`select_true_last_standardized` (the true-last-period-per-entity, full-history-z-scored
    counterpart to this function's truncated-history synthetic rows)::

        real = select_true_last_standardized(df, entity_col, time_col, feature_cols)
        augmented = augment_temporal_drift(df, entity_col, time_col, feature_cols)
        synthetic_only = augmented.loc[augmented["_temporal_drift_augmented"]]
        train = pd.concat([real, synthetic_only], axis=0, ignore_index=True)
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

    first_frame = ordered.assign(_temporal_drift_augmented=False)
    if weight_by_recency:
        first_frame["_sample_weight"] = 1.0
    augmented_frames = [first_frame]
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
        if weight_by_recency:
            retained = (new_last_rank + 1).loc[eligible].astype(np.float64)
            full = count_within_entity.loc[eligible].astype(np.float64)
            synth["_sample_weight"] = (retained / full).clip(lower=min_augmented_weight)
        augmented_frames.append(synth)

    return pd.concat(augmented_frames, axis=0, ignore_index=True)


def select_true_last_standardized(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    feature_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Collapse a (entity, time) panel to one row per entity -- the TRUE last-observed period,
    ``feature_cols`` standardized (z-score, ``ddof=1``) against that entity's OWN FULL history.

    This is the "real-row" half of the usage pattern :func:`augment_temporal_drift` intentionally leaves to
    the caller (see its docstring's Usage section): concatenate this function's output with
    ``augment_temporal_drift(...).loc[lambda d: d["_temporal_drift_augmented"]]`` to get a single frame
    where every row -- true-last (full-history z-score) and synthetic (truncated-history z-score) alike --
    is standardized the same way, avoiding the raw-vs-z-score scale mismatch. Vectorized equivalent of the per-entity Python loop
    ``tests/preprocessing/test_biz_val_temporal_drift_augment.py`` hand-rolls for this exact purpose.

    Parameters
    ----------
    df
        One row per (entity, time period).
    entity_col, time_col
        Grouping and ordering columns.
    feature_cols
        Columns to standardize; defaults to all numeric columns other than ``entity_col``/``time_col``.

    Returns
    -------
    pd.DataFrame
        One row per entity (the true last period), with ``feature_cols`` z-scored against that entity's own
        full history (``0.0`` where the entity has a single period or zero variance, i.e. an undefined
        ``ddof=1`` std -- same fallback convention as :func:`augment_temporal_drift`'s synthetic rows). All
        other columns keep that row's original values.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in (entity_col, time_col)]
    feature_cols = list(feature_cols)

    ordered = df.sort_values([entity_col, time_col], kind="mergesort").reset_index(drop=True)
    entity_groups = ordered.groupby(entity_col, sort=False)

    full_mean = entity_groups[feature_cols].transform("mean")
    full_std = entity_groups[feature_cols].transform("std")
    safe_std = full_std.where(full_std > 0, np.nan)
    standardized = (ordered[feature_cols] - full_mean) / safe_std

    last_mask = entity_groups.cumcount() == (entity_groups[entity_col].transform("size") - 1)
    out = ordered.loc[last_mask].copy()
    out[feature_cols] = standardized.loc[last_mask].where(safe_std.loc[last_mask].notna(), 0.0)
    return out.reset_index(drop=True)


__all__ = ["augment_temporal_drift", "select_true_last_standardized"]
