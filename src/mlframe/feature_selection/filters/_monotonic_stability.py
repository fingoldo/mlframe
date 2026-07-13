"""Resampling-stability filter for a feature's "deviation from own group baseline -> target sign" property.

A feature can look informative on MI/permutation-importance yet be "jumpy": its apparent relationship to the
target (via its deviation from a per-entity/group baseline) flips sign across different subsamples of
entities/groups, meaning the signal doesn't generalize and is likely a resampling artifact rather than a real
effect. This complements plain permutation-MI/MRMR (which score on the full sample once) with a check for
whether the monotonic separation actually holds up under resampling -- the diagnostic an 8th-place Ubiquant
writeup used to discard features that were "jumpy... across many different baskets" of entities.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

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
    segment_col: Optional[str] = None,
    segment_min_agreement: Optional[float] = None,
) -> pd.DataFrame:
    """Score each feature's group-baseline-deviation sign-stability across random group subsamples.

    For each feature, computes ``deviation = feature - group_mean(feature)`` (deviation from the entity's own
    baseline), then repeatedly subsamples a fraction of the groups and measures the Spearman-sign correlation
    between ``deviation`` and ``y`` within that subsample. A feature is "stable" when its correlation sign
    agrees with the full-sample sign in at least ``min_stable_fraction`` of subsamples.

    Random group subsampling mixes groups from every segment of the data in each draw, so a feature whose
    deviation-target relationship flips sign inside one specific segment (e.g. one instrument/basket/regime)
    but holds in the rest gets averaged away: most draws still contain a majority of "good" groups and keep
    the majority sign, so the feature reads as globally stable even though it is genuinely jumpy within that
    segment. Passing ``segment_col`` adds an opt-in, deterministic per-segment check (one pass per distinct
    segment value, no resampling) that surfaces exactly this pattern via the ``segment_*`` output columns,
    without touching the random-subsample columns or scores.

    Parameters
    ----------
    df
        Feature frame, one row per (entity/group, observation).
    y
        Target aligned to ``df``.
    group_col
        Entity/group column that ``deviation`` is computed relative to.
    feature_cols
        Columns to audit; defaults to all numeric columns other than ``group_col`` (and ``segment_col``).
    n_subsamples
        Number of random group subsamples to draw.
    group_fraction
        Fraction of distinct groups included in each subsample.
    min_stable_fraction
        A feature's ``stable`` flag requires its sign-agreement rate across subsamples to reach this.
    random_state
        Controls the group subsampling draws.
    segment_col
        Optional explicit segment/instrument/entity column to stratify the stability check by, in addition to
        (not instead of) the random group-subsample check above. ``None`` (default) leaves output and scoring
        bit-identical to the original random-subsample-only behavior.
    segment_min_agreement
        Threshold for ``segment_stable``; defaults to ``min_stable_fraction`` when ``segment_col`` is given.

    Returns
    -------
    pd.DataFrame
        One row per audited feature: ``feature``, ``full_sample_correlation``, ``sign_agreement_fraction``,
        ``stable`` (bool), plus ``segment_sign_agreement_fraction`` and ``segment_stable`` (bool) when
        ``segment_col`` is given.
    """
    if feature_cols is None:
        excluded = {group_col} if segment_col is None else {group_col, segment_col}
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]
    feature_cols = list(feature_cols)

    y = np.asarray(y, dtype=np.float64)
    group_means = df.groupby(group_col, sort=False)[feature_cols].transform("mean")
    # Materialise the (n, p) deviation matrix ONCE -- the loops below used to call ``.to_numpy()`` on a
    # single column per (subsample, feature)/(segment, feature) cell (30x500 = ~15,000 re-materialisations
    # by default); every consumer below slices ROWS out of this one array instead.
    dev_matrix = np.asarray(df[feature_cols].to_numpy(), dtype=np.float64) - np.asarray(group_means.to_numpy(), dtype=np.float64)

    groups = df[group_col].to_numpy()
    unique_groups = np.unique(groups)
    n_groups_per_draw = max(1, round(group_fraction * len(unique_groups)))
    rng = np.random.default_rng(random_state)

    full_sample_corr_arr = _spearman_corr_batch(dev_matrix, y)
    full_sample_corr = dict(zip(feature_cols, full_sample_corr_arr.tolist()))
    full_sign_arr = np.sign(full_sample_corr_arr)

    # BATCHED SPEARMAN (2026-07-13): the pre-fix code called ``scipy.stats.spearmanr`` ONCE PER (subsample,
    # feature) -- default 30 x 500 = ~15,000 individual scipy calls. Spearman correlation IS Pearson
    # correlation of the (average-tie) ranks by definition, so every feature sharing one subsample's row
    # mask can be rank-transformed and correlated in ONE batched pass instead of one scipy call per column
    # (see ``_spearman_corr_batch``). ``agreement_counts_arr`` replaces the per-column dict with a running
    # vector, summed once at the end.
    agreement_counts_arr = np.zeros(len(feature_cols), dtype=np.int64)
    for _ in range(n_subsamples):
        chosen_groups = rng.choice(unique_groups, size=n_groups_per_draw, replace=False)
        row_mask = np.isin(groups, chosen_groups)
        y_sub = y[row_mask]
        corr_arr = _spearman_corr_batch(dev_matrix[row_mask, :], y_sub)
        sub_sign_arr = np.sign(corr_arr)
        agreement_counts_arr += (full_sign_arr == 0) | (sub_sign_arr == full_sign_arr)
    agreement_counts = dict(zip(feature_cols, agreement_counts_arr.tolist()))

    segment_agreement_fraction: Dict[str, float] = {}
    segment_stable: Dict[str, bool] = {}
    if segment_col is not None:
        seg_min_agreement = min_stable_fraction if segment_min_agreement is None else segment_min_agreement
        segments = df[segment_col].to_numpy()
        unique_segments = np.unique(segments)
        segment_agreement_counts_arr = np.zeros(len(feature_cols), dtype=np.int64)
        n_valid_segments = 0
        for seg in unique_segments:
            seg_mask = segments == seg
            if seg_mask.sum() < 2:
                continue
            n_valid_segments += 1
            y_seg = y[seg_mask]
            corr_arr = _spearman_corr_batch(dev_matrix[seg_mask, :], y_seg)
            seg_sign_arr = np.sign(corr_arr)
            segment_agreement_counts_arr += (full_sign_arr == 0) | (seg_sign_arr == full_sign_arr)
        segment_agreement_counts = dict(zip(feature_cols, segment_agreement_counts_arr.tolist()))
        for col in feature_cols:
            frac = segment_agreement_counts[col] / n_valid_segments if n_valid_segments > 0 else np.nan
            segment_agreement_fraction[col] = frac
            segment_stable[col] = bool(n_valid_segments > 0 and frac >= seg_min_agreement)

    rows = []
    for col in feature_cols:
        agreement_fraction = agreement_counts[col] / n_subsamples
        row: Dict[str, Any] = {
            "feature": col,
            "full_sample_correlation": full_sample_corr[col],
            "sign_agreement_fraction": agreement_fraction,
            "stable": agreement_fraction >= min_stable_fraction,
        }
        if segment_col is not None:
            row["segment_sign_agreement_fraction"] = segment_agreement_fraction[col]
            row["segment_stable"] = segment_stable[col]
        rows.append(row)
    return pd.DataFrame(rows)


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    from scipy.stats import spearmanr

    corr, _p = spearmanr(x, y)
    return float(corr) if np.isfinite(corr) else 0.0


def _spearman_corr_batch(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Batched twin of ``_spearman_corr``: signed Spearman correlation of every column of ``x`` (n, p)
    against the shared ``y`` (n,), in ONE rank-transform + ONE vectorized Pearson-on-ranks pass instead of
    one ``scipy.stats.spearmanr`` call per column.

    Spearman correlation IS Pearson correlation of the (average-tie) ranks by definition -- not an
    approximation -- and ``scipy.stats.rankdata(x, axis=0)`` ranks each column of a 2D array
    INDEPENDENTLY (a NaN confined to one column only poisons that column's own rank vector, verified), the
    same average-tie method ``scipy.stats.spearmanr`` uses internally, so batching the rank transform
    across every column sharing one ``y`` reproduces ``_spearman_corr``'s per-column results to ~1e-12.

    Matches ``_spearman_corr``'s degenerate-input contract per column: 0.0 when ``n < 2``, ``y`` is
    NaN-containing or constant (checked once, since ``y`` is shared -- the sub-loop calling this once per
    subsample no longer re-derives that per feature), or the column itself is NaN-containing or constant
    (``scipy.stats.spearmanr`` propagates any NaN in either side to the WHOLE correlation, matching the
    isfinite-mask precheck here rather than a masked/partial-overlap correlation).
    """
    x = np.asarray(x, dtype=np.float64)
    n, p = x.shape
    out = np.zeros(p, dtype=np.float64)
    if n < 2:
        return out
    y = np.asarray(y, dtype=np.float64)
    if y.shape[0] != n or not np.isfinite(y).all() or np.std(y) == 0.0:
        return out
    col_finite = np.isfinite(x).all(axis=0)
    col_std = np.std(x, axis=0)
    valid = col_finite & (col_std != 0.0)
    if not valid.any():
        return out
    from scipy.stats import rankdata

    ranks = rankdata(x[:, valid], axis=0)
    y_rank = rankdata(y)
    y_rank_c = y_rank - y_rank.mean()
    y_ss = float(np.dot(y_rank_c, y_rank_c))
    ranks_c = ranks - ranks.mean(axis=0, keepdims=True)
    num = ranks_c.T @ y_rank_c
    den = np.sqrt(np.sum(ranks_c * ranks_c, axis=0) * y_ss)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(den > 0.0, num / den, 0.0)
    out[valid] = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return out


__all__ = ["monotonic_deviation_stability_filter"]
