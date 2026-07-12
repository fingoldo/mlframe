"""``latent_parameter_recovery_features``: reverse-engineer a hidden generative parameter from a closed-form
relation among observed columns, via candidate enumeration + summary statistics.

Source: 5th_home-credit-default-risk.md -- recovering an interest rate from ``Annuity, Amount, CNT_PAYMENT``
via the compound-interest formula: enumerate a discrete candidate set (multiples of 6 months for the unknown
duration), compute summary stats (min/max/mean/median) of the values consistent with the observed columns,
then (OOF, against a partially-labeled historical subset) train a supervised model to disambiguate. The
candidate-generation/constraint-check is inherently domain-specific (a different closed-form relation per
use case), but the ENUMERATE-THEN-SUMMARIZE machinery is fully generic: given any per-row constraint
predicate over a candidate grid, this produces the summary-stat features directly usable by any downstream
model (including mlframe's existing OOF/CV training machinery -- the "disambiguation" half needs no new code,
just training on these features against whatever partially-labeled ground truth exists).
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd


def latent_parameter_recovery_features(
    df: pd.DataFrame,
    candidate_grid: Sequence[float] | np.ndarray,
    constraint_fn: Callable[[pd.DataFrame, float], np.ndarray],
    tolerance: float = 0.05,
    column_prefix: str = "latent_param",
    weight_fn: Callable[[pd.DataFrame, float], np.ndarray] | None = None,
) -> pd.DataFrame:
    """Per-row summary statistics of candidate values consistent with a closed-form relation.

    Parameters
    ----------
    df
        Frame with the OBSERVED columns the closed-form relation depends on.
    candidate_grid
        Discrete hypothesis values for the hidden parameter to test (e.g. multiples of 6 for a duration in
        months).
    constraint_fn
        ``constraint_fn(df, candidate_value) -> (n,) float array`` -- returns, for the given candidate value,
        a per-row RESIDUAL of the closed-form relation (0 = the candidate exactly reproduces the observed
        columns under the relation; nonzero = inconsistent). E.g. for a compound-interest annuity relation,
        ``constraint_fn`` computes the annuity implied by ``candidate_value`` months and returns
        ``implied_annuity - observed_annuity``.
    tolerance
        A candidate is "consistent" with a row when ``abs(constraint_fn(df, candidate)) <= tolerance``.
    column_prefix
        Output column-name prefix.
    weight_fn
        Optional ``weight_fn(df, candidate_value) -> (n,) float array`` of nonnegative PRIOR PLAUSIBILITY
        weights for that candidate (e.g. a prior density over the hidden generative parameter). When given,
        every constraint-satisfying candidate is no longer treated as equally likely: ``mean``/``median`` are
        weighted by ``weight_fn``, and candidates with weight <= 0 are excluded from all summary stats
        (a rejection, not just a down-weighting). ``None`` (default) reproduces the original uniform-weight
        behavior bit-for-bit -- this is a strictly opt-in extension.

    Returns
    -------
    pd.DataFrame
        ``{prefix}_min``, ``{prefix}_max``, ``{prefix}_mean``, ``{prefix}_median``, ``{prefix}_n_candidates``
        -- summary of the candidate grid values consistent with each row (NaN/0 when no candidate is
        consistent for that row). With ``weight_fn``, ``mean``/``median`` are prior-weighted and all stats
        additionally exclude zero-weight candidates.
    """
    n = len(df)
    consistency = np.full((n, len(candidate_grid)), False, dtype=bool)
    grid_arr = np.asarray(candidate_grid, dtype=np.float64)

    for col_idx, candidate in enumerate(grid_arr):
        residual = np.asarray(constraint_fn(df, float(candidate)), dtype=np.float64)
        consistency[:, col_idx] = np.abs(residual) <= tolerance

    if weight_fn is None:
        return _uniform_summary(df, grid_arr, consistency, column_prefix)
    return _weighted_summary(df, grid_arr, consistency, weight_fn, column_prefix)


def _uniform_summary(df: pd.DataFrame, grid_arr: np.ndarray, consistency: np.ndarray, column_prefix: str) -> pd.DataFrame:
    n = len(df)
    n_candidates = consistency.sum(axis=1)
    masked_grid = np.where(consistency, grid_arr[np.newaxis, :], np.nan)

    # rows with zero consistent candidates are an expected, handled case (all-NaN -> NaN summary), not an
    # error -- suppress numpy's all-NaN-slice RuntimeWarning rather than let it leak to the caller's console.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_min = np.nanmin(masked_grid, axis=1)
        col_max = np.nanmax(masked_grid, axis=1)
        col_mean = np.nanmean(masked_grid, axis=1)

    # np.nanmedian routes through numpy.ma (masked-array) internals -- measured as the dominant cost (22.5s
    # of 77s cProfile total at n=200000/grid=500). A plain np.sort already pushes NaN to the end of each row
    # (numpy's documented ascending-sort NaN convention), so the median of the CONSISTENT values is just the
    # middle element(s) of that sorted row, addressable by n_candidates alone -- no masked-array machinery.
    sorted_grid = np.sort(masked_grid, axis=1)
    row_idx = np.arange(n)
    mid_low = np.clip((n_candidates - 1) // 2, 0, None)
    mid_high = np.clip(n_candidates // 2, 0, None)
    col_median = (sorted_grid[row_idx, mid_low] + sorted_grid[row_idx, mid_high]) / 2.0
    col_median = np.where(n_candidates > 0, col_median, np.nan)

    out: Dict[str, np.ndarray] = {
        f"{column_prefix}_min": col_min,
        f"{column_prefix}_max": col_max,
        f"{column_prefix}_mean": col_mean,
        f"{column_prefix}_median": col_median,
        f"{column_prefix}_n_candidates": n_candidates.astype(np.float64),
    }
    return pd.DataFrame(out, index=df.index)


def _weighted_summary(
    df: pd.DataFrame,
    grid_arr: np.ndarray,
    consistency: np.ndarray,
    weight_fn: Callable[[pd.DataFrame, float], np.ndarray],
    column_prefix: str,
) -> pd.DataFrame:
    n = len(df)
    weights = np.zeros((n, len(grid_arr)), dtype=np.float64)
    for col_idx, candidate in enumerate(grid_arr):
        weights[:, col_idx] = np.asarray(weight_fn(df, float(candidate)), dtype=np.float64)

    # a zero (or negative) prior weight is a REJECTION of that candidate, not merely a down-weighting --
    # it drops out of min/max/n_candidates too, matching the caller's stated implausibility judgment.
    eff_mask = consistency & (weights > 0)
    n_candidates = eff_mask.sum(axis=1)
    masked_grid = np.where(eff_mask, grid_arr[np.newaxis, :], np.nan)
    eff_w = np.where(eff_mask, weights, 0.0)
    total_w = eff_w.sum(axis=1)

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_min = np.nanmin(masked_grid, axis=1)
        col_max = np.nanmax(masked_grid, axis=1)
        col_mean = np.where(total_w > 0, (eff_w * np.where(eff_mask, grid_arr[np.newaxis, :], 0.0)).sum(axis=1) / np.where(total_w > 0, total_w, 1.0), np.nan)

    # weighted median: sort each row by candidate value, then take the first candidate whose cumulative
    # prior weight reaches half the row's total weight (the standard weighted-median definition).
    sort_order = np.argsort(grid_arr)
    grid_sorted = grid_arr[sort_order]
    w_sorted = eff_w[:, sort_order]
    cum_w = np.cumsum(w_sorted, axis=1)
    half_w = total_w / 2.0
    reaches_half = cum_w >= half_w[:, np.newaxis]
    # rows with total_w == 0 have no True entry; argmax on an all-False row returns 0, masked out below.
    median_col_idx = np.argmax(reaches_half, axis=1)
    col_median = np.where(total_w > 0, grid_sorted[median_col_idx], np.nan)

    out: Dict[str, np.ndarray] = {
        f"{column_prefix}_min": col_min,
        f"{column_prefix}_max": col_max,
        f"{column_prefix}_mean": col_mean,
        f"{column_prefix}_median": col_median,
        f"{column_prefix}_n_candidates": n_candidates.astype(np.float64),
    }
    return pd.DataFrame(out, index=df.index)


__all__ = ["latent_parameter_recovery_features"]
