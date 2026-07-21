"""``ewma_multi_alpha_features``: per-entity grouped EWMA at multiple alphas, as plain raw features.

Source: 4th_santander-product-recommendation.md -- "Exponential weighted average of each product's presence
per client as time goes. I've used two different alphas - 0.5 and 0.1... I wanted features that could hold
some temporal meaning but that would at the same time portray long lasting effect... least susceptible to the
amount of given data points." Distinct from mlframe's existing `ewma_residual` composite-target transform
(paired with a residual computation for a SINGLE alpha, k-parameterized) and from `mlframe.core.ewma.ewma`
(a plain flat-array function with no per-entity grouping): this emits one raw EWMA feature column PER alpha,
reset at each entity's group boundary -- directly usable as a feature block on its own, not tied to any
composite-target machinery. EWMA needs no fixed window (unlike simple rolling means), so it degrades
gracefully for short per-entity histories, which is exactly why the source picked it over rolling windows.

Optional extension: the source's fixed alphas (0.5/0.1) implicitly assume every entity decays at a similar
rate. Real entities differ in volatility (a bursty client vs a stable one genuinely need different alpha to
be tracked well), so ``adaptive_alpha_grid`` (opt-in, default off -- fixed-alpha output is bit-identical when
omitted) walk-forward-validates a small alpha grid PER ENTITY on that entity's own history and emits one
extra ``ewma_adaptive`` column built from each entity's own best-fit alpha.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from numba import njit


@njit(fastmath=False, cache=True)
def _grouped_ewma_njit(values_sorted: np.ndarray, starts: np.ndarray, ends: np.ndarray, alpha: float) -> np.ndarray:
    """Compute a per-group exponentially weighted moving average with a fixed alpha over sorted values."""
    n = values_sorted.shape[0]
    out = np.empty(n, dtype=np.float64)
    one_minus = 1.0 - alpha
    for g in range(starts.shape[0]):
        s, e = starts[g], ends[g]
        if e <= s:
            continue
        out[s] = values_sorted[s]
        for i in range(s + 1, e):
            out[i] = alpha * values_sorted[i] + one_minus * out[i - 1]
    return out


@njit(fastmath=False, cache=True)
def _grouped_ewma_adaptive_njit(
    values_sorted: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    alphas_grid: np.ndarray,
    val_frac: float,
    min_val_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a per-group EWMA using, per entity, the alpha from alphas_grid that best walk-forward-validates."""
    n = values_sorted.shape[0]
    out = np.empty(n, dtype=np.float64)
    chosen_alpha = np.empty(n, dtype=np.float64)
    n_alphas = alphas_grid.shape[0]
    for g in range(starts.shape[0]):
        s, e = starts[g], ends[g]
        length = e - s
        if length <= 0:
            continue
        best_alpha = alphas_grid[0]
        if length >= min_val_points + 2:
            # walk-forward validation: for each alpha, fit online over the whole entity history but
            # only score one-step-ahead prediction error (yesterday's smoothed value predicting today)
            # on the held-out tail -- the training prefix never sees its own future.
            val_len = min_val_points if int(length * val_frac) < min_val_points else int(length * val_frac)
            if val_len > length - 2:
                val_len = length - 2
            train_end = length - val_len
            best_sse = np.inf
            for a_idx in range(n_alphas):
                alpha = alphas_grid[a_idx]
                one_minus = 1.0 - alpha
                prev = values_sorted[s]
                sse = 0.0
                for i in range(1, length):
                    pred = prev
                    cur = alpha * values_sorted[s + i] + one_minus * prev
                    if i >= train_end:
                        err = values_sorted[s + i] - pred
                        sse += err * err
                    prev = cur
                if sse < best_sse:
                    best_sse = sse
                    best_alpha = alpha
        one_minus_best = 1.0 - best_alpha
        out[s] = values_sorted[s]
        chosen_alpha[s] = best_alpha
        for i in range(s + 1, e):
            out[i] = best_alpha * values_sorted[i] + one_minus_best * out[i - 1]
            chosen_alpha[i] = best_alpha
    return out, chosen_alpha


def ewma_multi_alpha_features(
    values: np.ndarray,
    group_ids: np.ndarray,
    alphas: Sequence[float] = (0.5, 0.1),
    *,
    adaptive_alpha_grid: Optional[Sequence[float]] = None,
    adaptive_val_frac: float = 0.3,
    adaptive_min_val_points: int = 3,
) -> Dict[str, np.ndarray]:
    """Per-entity EWMA of ``values`` at each alpha in ``alphas``, reset at every group boundary.

    Parameters
    ----------
    values
        ``(n,)`` value column (e.g. a binary product-presence indicator), in the row order reflecting each
        entity's true chronological sequence. NaN/inf are zero-filled before the recurrence (matching
        ``training.targets``/``stationarity.py``'s ``ewma_residual`` convention elsewhere in this codebase)
        -- without this, a single missing observation anywhere in an entity's history would permanently
        NaN-poison every subsequent EWMA value for that entity (``alpha*NaN + (1-alpha)*x`` is NaN, and stays
        NaN forever once introduced).
    group_ids
        ``(n,)`` entity/group key aligned to ``values``.
    alphas
        Smoothing factors to compute, each in ``(0, 1]``. Larger alpha weights recent observations more
        heavily (short memory); smaller alpha gives a longer-lasting, slower-decaying effect.
    adaptive_alpha_grid
        Opt-in. When given (non-empty), each entity independently walk-forward-validates every alpha in
        this grid on its OWN history (fit online, score one-step-ahead error on a held-out tail) and picks
        its own best-fitting alpha, instead of every entity sharing the same fixed alphas from ``alphas``.
        Useful when entities have genuinely different volatility/decay characteristics that no single
        shared alpha (or small fixed set) fits well. Adds two extra columns to the result, on top of the
        usual per-fixed-alpha ones; leaving this ``None`` (the default) reproduces the pre-existing
        fixed-alpha-only output bit-for-bit.
    adaptive_val_frac
        Fraction of each entity's history (tail) held out for walk-forward alpha scoring. Only used when
        ``adaptive_alpha_grid`` is given.
    adaptive_min_val_points
        Minimum number of held-out points required to run validation for an entity; entities shorter than
        ``adaptive_min_val_points + 2`` fall back to the grid's first alpha (too little history to validate
        reliably). Only used when ``adaptive_alpha_grid`` is given.

    Returns
    -------
    dict[str, np.ndarray]
        ``{"ewma_alpha_{a}": array}`` for each requested alpha, one column per alpha, same row order as the
        input (not the internal group-sorted order). When ``adaptive_alpha_grid`` is given, also includes
        ``"ewma_adaptive"`` (each entity's EWMA at its own validated-best alpha) and
        ``"ewma_adaptive_alpha"`` (the chosen alpha value, constant within an entity, for interpretability).
    """
    from mlframe.feature_engineering.grouped import iter_group_segments

    values_arr = np.ascontiguousarray(values, dtype=np.float64)
    sort_idx, starts, ends = iter_group_segments(group_ids)
    values_sorted = values_arr[sort_idx]
    values_sorted = np.where(np.isfinite(values_sorted), values_sorted, 0.0)
    starts64 = starts.astype(np.int64)
    ends64 = ends.astype(np.int64)

    n = values_arr.shape[0]
    result: Dict[str, np.ndarray] = {}
    for alpha in alphas:
        alpha_f = float(alpha)
        if not (0.0 < alpha_f <= 1.0):
            raise ValueError(f"ewma_multi_alpha_features: alpha must be in (0, 1]; got {alpha_f!r}.")
        ewma_sorted = _grouped_ewma_njit(values_sorted, starts64, ends64, alpha_f)
        out = np.empty(n, dtype=np.float64)
        out[sort_idx] = ewma_sorted
        result[f"ewma_alpha_{alpha_f}"] = out

    if adaptive_alpha_grid is not None:
        if len(adaptive_alpha_grid) == 0:
            raise ValueError("ewma_multi_alpha_features: adaptive_alpha_grid must be non-empty when given.")
        grid_arr = np.asarray([float(a) for a in adaptive_alpha_grid], dtype=np.float64)
        for alpha_f in grid_arr:
            if not (0.0 < alpha_f <= 1.0):
                raise ValueError(f"ewma_multi_alpha_features: adaptive_alpha_grid alpha must be in (0, 1]; got {alpha_f!r}.")
        if not (0.0 < adaptive_val_frac < 1.0):
            raise ValueError(f"ewma_multi_alpha_features: adaptive_val_frac must be in (0, 1); got {adaptive_val_frac!r}.")
        if adaptive_min_val_points < 1:
            raise ValueError(f"ewma_multi_alpha_features: adaptive_min_val_points must be >= 1; got {adaptive_min_val_points!r}.")

        adaptive_sorted, alpha_sorted = _grouped_ewma_adaptive_njit(
            values_sorted, starts64, ends64, grid_arr, float(adaptive_val_frac), int(adaptive_min_val_points)
        )
        adaptive_out = np.empty(n, dtype=np.float64)
        alpha_out = np.empty(n, dtype=np.float64)
        adaptive_out[sort_idx] = adaptive_sorted
        alpha_out[sort_idx] = alpha_sorted
        result["ewma_adaptive"] = adaptive_out
        result["ewma_adaptive_alpha"] = alpha_out

    return result


__all__ = ["ewma_multi_alpha_features"]
