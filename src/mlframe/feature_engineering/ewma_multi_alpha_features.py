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
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from numba import njit


@njit(fastmath=False, cache=True)
def _grouped_ewma_njit(values_sorted: np.ndarray, starts: np.ndarray, ends: np.ndarray, alpha: float) -> np.ndarray:
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


def ewma_multi_alpha_features(values: np.ndarray, group_ids: np.ndarray, alphas: Sequence[float] = (0.5, 0.1)) -> Dict[str, np.ndarray]:
    """Per-entity EWMA of ``values`` at each alpha in ``alphas``, reset at every group boundary.

    Parameters
    ----------
    values
        ``(n,)`` value column (e.g. a binary product-presence indicator), in the row order reflecting each
        entity's true chronological sequence.
    group_ids
        ``(n,)`` entity/group key aligned to ``values``.
    alphas
        Smoothing factors to compute, each in ``(0, 1]``. Larger alpha weights recent observations more
        heavily (short memory); smaller alpha gives a longer-lasting, slower-decaying effect.

    Returns
    -------
    dict[str, np.ndarray]
        ``{"ewma_alpha_{a}": array}`` for each requested alpha, one column per alpha, same row order as the
        input (not the internal group-sorted order).
    """
    from mlframe.feature_engineering.grouped import iter_group_segments

    values_arr = np.ascontiguousarray(values, dtype=np.float64)
    sort_idx, starts, ends = iter_group_segments(group_ids)
    values_sorted = values_arr[sort_idx]
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

    return result


__all__ = ["ewma_multi_alpha_features"]
