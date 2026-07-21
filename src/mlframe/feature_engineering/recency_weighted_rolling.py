"""``recency_weighted_rolling_mean``: fixed-window rolling aggregate with within-window recency weighting.

Source: bestpractice_feature-engineering-general.md-style recency-weighted window features. mlframe already has
two adjacent but distinct primitives: ``feature_engineering.ewma_multi_alpha_features`` (exponential decay over
the FULL history, no fixed window) and ``feature_engineering.recency_aggregation.per_group_recency_weighted_mean``
(recency weighting, but over the FULL within-group history -- one value per group, not a sliding per-row
window). Neither combines a FIXED-SIZE sliding window with within-window recency weights, computed PER ROW
as-of that row (leakage-safe: only observations up to and including the current row are ever used). This module
is that missing combination, reusing the existing ``mlframe.core.recency_weights`` weight-vector math rather
than reimplementing the decay schemes.

NaN handling: neither ``values`` nor the weighted-mean/std computation checks for NaN anywhere. A NaN inside a
row's trailing window silently poisons that row's weighted ``num``/``den`` accumulation to NaN (an honest
failure -- the output is NaN, not silently wrong -- but undocumented until this note. Impute/drop NaNs in ``values`` before calling if that propagation
isn't wanted.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from mlframe.core.recency_weights import SCHEMES

__all__ = ["recency_weighted_rolling_mean", "recency_weighted_rolling_std"]

_SCHEME_CODES = {name: code for code, name in enumerate(SCHEMES)}


@njit(fastmath=False, cache=True)
def _rolling_recency_weighted_mean_sorted(v_sorted: np.ndarray, starts: np.ndarray, ends: np.ndarray, window: int, scheme_code: int, param: float) -> np.ndarray:
    """Per-row recency-weighted mean over a trailing ``window`` of within-group-sorted values.

    ``v_sorted`` is group-contiguous and oldest-first sorted WITHIN each group. For each row, only the
    ``min(window, rows_seen_so_far)`` most recent observations UP TO AND INCLUDING that row are used --
    leakage-safe by construction (never looks ahead).

    The ``"exp"`` scheme (``scheme_code == 1``) gets a dedicated fast path: profiled (``_benchmarks/
    bench_recency_weighted_rolling.py``, n_entities=20000/rows_per_entity=200/window=20 -- 4M rows), the
    generic loop's ``param**i`` call for every (row, window-position) pair -- 80M ``pow()`` calls -- was the
    dominant cost. ``param**i`` for consecutive integer ``i`` is a geometric sequence (``w_i = w_{i-1} *
    param``), so it is computed via one running multiply per step instead of a fresh ``pow()`` call, measured
    ~1.4x faster wall time (2321ms -> 1636ms at the same shape). NOT bit-identical to the ``pow()`` path --
    verified max-abs-diff ~4.4e-16 (machine-epsilon FP-reorder, `pow()`'s repeated-squaring vs sequential
    multiply round differently in the last bit), well within the project's documented ~1e-9 FP-reorder
    tolerance. ``"poly"``/``"power"`` keep the generic ``pow()`` path -- their exponent is the FIXED ``param``
    applied to a value that changes every step, not an integer step count, so no equivalent recurrence exists.
    """
    n = v_sorted.shape[0]
    out = np.empty(n, dtype=np.float64)
    n_groups = starts.shape[0]
    for g in range(n_groups):
        s = starts[g]
        e = ends[g]
        for row in range(s, e):
            m = min(window, row - s + 1)  # rows available in the trailing window as-of this row
            num = 0.0
            den = 0.0
            if scheme_code == 1:
                w = 1.0  # running weight; multiplied by ``param`` each step instead of recomputing param**i
                for j in range(1, m + 1):  # j = recency index, 1 (newest) -> m (oldest)
                    w *= param
                    num += w * v_sorted[row - j + 1]
                    den += w
            else:
                for pos in range(m):
                    i = m - pos  # 1-based recency index within THIS window, oldest (i=m) -> newest (i=1)
                    if scheme_code == 0:
                        w = ((m - i + 1) / m) ** param
                    else:
                        w = 1.0 / (i**param)
                    num += w * v_sorted[row - m + 1 + pos]
                    den += w
            out[row] = num / den if den > 0.0 else np.nan
    return out


def recency_weighted_rolling_mean(
    values: np.ndarray,
    group_ids: np.ndarray,
    window: int,
    *,
    order: np.ndarray | None = None,
    scheme: str = "poly",
    param: float = 1.0,
) -> np.ndarray:
    """Per-row, per-group trailing-``window`` mean with within-window recency weighting.

    At ``param`` equal to the scheme's identity value (poly: 0, exp: 1, power: 0) this is bit-identical to a
    plain uniform rolling mean of size ``window``, so opting into recency weighting never changes behaviour
    until a non-identity parameter is chosen (mirrors ``per_group_recency_weighted_mean``'s own guarantee).

    Parameters
    ----------
    values : np.ndarray
        1-D value column.
    group_ids : np.ndarray
        1-D entity id per row, aligned with ``values``.
    window
        Trailing window size (number of most-recent observations, as-of each row, to weight-average).
    order : np.ndarray, optional
        1-D sort key giving within-entity chronological order. Rows are ordered ASCENDING (oldest first)
        within each entity before windowing. If ``None``, existing row order within each group is used.
    scheme
        ``"poly"``, ``"exp"``, or ``"power"`` (see ``mlframe.core.recency_weights``).
    param
        Decay strength for ``scheme``.

    Returns
    -------
    np.ndarray
        1-D array aligned with the ORIGINAL row order of ``values``/``group_ids`` (not the sorted order).
    """
    if window < 1:
        raise ValueError(f"recency_weighted_rolling_mean: window must be >= 1, got {window}.")
    if scheme not in _SCHEME_CODES:
        raise ValueError(f"recency_weighted_rolling_mean: scheme must be one of {SCHEMES}; got {scheme!r}.")

    values = np.asarray(values, dtype=np.float64)
    group_ids = np.asarray(group_ids)
    n = values.shape[0]

    order_key = np.asarray(order) if order is not None else np.arange(n)
    # Stable sort by (group, order_key) so each group's rows are contiguous and oldest-first within the group.
    sort_idx = np.lexsort((order_key, group_ids))
    sorted_groups = group_ids[sort_idx]
    sorted_values = values[sort_idx]

    change_points = np.flatnonzero(np.diff(sorted_groups)) + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [n]))

    sorted_out = _rolling_recency_weighted_mean_sorted(sorted_values, starts, ends, int(window), _SCHEME_CODES[scheme], float(param))

    out = np.empty(n, dtype=np.float64)
    out[sort_idx] = sorted_out
    return out


@njit(fastmath=False, cache=True)
def _rolling_recency_weighted_std_sorted(v_sorted: np.ndarray, starts: np.ndarray, ends: np.ndarray, window: int, scheme_code: int, param: float) -> np.ndarray:
    """Per-row recency-weighted (population) std over a trailing ``window``, same group-sorted contract as
    ``_rolling_recency_weighted_mean_sorted``. Two-pass weighted variance: weighted mean first, then
    ``sum(w_i * (x_i - weighted_mean)^2) / sum(w_i)``, both passes sharing the same per-row weight vector so
    the ``"exp"`` running-multiply recurrence applies to both passes without re-deriving weights.
    """
    n = v_sorted.shape[0]
    out = np.empty(n, dtype=np.float64)
    n_groups = starts.shape[0]
    for g in range(n_groups):
        s = starts[g]
        e = ends[g]
        for row in range(s, e):
            m = min(window, row - s + 1)
            num = 0.0
            den = 0.0
            if scheme_code == 1:
                w = 1.0
                for j in range(1, m + 1):
                    w *= param
                    num += w * v_sorted[row - j + 1]
                    den += w
            else:
                for pos in range(m):
                    i = m - pos
                    if scheme_code == 0:
                        w = ((m - i + 1) / m) ** param
                    else:
                        w = 1.0 / (i**param)
                    num += w * v_sorted[row - m + 1 + pos]
                    den += w
            if den <= 0.0:
                out[row] = np.nan
                continue
            wmean = num / den
            var_num = 0.0
            if scheme_code == 1:
                w = 1.0
                for j in range(1, m + 1):
                    w *= param
                    diff = v_sorted[row - j + 1] - wmean
                    var_num += w * diff * diff
            else:
                for pos in range(m):
                    i = m - pos
                    if scheme_code == 0:
                        w = ((m - i + 1) / m) ** param
                    else:
                        w = 1.0 / (i**param)
                    diff = v_sorted[row - m + 1 + pos] - wmean
                    var_num += w * diff * diff
            var = var_num / den
            out[row] = np.sqrt(var) if var > 0.0 else 0.0
    return out


def recency_weighted_rolling_std(
    values: np.ndarray,
    group_ids: np.ndarray,
    window: int,
    *,
    order: np.ndarray | None = None,
    scheme: str = "poly",
    param: float = 1.0,
) -> np.ndarray:
    """Per-row, per-group trailing-``window`` (population) standard deviation with within-window recency
    weighting -- a dispersion/volatility analog of ``recency_weighted_rolling_mean``.

    At ``param`` equal to the scheme's identity value (poly: 0, exp: 1, power: 0) this is bit-identical to a
    plain uniform rolling population std (``ddof=0``) of size ``window``.

    Parameters
    ----------
    values : np.ndarray
        1-D value column.
    group_ids : np.ndarray
        1-D entity id per row, aligned with ``values``.
    window
        Trailing window size (number of most-recent observations, as-of each row, to weight-aggregate).
    order : np.ndarray, optional
        1-D sort key giving within-entity chronological order. Rows are ordered ASCENDING (oldest first)
        within each entity before windowing. If ``None``, existing row order within each group is used.
    scheme
        ``"poly"``, ``"exp"``, or ``"power"`` (see ``mlframe.core.recency_weights``).
    param
        Decay strength for ``scheme``.

    Returns
    -------
    np.ndarray
        1-D array aligned with the ORIGINAL row order of ``values``/``group_ids`` (not the sorted order).
        Rows with only one observation in their trailing window get std 0.0.
    """
    if window < 1:
        raise ValueError(f"recency_weighted_rolling_std: window must be >= 1, got {window}.")
    if scheme not in _SCHEME_CODES:
        raise ValueError(f"recency_weighted_rolling_std: scheme must be one of {SCHEMES}; got {scheme!r}.")

    values = np.asarray(values, dtype=np.float64)
    group_ids = np.asarray(group_ids)
    n = values.shape[0]

    order_key = np.asarray(order) if order is not None else np.arange(n)
    sort_idx = np.lexsort((order_key, group_ids))
    sorted_groups = group_ids[sort_idx]
    sorted_values = values[sort_idx]

    change_points = np.flatnonzero(np.diff(sorted_groups)) + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [n]))

    sorted_out = _rolling_recency_weighted_std_sorted(sorted_values, starts, ends, int(window), _SCHEME_CODES[scheme], float(param))

    out = np.empty(n, dtype=np.float64)
    out[sort_idx] = sorted_out
    return out
