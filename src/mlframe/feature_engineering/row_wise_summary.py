"""``row_wise_summary_stats``: per-row cross-sectional summary statistics across a block of feature columns.

Source: Ubiquant Market Prediction 2nd place -- per-row cross-sectional aggregates (mean/std/quantiles
0.1/0.5/0.9 across all numeric features) as "macro" features summarizing an asset's overall state at a
timestamp. Distinct from every existing row-wise ``axis=1`` utility in mlframe (k-nearest-neighbour-weighted
aggregates in ``transformer/_aggregation.py``, spatial-neighbor distances, windowed TIME-series quantiles):
those all aggregate across ROWS (neighbors/time); this aggregates across COLUMNS for a single row -- a
"how does this row look overall, right now" summary rather than a "how does this row compare to its
neighbors/history" one.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

_QUANTILE_STATS = {"q10": 0.1, "q50": 0.5, "q90": 0.9}
_DEFAULT_STATS: tuple[str, ...] = ("mean", "std", "q10", "q50", "q90")

try:
    from numba import njit, prange

    @njit(parallel=True, cache=True, fastmath=False)
    def _row_nanquantile_njit(values: np.ndarray, qs: np.ndarray) -> np.ndarray:
        """Per-row (axis=1) NaN-aware quantile: one sort of the row's finite values, all ``qs`` read off it.

        ``np.nanquantile(values, qs, axis=1)`` dispatches to ``apply_along_axis`` (one Python-level call per
        row) regardless of row count; this does the same finite-value sort + linear-interpolation numpy uses,
        but as a single ``prange``-parallel pass with no per-row Python dispatch. Bit-identical to
        ``np.nanquantile`` to ~1e-15 (float64 rounding order only)."""
        n, p = values.shape
        nq = qs.shape[0]
        out = np.empty((nq, n), dtype=np.float64)
        for i in prange(n):
            row = values[i]
            buf = np.empty(p, dtype=np.float64)
            m = 0
            for j in range(p):
                v = row[j]
                if not np.isnan(v):
                    buf[m] = v
                    m += 1
            if m == 0:
                for k in range(nq):
                    out[k, i] = np.nan
                continue
            s = np.sort(buf[:m])
            for k in range(nq):
                q = qs[k]
                if m == 1:
                    out[k, i] = s[0]
                    continue
                idx = q * (m - 1)
                lo = int(np.floor(idx))
                hi = int(np.ceil(idx))
                frac = idx - lo
                out[k, i] = s[lo] if lo == hi else s[lo] * (1.0 - frac) + s[hi] * frac
        return out

    _HAS_NUMBA = True
except Exception:  # numba unavailable: np.nanquantile fallback below is exact, just slower.
    _HAS_NUMBA = False


def _summary_stats_block(values: np.ndarray, stats: Sequence[Union[str, float]], column_prefix: str) -> dict[str, np.ndarray]:
    """Compute the requested per-row stats for one ``(n_rows, n_cols)`` numeric block.

    Shared reduction core reused by both the flat (single-block) and grouped (multi-block) entry points.
    """
    # np.nanquantile/np.nanmedian unconditionally take a slow per-ROW apply_along_axis path (one Python-level
    # call per row) regardless of whether any NaN is actually present, unlike np.nanmean/nanstd/nanmin/nanmax
    # (proper vectorized ufuncs) -- measured 39-54s at n=100,000 for a 5-stat block including 3 quantiles.
    # When the block has no NaN at all, np.quantile/np.median are fully vectorized C-level reductions with
    # identical results -- dispatching to them when safe cuts this to well under 1s (see bench_row_wise_
    # summary.py for the exact before/after numbers).
    has_nan = bool(np.isnan(values).any())
    # When NaN IS present, np.nanquantile's apply_along_axis dominates the block cost regardless of the
    # single-call q-batching below (measured 20.2s -> 2.78s at n=200k/p=30/5% NaN, ~1e-16 max abs diff) --
    # the njit prange kernel does the identical finite-sort + linear-interpolation per row without the
    # per-row Python dispatch. "median" is routed through the SAME kernel (q=0.5 is the median by
    # definition) so it shares the one sort-per-row with the other quantile stats instead of a separate
    # np.nanmedian apply_along_axis pass. Falls back to np.nanquantile/np.nanmedian when numba is unavailable.
    _use_njit_nan_path = has_nan and _HAS_NUMBA
    quantile_fn = np.nanquantile if has_nan else np.quantile
    median_fn = np.nanmedian if has_nan else np.median

    out: dict[str, np.ndarray] = {}
    # Batch every quantile-type stat (q10/q50/q90/qNN/bare-float) into ONE quantile_fn call with a q-array
    # instead of one call per stat: np.quantile/np.nanquantile sort (partition) each row once per call
    # regardless of how many q values are requested, so N separate calls redundantly re-sort every row N
    # times -- one call with q=[q1, q2, ...] sorts each row once and reads off all requested percentiles.
    # Bit-identical to the per-stat calls (same algorithm, same interpolation default, just one shared sort).
    _quantile_names: list[str] = []
    _quantile_qs: list[float] = []
    for stat in stats:
        stat_name = stat if isinstance(stat, str) else f"q{round(stat * 100)}"
        if stat == "mean":
            out[f"{column_prefix}_mean"] = np.nanmean(values, axis=1)
        elif stat == "std":
            out[f"{column_prefix}_std"] = np.nanstd(values, axis=1)
        elif stat == "min":
            out[f"{column_prefix}_min"] = np.nanmin(values, axis=1)
        elif stat == "max":
            out[f"{column_prefix}_max"] = np.nanmax(values, axis=1)
        elif stat == "median":
            if _use_njit_nan_path:
                _quantile_names.append(f"{column_prefix}_median")
                _quantile_qs.append(0.5)
            else:
                out[f"{column_prefix}_median"] = median_fn(values, axis=1)
        elif isinstance(stat, str) and stat in _QUANTILE_STATS:
            _quantile_names.append(f"{column_prefix}_{stat}")
            _quantile_qs.append(_QUANTILE_STATS[stat])
        elif isinstance(stat, str) and stat.startswith("q") and stat[1:].isdigit():
            _quantile_names.append(f"{column_prefix}_{stat}")
            _quantile_qs.append(int(stat[1:]) / 100.0)
        elif isinstance(stat, (int, float)) and 0.0 <= stat <= 1.0:
            _quantile_names.append(f"{column_prefix}_{stat_name}")
            _quantile_qs.append(float(stat))
        else:
            raise ValueError(f"row_wise_summary_stats: unrecognized stat {stat!r}")
    if _use_njit_nan_path and _quantile_qs:
        _q_arr = np.asarray(_quantile_qs, dtype=np.float64)
        _q_result = _row_nanquantile_njit(np.ascontiguousarray(values, dtype=np.float64), _q_arr)
        out.update(zip(_quantile_names, _q_result))
    elif len(_quantile_qs) == 1:
        # A length-1 q ARRAY still adds a leading axis (shape (1, n_rows), not (n_rows,)) -- pass the
        # scalar directly so the single-quantile case matches the original per-stat call's output shape.
        out[_quantile_names[0]] = quantile_fn(values, _quantile_qs[0], axis=1)
    elif _quantile_qs:
        _q_result = quantile_fn(values, _quantile_qs, axis=1)
        out.update(zip(_quantile_names, _q_result))
    return out


def row_wise_summary_stats(
    X: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    stats: Sequence[Union[str, float]] = _DEFAULT_STATS,
    column_prefix: str = "row_summary",
    groups: Optional[Mapping[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """Per-row summary statistics computed ACROSS a block of feature columns (not across rows).

    Parameters
    ----------
    X
        Feature frame.
    columns
        Column subset to summarize per row (default: every numeric column of ``X``). Ignored when ``groups``
        is given.
    stats
        Any of ``"mean"``, ``"std"``, ``"min"``, ``"max"``, ``"median"``; ``"q{P}"`` (e.g. ``"q10"``) or a
        bare float in ``[0, 1]`` for an arbitrary quantile.
    column_prefix
        Output column-name prefix (flat mode) / prefix stem (grouped mode, see below).
    groups
        Opt-in: ``{group_name: column_names}`` to compute the SAME ``stats`` separately per named column
        group in one call instead of flattening every column into a single summary. Different feature
        families (e.g. price-scale vs volume-scale vs ratio columns) have very different scales/meanings --
        a flat mean/std across all of them blurs the families together, while per-group stats keep each
        family's signal intact. Output columns are named ``{column_prefix}_{group_name}_{stat}``. Reuses the
        same per-row reduction core once per group -- equivalent to (but faster and more convenient than)
        calling ``row_wise_summary_stats`` once per group and ``pd.concat``-ing the results, since the
        source ``DataFrame`` is sliced to ``float64`` only once overall (one ``.to_numpy()`` per group,
        same as the manual-loop alternative would need, but without per-call Python/pandas overhead or the
        caller having to manage the concat/naming itself).

    Returns
    -------
    pd.DataFrame
        One column per requested stat (flat mode), or per ``group x stat`` (grouped mode), same row
        count/order as ``X``. NaN values in the row are ignored (``nan``-aware reductions), matching the
        "how does this row look overall" intent even when some features are missing for that row.
    """
    if groups is not None:
        out_grouped: dict[str, np.ndarray] = {}
        for group_name, group_cols in groups.items():
            values = X[list(group_cols)].to_numpy(dtype=np.float64)
            out_grouped.update(_summary_stats_block(values, stats, f"{column_prefix}_{group_name}"))
        return pd.DataFrame(out_grouped, index=X.index)

    cols = list(columns) if columns is not None else list(X.select_dtypes(include=[np.number]).columns)
    values = X[cols].to_numpy(dtype=np.float64)
    out = _summary_stats_block(values, stats, column_prefix)
    return pd.DataFrame(out, index=X.index)


__all__ = ["row_wise_summary_stats"]
