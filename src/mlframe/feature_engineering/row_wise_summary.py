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

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

_QUANTILE_STATS = {"q10": 0.1, "q50": 0.5, "q90": 0.9}
_DEFAULT_STATS: tuple[str, ...] = ("mean", "std", "q10", "q50", "q90")


def row_wise_summary_stats(
    X: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    stats: Sequence[Union[str, float]] = _DEFAULT_STATS,
    column_prefix: str = "row_summary",
) -> pd.DataFrame:
    """Per-row summary statistics computed ACROSS a block of feature columns (not across rows).

    Parameters
    ----------
    X
        Feature frame.
    columns
        Column subset to summarize per row (default: every numeric column of ``X``).
    stats
        Any of ``"mean"``, ``"std"``, ``"min"``, ``"max"``, ``"median"``; ``"q{P}"`` (e.g. ``"q10"``) or a
        bare float in ``[0, 1]`` for an arbitrary quantile.
    column_prefix
        Output column-name prefix.

    Returns
    -------
    pd.DataFrame
        One column per requested stat, named ``{column_prefix}_{stat}``, same row count/order as ``X``.
        NaN values in the row are ignored (``nan``-aware reductions), matching the "how does this row look
        overall" intent even when some features are missing for that row.
    """
    cols = list(columns) if columns is not None else list(X.select_dtypes(include=[np.number]).columns)
    values = X[cols].to_numpy(dtype=np.float64)

    # np.nanquantile/np.nanmedian unconditionally take a slow per-ROW apply_along_axis path (one Python-level
    # call per row) regardless of whether any NaN is actually present, unlike np.nanmean/nanstd/nanmin/nanmax
    # (proper vectorized ufuncs) -- measured 39-54s at n=100,000 for a 5-stat block including 3 quantiles.
    # When the block has no NaN at all, np.quantile/np.median are fully vectorized C-level reductions with
    # identical results -- dispatching to them when safe cuts this to well under 1s (see bench_row_wise_
    # summary.py for the exact before/after numbers).
    has_nan = bool(np.isnan(values).any())
    quantile_fn = np.nanquantile if has_nan else np.quantile
    median_fn = np.nanmedian if has_nan else np.median

    out: dict[str, np.ndarray] = {}
    for stat in stats:
        stat_name = stat if isinstance(stat, str) else f"q{int(round(stat * 100))}"
        if stat == "mean":
            out[f"{column_prefix}_mean"] = np.nanmean(values, axis=1)
        elif stat == "std":
            out[f"{column_prefix}_std"] = np.nanstd(values, axis=1)
        elif stat == "min":
            out[f"{column_prefix}_min"] = np.nanmin(values, axis=1)
        elif stat == "max":
            out[f"{column_prefix}_max"] = np.nanmax(values, axis=1)
        elif stat == "median":
            out[f"{column_prefix}_median"] = median_fn(values, axis=1)
        elif isinstance(stat, str) and stat in _QUANTILE_STATS:
            out[f"{column_prefix}_{stat}"] = quantile_fn(values, _QUANTILE_STATS[stat], axis=1)
        elif isinstance(stat, str) and stat.startswith("q") and stat[1:].isdigit():
            out[f"{column_prefix}_{stat}"] = quantile_fn(values, int(stat[1:]) / 100.0, axis=1)
        elif isinstance(stat, (int, float)) and 0.0 <= stat <= 1.0:
            out[f"{column_prefix}_{stat_name}"] = quantile_fn(values, float(stat), axis=1)
        else:
            raise ValueError(f"row_wise_summary_stats: unrecognized stat {stat!r}")

    return pd.DataFrame(out, index=X.index)


__all__ = ["row_wise_summary_stats"]
