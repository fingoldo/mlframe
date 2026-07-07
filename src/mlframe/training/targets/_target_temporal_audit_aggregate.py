"""Granularity picker + group-by-time aggregation kernels for temporal target audit.

Carved out of ``target_temporal_audit`` so the parent facade stays under budget.
This module depends on the timestamp coercion helper (sibling
``_target_temporal_audit_coerce``) and produces pandas frames consumed by the
audit runners in the parent.
"""
from __future__ import annotations

import math
from typing import Literal, Sequence

import pandas as pd

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False

from ._target_temporal_audit_coerce import coerce_timestamps_for_audit

DEFAULT_TARGET_BINS_RANGE: tuple[int, int] = (30, 50)
"""Granularity auto-picker aims for this many non-empty bins."""

Granularity = Literal["minute", "hour", "day", "week", "month", "quarter", "year"]
_GRANULARITY_ORDER: list[Granularity] = [
    "minute", "hour", "day", "week", "month", "quarter", "year",
]
_GRANULARITY_SECONDS: dict[Granularity, float] = {
    "minute": 60.0,
    "hour": 3600.0,
    "day": 86_400.0,
    "week": 7 * 86_400.0,
    "month": 30.44 * 86_400.0,
    "quarter": 91.31 * 86_400.0,
    "year": 365.25 * 86_400.0,
}


def _pick_granularity(
    timestamps: Sequence,
    target_bins_range: tuple[int, int] = DEFAULT_TARGET_BINS_RANGE,
) -> Granularity:
    """Choose a bin width that yields ~30-50 non-empty bins.

    Strategy: pick the smallest granularity whose total span / bin
    size sits inside ``target_bins_range``. If no granularity fits
    perfectly, return the one with the count closest to the geometric
    mean of the range (sqrt(30 * 50) ~= 38).

    Accepts ``(min_ts, max_ts)`` as a 2-tuple shortcut: callers with
    direct polars / numpy / pandas span knowledge can skip the
    ``pd.to_datetime(pd.Series(...))`` materialisation of the full
    1M+ timestamp array (which costs ~1s on 1M rows for a probe that
    only needs the two extremes).
    """
    # Fast path: caller pre-computed (min, max). Avoids the O(N)
    # round-trip through pd.Series for 1M+ row inputs.
    if isinstance(timestamps, tuple) and len(timestamps) == 2:
        _t_min, _t_max = timestamps
        if _t_min is None or _t_max is None:
            return "month"
        ts_min = pd.Timestamp(_t_min)
        ts_max = pd.Timestamp(_t_max)
        span_seconds = (ts_max - ts_min).total_seconds()
        if span_seconds <= 0:
            return "month"
    else:
        if len(timestamps) == 0:
            return "month"
        ts = pd.Series(coerce_timestamps_for_audit(timestamps))
        span_seconds = (ts.max() - ts.min()).total_seconds()
        if span_seconds <= 0:
            return "month"

    target_geomean = math.sqrt(target_bins_range[0] * target_bins_range[1])
    best: Granularity | None = None
    best_score = math.inf

    for g in _GRANULARITY_ORDER:
        n = span_seconds / _GRANULARITY_SECONDS[g]
        if target_bins_range[0] <= n <= target_bins_range[1]:
            return g
        score = abs(math.log(n) - math.log(target_geomean)) if n > 0 else math.inf
        if score < best_score:
            best_score = score
            best = g

    return best or "month"


_POLARS_BIN_TRUNCATE: dict[Granularity, str] = {
    "minute": "1m",
    "hour": "1h",
    "day": "1d",
    "week": "1w",
    "month": "1mo",
    "quarter": "1q",
    "year": "1y",
}


def _polars_rate_expr(target_col: str, target_type: str, alias: str):
    """Build a polars aggregation expr for one target's per-bin rate."""
    if target_type == "binary_classification":
        # P(y=1): treat null as 0, then mean over (val > 0)
        return (pl.col(target_col).fill_null(0) > 0).cast(pl.Float64).mean().alias(alias)
    return pl.col(target_col).cast(pl.Float64).mean().alias(alias)


def _aggregate_by_time_polars(
    df,
    timestamp_col: str,
    target_col: str,
    granularity: Granularity,
    *,
    target_type: str,
) -> pd.DataFrame:
    """Polars-native group-by-time aggregation, single target. Returns
    pandas DF for downstream sklearn / matplotlib compatibility."""
    if not _HAS_POLARS:
        raise ImportError("polars not installed; pass a pandas df instead.")
    bin_expr = pl.col(timestamp_col).dt.truncate(_POLARS_BIN_TRUNCATE[granularity])
    rate_expr = _polars_rate_expr(target_col, target_type, "target_rate")

    from ..utils import get_pandas_view_of_polars_df as _get_pandas_view
    # Arrow-backed split-blocks bridge: ~32x faster than default .to_pandas() on
    # the per-bin aggregate -- bin counts grow O(n_obs / bin_width) so the saving
    # scales with input frame size.
    agg = _get_pandas_view(
        df.select([timestamp_col, target_col]).with_columns(bin_expr.alias("__bin")).group_by("__bin").agg(pl.len().alias("n_obs"), rate_expr).sort("__bin")
    )
    agg = agg.rename(columns={"__bin": "bin_start"})
    agg["bin_start"] = pd.to_datetime(agg["bin_start"])
    return agg


def _aggregate_by_time_polars_multi(
    df,
    timestamp_col: str,
    target_specs: list[tuple[str, str, str]],
    granularity: Granularity,
) -> pd.DataFrame:
    """Polars multi-target group-by-time aggregation: one pass over the
    data computes per-bin n_obs and per-target rate columns side-by-side.

    Parameters
    ----------
    df : pl.DataFrame
    timestamp_col : str
    target_specs : list of (target_col, target_type, alias)
        ``target_col`` is the source column name in df. ``alias`` is
        the column name to use for that target's rate in the output
        (typically ``f"rate__{target_col}"`` or any caller-chosen
        identifier so multiple targets sharing one source col don't
        collide).
    granularity : Granularity

    Returns
    -------
    pandas.DataFrame
        Columns: ``bin_start``, ``n_obs``, plus one rate column per
        target_spec (named by the spec's ``alias``). Sorted by
        ``bin_start``. The rate columns can then be sliced per-target
        by callers without re-running the groupby.
    """
    if not _HAS_POLARS:
        raise ImportError("polars not installed; pass a pandas df instead.")
    if not target_specs:
        raise ValueError("target_specs must be non-empty.")

    bin_expr = pl.col(timestamp_col).dt.truncate(_POLARS_BIN_TRUNCATE[granularity])
    rate_exprs = [_polars_rate_expr(col, ttype, alias) for (col, ttype, alias) in target_specs]
    select_cols = [timestamp_col, *{spec[0] for spec in target_specs}]

    from ..utils import get_pandas_view_of_polars_df as _get_pandas_view
    # Arrow-backed split-blocks bridge -- same rationale as the single-target variant
    # above; multi-target aggregates have wider output columns so the saving
    # (avoided per-column consolidation copy) compounds.
    agg = _get_pandas_view(
        df.select(select_cols).with_columns(bin_expr.alias("__bin")).group_by("__bin").agg(pl.len().alias("n_obs"), *rate_exprs).sort("__bin")
    )
    agg = agg.rename(columns={"__bin": "bin_start"})
    agg["bin_start"] = pd.to_datetime(agg["bin_start"])
    return agg


def _aggregate_by_time_pandas(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    granularity: Granularity,
    *,
    target_type: str,
) -> pd.DataFrame:
    """Pandas-side aggregation for non-polars callers / unit tests."""
    s = df[[timestamp_col, target_col]].copy()
    s[timestamp_col] = pd.to_datetime(s[timestamp_col])
    # Period-frequency strings (pandas accepts a different set than offset
    # aliases -- "QS" works as an offset but not as a Period frequency).
    # Use Period for clean bin labels, then materialise.
    period_freq = {
        "minute": "min",
        "hour": "h",
        "day": "D",
        "week": "W",
        "month": "M",
        "quarter": "Q",
        "year": "Y",
    }[granularity]
    s["__bin"] = s[timestamp_col].dt.to_period(period_freq).dt.to_timestamp()

    if target_type == "binary_classification":
        # The prior `fillna(0) > 0` counted NaN rows as negatives, deflating
        # the reported positive rate per bin. Drop NaN explicitly so the rate
        # is the honest empirical positive fraction over non-missing rows;
        # NaN fraction is surfaced separately via n_obs.
        rate = s.groupby("__bin")[target_col].apply(lambda c: (c.dropna() > 0).mean() if c.notna().any() else float("nan"))
    else:
        rate = s.groupby("__bin")[target_col].mean()
    n_obs = s.groupby("__bin")[target_col].size()

    agg = pd.DataFrame({
        "bin_start": rate.index,
        "n_obs": n_obs.values,
        "target_rate": rate.values,
    }).sort_values("bin_start").reset_index(drop=True)
    return agg


def _format_bin_label(ts: pd.Timestamp, granularity: Granularity) -> str:
    fmt = {
        "minute": "%Y-%m-%d %H:%M",
        "hour": "%Y-%m-%d %H",
        "day": "%Y-%m-%d",
        "week": "%Y-%m-%d (W)",
        "month": "%Y-%m",
        "quarter": "%Y-Q%q",
        "year": "%Y",
    }[granularity]
    if granularity == "quarter":
        return f"{ts.year}-Q{(ts.month - 1) // 3 + 1}"
    return ts.strftime(fmt)
