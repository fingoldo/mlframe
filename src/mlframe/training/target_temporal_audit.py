"""Temporal target audit вЂ” detect P(y) shifts over time, find change points.

The user's production drift incident (Upwork jobs-hired forward-mode
training) revealed that:

- The target rate isn't constant over time. The historical period was
  positive-only (selection-biased) at ~98%; a 4-month window in
  2021-2022 had unbiased ~40% rate; the recent month (full uid scrape)
  is unbiased ~40% again.
- Without a per-bin time-series view of the target, this regime change
  is invisible вЂ” the operator sees the AGGREGATE rate (74%) and
  thinks they have a balanced classification problem when in fact
  they have multiple regimes mixed.

This module ships:

- `audit_target_over_time(...)` вЂ” group-by-time + change-point
  detection, returns a structured `TemporalAuditResult`.
- `plot_target_over_time(...)` вЂ” matplotlib-based time-series plot
  with change points marked, saves to disk (or returns Figure).
- `_pick_granularity(...)` вЂ” auto-pick a time bin width that yields
  30-50 non-empty bins (configurable via env / args).
- `find_change_points_zscore(...)` вЂ” generic 1-D change-point
  detector. Could move to `pyutilz` later (useful for trading too вЂ”
  the user said "Р±СѓРґРµС‚ РїРѕР»РµР·РЅРѕ Рё РґР»СЏ С‚СЂРµР№РґРёРЅРіР° РїРѕР·Р¶Рµ").

Wiring: opt-in via `TrainingBehaviorConfig.target_temporal_audit_*`.
When the config field `target_temporal_audit_column` is set,
`train_mlframe_models_suite` runs the audit per-target right before
training, logs the report, plots to disk under the per-target charts
folder, and stores the structured result on `metadata`.

Public surface:
- audit_target_over_time
- plot_target_over_time
- find_change_points_zscore
- TemporalAuditResult (dataclass)
- DEFAULT_MIN_BIN_FRACTION_FOR_FILTER  вЂ” bin must have в‰Ґ this many obs
  to be plotted / used; default 0.5 of the median bin size.
- DEFAULT_ZSCORE_THRESHOLD вЂ” 3.0 (3 MADs from rolling median)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False

def _import_ruptures():
    """Lazy import of ``ruptures`` so callers using only the z-score detector
    never pay the import cost (ruptures pulls in scipy.sparse.linalg + numba
    extensions; ~150ms cold). ``ImportError`` propagates to the caller; the
    z-score path is dependency-free and the obvious fallback.
    """
    import ruptures as rpt  # type: ignore[import-not-found]
    return rpt

logger = logging.getLogger(__name__)


# Lower / upper datetime sanity bounds for auto-unit detection.
_AUDIT_DATETIME_LOW_NS = np.int64(0)  # 1970-01-01 in nanoseconds-since-epoch
_AUDIT_DATETIME_HIGH_NS = np.int64(7258118400_000_000_000)  # 2200-01-01 in ns
_AUDIT_UNIT_NS_FACTOR: dict[str, int] = {
    "s":  1_000_000_000,
    "ms":     1_000_000,
    "us":         1_000,
    "ns":             1,
}


def coerce_timestamps_for_audit(
    arr,
    *,
    explicit_unit: str | None = None,
) -> np.ndarray:
    """Coerce a numeric timestamp array to numpy datetime64[ns] for audit binning.

    Unit auto-detection (when ``explicit_unit`` is None):
    - Pure datetime64 input or pd.DatetimeIndex / pd.Series of datetimes: returned as-is.
    - Numeric input: try each candidate unit in (s, ms, us, ns) in COARSEST-first
      order. For each, compute resulting timestamp range. Accept the first unit
      whose entire min..max range lands in [1970-01-01, 2200-01-01]; this gives
      the audit the widest meaningful time-span for binning. Coarsest-first matters
      because a value like 1_700_000_000 is legal as ns (=1.7 sec past epoch,
      degenerate single-bin audit) AND as s (=2023-11-14, useful audit); we want
      the latter.
    - If no unit is in-range, log a warning and fall back to ns to preserve
      pre-existing behaviour.

    ``explicit_unit`` (optional): forces the unit interpretation. Must be one of
    {'s', 'ms', 'us', 'ns'}.

    Returns a 1-D numpy datetime64[ns] array. Accepts numpy ndarray, list,
    pd.Series, or any 1-D ``Sequence``-like input.
    """
    arr = np.asarray(arr) if not isinstance(arr, np.ndarray) else arr

    if np.issubdtype(arr.dtype, np.datetime64):
        return arr
    if not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)):
        # Object / string / pd.Timestamp etc. -- let pandas figure it out (string parsing path).
        # Wave 34 P2 fix (2026-05-20): if the input contains tz-aware
        # pd.Timestamp entries, ``pd.to_datetime(arr).to_numpy()`` returns
        # an OBJECT dtype (tz info preserved as Timestamp objects), NOT
        # the documented ``datetime64[ns]`` invariant. Downstream callers
        # (recommended_filter_mask at L335, compute_ml_perf_by_time)
        # silently get a mixed-dtype array; Grouper / comparisons
        # then behave inconsistently across runs.
        # Force the documented contract: coerce to UTC-aware via
        # ``utc=True``, then strip the tz so the resulting numpy view IS
        # ``datetime64[ns]``. WARN-log when tz info gets dropped so the
        # operator knows the audit assumes UTC.
        _coerced = pd.to_datetime(arr, utc=True, errors="coerce")
        _had_tz = (getattr(_coerced.dtype, "tz", None) is not None)
        if _had_tz:
            _coerced = _coerced.tz_convert("UTC").tz_localize(None)
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "coerce_timestamps_for_audit: input contained tz-aware "
                "Timestamps; converted to UTC then tz-stripped so the "
                "returned dtype matches the documented datetime64[ns] "
                "contract. If your data are not UTC-anchored, localise "
                "BEFORE calling this helper."
            )
        return _coerced.to_numpy().astype("datetime64[ns]")

    # pandas 2.0+ ``DatetimeIndex.to_numpy()`` can return ``datetime64[s]`` /
    # ``[ms]`` / ``[us]`` (preserved resolution) instead of always ``[ns]``.
    # Downstream ``.view("int64")`` callers (``_normalize_timestamps``) then
    # read the raw integer in the original unit, not nanoseconds, and
    # epoch-second values collapse to 1970. Force ``datetime64[ns]`` so the
    # documented contract holds across pandas versions.
    def _to_ns(_dti):
        return _dti.to_numpy().astype("datetime64[ns]")

    if arr.size == 0:
        return _to_ns(pd.to_datetime(arr, unit=explicit_unit or "ns"))

    if explicit_unit is not None:
        if explicit_unit not in _AUDIT_UNIT_NS_FACTOR:
            raise ValueError(
                f"audit timestamp unit={explicit_unit!r} not in "
                f"{sorted(_AUDIT_UNIT_NS_FACTOR)!r}"
            )
        return _to_ns(pd.to_datetime(arr, unit=explicit_unit))

    # Auto-detect: coarsest-first so degenerate ns-as-seconds reads pick "s" not "ns".
    _vmin_f = float(np.nanmin(arr))
    _vmax_f = float(np.nanmax(arr))
    for _unit in ("s", "ms", "us", "ns"):
        _ns_factor = _AUDIT_UNIT_NS_FACTOR[_unit]
        _lo_ns = _vmin_f * _ns_factor
        _hi_ns = _vmax_f * _ns_factor
        if _AUDIT_DATETIME_LOW_NS <= _lo_ns and _hi_ns <= _AUDIT_DATETIME_HIGH_NS:
            return _to_ns(pd.to_datetime(arr, unit=_unit))

    logger.warning(
        "audit timestamp values (min=%g, max=%g) fall outside [1970-01-01, 2200-01-01] "
        "under every candidate unit (s/ms/us/ns). Falling back to ns interpretation; "
        "audit may degenerate to a single bin. Set explicit unit to override.",
        _vmin_f, _vmax_f,
    )
    return _to_ns(pd.to_datetime(arr, unit="ns"))


DEFAULT_MIN_BIN_FRACTION_FOR_FILTER: float = 0.5
"""A bin is kept (plotted, used in change-point detection) only if its
n_obs >= this fraction Г— median bin size. Filters out the tiny tail
bins that would dominate the curve with high-variance noise."""

DEFAULT_ZSCORE_THRESHOLD: float = 3.0
"""Default threshold for the z-score change-point detector. A bin is
flagged if |rate - rolling_median| > threshold Г— rolling_MAD (modified
z-score)."""

DEFAULT_ZSCORE_WINDOW: int = 7
"""Default centred rolling-window width for the z-score detector.
Wider windows = smoother baseline, less sensitive to short anomalies.
7 catches month-long dips on a monthly-binned series."""

DEFAULT_TARGET_BINS_RANGE: tuple[int, int] = (30, 50)
"""Granularity auto-picker aims for this many non-empty bins."""

DEFAULT_PELT_MODEL: str = "l2"
"""Default cost model for ruptures.Pelt: ``"l2"`` (mean-shift detection
via squared error). Other choices: ``"l1"`` (median-shift, more
robust to outliers), ``"rbf"`` (kernel-based, detects general
distributional changes вЂ” slower, sensitive to penalty)."""

DEFAULT_PELT_MIN_SEGMENT_SIZE: int = 2
"""Minimum number of bins per segment in Pelt. ``2`` is the natural
floor for change-point detection (a "segment" of one bin is just an
outlier, not a regime). Raise to 3+ to filter out short transients."""

ChangePointMethod = Literal["pelt", "zscore"]

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


@dataclass
class TimeBin:
    """One row of the time-binned target series."""
    bin_label: str           # e.g. "2024-01" (display)
    bin_start: pd.Timestamp  # bin's left edge (sorting / interval math)
    n_obs: int               # number of rows in this bin
    target_rate: float       # binary: P(y=1); regression: mean(y)
    kept: bool = True        # False if filtered out as too-sparse


@dataclass
class TemporalAuditResult:
    """Structured outcome of a temporal target audit."""
    target_name: str
    target_type: str
    timestamp_col: str
    granularity: Granularity
    bins: list[TimeBin]
    change_point_indices: list[int]  # indices into kept-bin list
    segments: list[dict[str, Any]]   # [{start: ..., end: ..., mean_rate: ..., n_obs: ..., n_bins: ...}, ...]
    warnings: list[str]
    actionable: dict[str, Any] = field(default_factory=dict)
    plot_path: str | None = None  # filled in if plot_target_over_time saves a file

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe round-trip-friendly dict (for metadata storage)."""
        return {
            "target_name": self.target_name,
            "target_type": self.target_type,
            "timestamp_col": self.timestamp_col,
            "granularity": self.granularity,
            "bins": [
                {
                    "bin_label": b.bin_label,
                    "bin_start": b.bin_start.isoformat() if hasattr(b.bin_start, "isoformat") else str(b.bin_start),
                    "n_obs": b.n_obs,
                    "target_rate": b.target_rate,
                    "kept": b.kept,
                }
                for b in self.bins
            ],
            "change_point_indices": self.change_point_indices,
            "segments": self.segments,
            "warnings": self.warnings,
            "actionable": self.actionable,
            "plot_path": self.plot_path,
        }

    def recommended_filter_mask(
        self,
        timestamps: Any,
        *,
        segment: str = "most_recent_stable",
    ) -> np.ndarray:
        """Return a boolean mask selecting rows whose timestamp falls
        inside the chosen audit segment.

        Used as the actionable bridge between "audit detected drift
        across N segments" and "let me actually train on the right
        subset". Common pattern:

            >>> result = audit_target_over_time(df, "ts", "y", ...)
            >>> mask = result.recommended_filter_mask(df["ts"])
            >>> df_clean = df.filter(mask) if isinstance(df, pl.DataFrame) else df[mask]

        Parameters
        ----------
        timestamps : array-like (numpy / pandas / polars)
            The timestamp values for which to compute the mask. Length
            of returned array equals length of this input.
        segment : str, default "most_recent_stable"
            Selection rule:
            * ``"most_recent_stable"`` вЂ” last segment with n_bins в‰Ґ 3
              (the operator-friendly default вЂ” most-recent regime,
              skipping single-bin spikes).
            * ``"largest"`` вЂ” segment with the most observations.
            * ``"all_stable"`` вЂ” all segments with n_bins в‰Ґ 3 (omits
              short transient regimes).
            * ``"first"`` / ``"last"`` вЂ” first / last segment by index.

        Returns
        -------
        ndarray of bool
            Same length as ``timestamps``. True for rows whose
            timestamp lies in [segment.start_bin, segment.end_bin)
            (the bin's exclusive end is the next bin's start, so
            inclusion is half-open as Pelt segments are).
        """
        if not self.segments or not self.bins:
            ts = coerce_timestamps_for_audit(timestamps)
            return np.zeros(len(ts), dtype=bool)

        kept = [b for b in self.bins if b.kept]
        if not kept:
            ts = coerce_timestamps_for_audit(timestamps)
            return np.zeros(len(ts), dtype=bool)

        chosen_segs: list[dict[str, Any]]
        if segment == "most_recent_stable":
            chosen_segs = []
            for s in reversed(self.segments):
                if s["n_bins"] >= 3:
                    chosen_segs = [s]
                    break
            if not chosen_segs:
                chosen_segs = [self.segments[-1]]
        elif segment == "largest":
            chosen_segs = [max(self.segments, key=lambda s: s["n_obs"])]
        elif segment == "all_stable":
            chosen_segs = [s for s in self.segments if s["n_bins"] >= 3]
            if not chosen_segs:
                chosen_segs = list(self.segments)
        elif segment == "first":
            chosen_segs = [self.segments[0]]
        elif segment == "last":
            chosen_segs = [self.segments[-1]]
        else:
            raise ValueError(
                f"Unknown segment selector: {segment!r}. "
                "Choose from: most_recent_stable / largest / all_stable / first / last."
            )

        # Compute (start_ts, end_ts) for each chosen segment. The
        # segment's start_idx is into the kept-bin list; the bin's
        # ``bin_start`` is the inclusive left edge. The end is the
        # NEXT bin's left edge OR (if we're at the tail) the last
        # kept bin's bin_start + the granularity span.
        gran_seconds = _GRANULARITY_SECONDS.get(self.granularity, 30.44 * 86_400.0)
        gran_delta = pd.Timedelta(seconds=gran_seconds)

        # Wave 34 P1 fix (2026-05-20): pre-fix the ``ts`` Series and the
        # ``start_ts/end_ts`` Timestamps could have mismatched tz state.
        # If ``bin_start`` was built from a tz-aware polars Datetime
        # (``Datetime(time_zone="UTC")``), the kept[..].bin_start values
        # are tz-aware while ``coerce_timestamps_for_audit`` returns
        # tz-naive datetime64[ns]. Direct ``ts >= start_ts`` then raises
        # ``TypeError: Cannot compare tz-naive and tz-aware``.
        # Coerce ``ts`` to a pandas Series and normalise BOTH sides to
        # tz-naive (the documented audit invariant): the
        # ``coerce_timestamps_for_audit`` helper above already enforces
        # this for the LHS; the RHS needs symmetric handling.
        ts = pd.Series(coerce_timestamps_for_audit(timestamps))

        def _normalize_bin_ts(_t):
            """Strip tz on a Timestamp / datetime if present so the
            comparison against the tz-naive ``ts`` Series succeeds.
            ``coerce_timestamps_for_audit`` has already converted any
            tz-aware input to UTC then dropped tz; do the same here for
            consistency."""
            _tz = getattr(_t, "tz", None) or getattr(_t, "tzinfo", None)
            if _tz is not None:
                return _t.tz_convert("UTC").tz_localize(None) if hasattr(_t, "tz_convert") else _t.astimezone(None).replace(tzinfo=None)
            return _t

        mask = np.zeros(len(ts), dtype=bool)
        for s in chosen_segs:
            start_idx = int(s["start_idx"])
            end_idx = int(s["end_idx"])  # exclusive into kept[]
            if start_idx >= len(kept):
                continue
            start_ts = _normalize_bin_ts(kept[start_idx].bin_start)
            if end_idx < len(kept):
                end_ts = _normalize_bin_ts(kept[end_idx].bin_start)
            else:
                # Tail segment вЂ” use last bin's left edge + granularity
                # span as the (open) right edge.
                end_ts = _normalize_bin_ts(kept[-1].bin_start) + gran_delta
            seg_mask = (ts >= start_ts) & (ts < end_ts)
            mask |= seg_mask.to_numpy() if hasattr(seg_mask, "to_numpy") else np.asarray(seg_mask)
        return mask


# -----------------------------------------------------------------------------
# Granularity picker
# -----------------------------------------------------------------------------

def _pick_granularity(
    timestamps: Sequence,
    target_bins_range: tuple[int, int] = DEFAULT_TARGET_BINS_RANGE,
) -> Granularity:
    """Choose a bin width that yields ~30-50 non-empty bins.

    Strategy: pick the smallest granularity whose total span / bin
    size sits inside ``target_bins_range``. If no granularity fits
    perfectly, return the one with the count closest to the geometric
    mean of the range (sqrt(30 * 50) в‰€ 38).

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
        # Distance (in log space) from the geometric-mean target count
        score = abs(math.log(n) - math.log(target_geomean)) if n > 0 else math.inf
        if score < best_score:
            best_score = score
            best = g

    return best or "month"


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------

_POLARS_BIN_TRUNCATE: dict[Granularity, str] = {
    "minute": "1m",
    "hour": "1h",
    "day": "1d",
    "week": "1w",
    "month": "1mo",
    "quarter": "1q",
    "year": "1y",
}


def _polars_rate_expr(target_col: str, target_type: str, alias: str) -> pl.Expr:
    """Build a polars aggregation expr for one target's per-bin rate."""
    if target_type == "binary_classification":
        # P(y=1): treat null as 0, then mean over (val > 0)
        return (pl.col(target_col).fill_null(0) > 0).cast(pl.Float64).mean().alias(alias)
    return pl.col(target_col).cast(pl.Float64).mean().alias(alias)


def _aggregate_by_time_polars(
    df: pl.DataFrame,
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

    from .utils import get_pandas_view_of_polars_df as _get_pandas_view
    # Arrow-backed split-blocks bridge: ~32x faster than default .to_pandas() on
    # the per-bin aggregate -- bin counts grow O(n_obs / bin_width) so the saving
    # scales with input frame size.
    agg = _get_pandas_view(
        df.select([timestamp_col, target_col])
          .with_columns(bin_expr.alias("__bin"))
          .group_by("__bin")
          .agg(pl.len().alias("n_obs"), rate_expr)
          .sort("__bin")
    )
    agg = agg.rename(columns={"__bin": "bin_start"})
    agg["bin_start"] = pd.to_datetime(agg["bin_start"])
    return agg


def _aggregate_by_time_polars_multi(
    df: pl.DataFrame,
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
    rate_exprs = [
        _polars_rate_expr(col, ttype, alias)
        for (col, ttype, alias) in target_specs
    ]
    select_cols = [timestamp_col, *{spec[0] for spec in target_specs}]

    from .utils import get_pandas_view_of_polars_df as _get_pandas_view
    # Arrow-backed split-blocks bridge -- same rationale as the single-target variant
    # above; multi-target aggregates have wider output columns so the saving
    # (avoided per-column consolidation copy) compounds.
    agg = _get_pandas_view(
        df.select(select_cols)
          .with_columns(bin_expr.alias("__bin"))
          .group_by("__bin")
          .agg(pl.len().alias("n_obs"), *rate_exprs)
          .sort("__bin")
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
    # Period-frequency strings (pandas accepts a different set than
    # offset aliases вЂ” e.g. "QS" works as an offset but not as a Period
    # frequency). Use Period for clean bin labels, then materialise.
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
        # Wave 50 (2026-05-20): the prior `fillna(0) > 0` counted NaN rows as
        # negatives, deflating reported positive rate per bin. Drop NaN
        # explicitly so the rate is the honest empirical positive fraction
        # over non-missing rows; NaN fraction is surfaced separately via n_obs.
        rate = s.groupby("__bin")[target_col].apply(
            lambda c: (c.dropna() > 0).mean() if c.notna().any() else float("nan")
        )
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


# -----------------------------------------------------------------------------
# Change-point detection (1-D, generic)
# -----------------------------------------------------------------------------


# Wave 106 (2026-05-21): change-point detection helpers moved to
# sibling _target_temporal_changepoint.py.
from ._target_temporal_changepoint import (  # noqa: F401, E402
    find_change_points_pelt,
    find_change_points_zscore,
    find_change_points,
    _segments_from_change_points,
)

def audit_target_over_time(
    df: Any,
    timestamp_col: str,
    target_col: str,
    *,
    target_name: str | None = None,
    target_type: str = "binary_classification",
    granularity: str = "auto",
    min_bin_fraction: float = DEFAULT_MIN_BIN_FRACTION_FOR_FILTER,
    method: ChangePointMethod = "pelt",
    pelt_model: str = DEFAULT_PELT_MODEL,
    pelt_penalty: float | None = None,
    pelt_min_segment_size: int = DEFAULT_PELT_MIN_SEGMENT_SIZE,
    z_threshold: float = DEFAULT_ZSCORE_THRESHOLD,
    z_window: int | None = None,
    min_anomaly_run: int = 2,
    drift_warn_threshold: float = 0.10,
) -> TemporalAuditResult:
    """Audit a target column over time, return a structured result.

    Steps:
    1. Coerce to pandas (lazy вЂ” uses polars groupby if df is pl.DataFrame).
    2. Pick granularity (or use the supplied one) to land in [30, 50] bins.
    3. Aggregate ``target_col`` by time bin.
    4. Filter out sparse bins (< `min_bin_fraction` Г— median(n_obs)).
    5. Run z-score change-point detection.
    6. Split into segments, compute per-segment mean_rate.
    7. Build warnings + actionable summary.

    Parameters
    ----------
    df : polars.DataFrame or pandas.DataFrame
        Must contain ``timestamp_col`` (datetime-typed) and ``target_col``.
    timestamp_col : str
        Name of the datetime column used for binning.
    target_col : str
        Name of the target column.
    target_name : str, optional
        Display name for the target (defaults to ``target_col``).
    target_type : str
        ``"binary_classification"`` (rate = P(y=1)) or any other string
        (rate = mean(target)). Multiclass / multilabel are NOT
        supported in this audit yet вЂ” fall through to mean(y).
    granularity : str
        ``"auto"`` (default) or one of minute/hour/day/week/month/
        quarter/year.
    min_bin_fraction : float
        Bin kept only if n_obs >= this Г— median(n_obs). 0.5 by default
        вЂ” drops the tail bins that have noisy rates.
    z_threshold : float
        Modified-z-score threshold for change-point detection.
    z_window : int
        Rolling window for z-score baseline (centred, odd-padded).
    min_anomaly_run : int
        Minimum consecutive bins to count as a change point. 2 = ignore
        single-bin spikes.
    drift_warn_threshold : float
        Spread between min and max segment mean_rate above which a
        WARN is emitted. 0.10 = a 10pp swing across segments.

    Returns
    -------
    TemporalAuditResult
        Structured report with bins, change_point_indices, segments,
        warnings, and an ``actionable`` dict that includes
        ``most_recent_stable_segment`` (operator can use this as a
        candidate training window).
    """
    target_name = target_name or target_col

    # 1. Aggregation вЂ” prefer polars path when input is polars.
    if _HAS_POLARS and isinstance(df, pl.DataFrame):
        # Granularity check needs timestamps as a python iterable
        ts_for_picker = df[timestamp_col].to_list()
        chosen = (
            _pick_granularity(ts_for_picker)
            if granularity == "auto"
            else granularity
        )
        agg = _aggregate_by_time_polars(
            df, timestamp_col, target_col, chosen, target_type=target_type,
        )
    else:
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        chosen = (
            _pick_granularity(df[timestamp_col])
            if granularity == "auto"
            else granularity
        )
        agg = _aggregate_by_time_pandas(
            df, timestamp_col, target_col, chosen, target_type=target_type,
        )

    return _audit_from_agg(
        agg=agg,
        target_name=target_name,
        target_type=target_type,
        timestamp_col=timestamp_col,
        granularity=chosen,  # type: ignore[arg-type]
        min_bin_fraction=min_bin_fraction,
        method=method,
        pelt_model=pelt_model,
        pelt_penalty=pelt_penalty,
        pelt_min_segment_size=pelt_min_segment_size,
        z_threshold=z_threshold,
        z_window=z_window,
        min_anomaly_run=min_anomaly_run,
        drift_warn_threshold=drift_warn_threshold,
    )


def audit_targets_over_time(
    df: Any,
    timestamp_col: str,
    targets: dict[str, Any],
    *,
    granularity: str = "auto",
    min_bin_fraction: float = DEFAULT_MIN_BIN_FRACTION_FOR_FILTER,
    method: ChangePointMethod = "pelt",
    pelt_model: str = DEFAULT_PELT_MODEL,
    pelt_penalty: float | None = None,
    pelt_min_segment_size: int = DEFAULT_PELT_MIN_SEGMENT_SIZE,
    z_threshold: float = DEFAULT_ZSCORE_THRESHOLD,
    z_window: int | None = None,
    min_anomaly_run: int = 2,
    drift_warn_threshold: float = 0.10,
) -> dict[str, TemporalAuditResult]:
    """Audit MULTIPLE targets in ONE pass over the data.

    Equivalent to calling :func:`audit_target_over_time` once per
    target, but does the costly polars group-by-time aggregation a
    single time and slices out per-target rates. For 9M rows Г— 5
    targets this is ~5Г— faster than the per-target loop because the
    bin-assignment + groupby cost dominates.

    Per-target ``TemporalAuditResult`` is returned by name. The
    timestamp column, granularity, and bin filtering rule are shared;
    change-point detection and segment summaries are independent
    per target.

    Parameters
    ----------
    df : polars.DataFrame or pandas.DataFrame
        Input frame. Polars is strongly preferred for >1M rows
        (zero-copy, native multi-aggregation). Pandas falls back to
        N separate aggregations under the hood (the N=1 case of
        ``_aggregate_by_time_pandas``); the polars path is the one
        that exploits the batch.
    timestamp_col : str
    targets : dict
        Mapping from target NAME (used as the key in the returned
        dict) to a spec, where the spec is one of:

        * a string вЂ” the source column name in ``df``. Target type is
          inferred as ``"binary_classification"`` (the most common
          case). Use the longer form below to override.
        * a tuple ``(column_name, target_type)`` вЂ” explicit
          ``column_name`` and ``target_type``
          (``"binary_classification"`` / ``"regression"``).

    granularity, min_bin_fraction, method, pelt_*, z_*, ...
        See :func:`audit_target_over_time`. Shared across targets
        (one granularity, one bin filter, one detector method per call).

    Returns
    -------
    dict[str, TemporalAuditResult]
        Keyed by the keys of ``targets``. Per-target ``segments`` and
        ``warnings`` reflect that target's own time-series.

    Examples
    --------
    >>> result = audit_targets_over_time(
    ...     df, timestamp_col="job_posted_at",
    ...     targets={
    ...         "hired_above_1": "cl_act_total_hired",
    ...         "spent_amount":  ("amount_spent", "regression"),
    ...     },
    ... )
    >>> result["hired_above_1"].segments
    [...]
    >>> result["spent_amount"].segments
    [...]
    """
    if not targets:
        return {}

    # Normalize target specs.
    target_specs: list[tuple[str, str, str]] = []  # (col, type, alias)
    name_to_alias: dict[str, str] = {}
    for name, spec in targets.items():
        if isinstance(spec, str):
            col, ttype = spec, "binary_classification"
        elif isinstance(spec, tuple) and len(spec) == 2:
            col, ttype = spec[0], spec[1]
        else:
            raise ValueError(
                f"targets[{name!r}] must be str or (col, target_type) tuple; "
                f"got {type(spec).__name__}"
            )
        alias = f"__rate__{name}"
        target_specs.append((col, ttype, alias))
        name_to_alias[name] = alias

    # 1. Pick granularity. Pass (min, max) directly so _pick_granularity
    # can skip the O(N) pd.Series + pd.to_datetime materialisation that
    # is wasted when the function only needs the span -- ~1s saved on
    # 1M-row polars inputs by avoiding the .to_list() Python objectification.
    if granularity == "auto":
        if _HAS_POLARS and isinstance(df, pl.DataFrame):
            _ts_col = df[timestamp_col]
            ts_for_picker = (_ts_col.min(), _ts_col.max())
        else:
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            _ts_col = df[timestamp_col]
            ts_for_picker = (_ts_col.min(), _ts_col.max())
        chosen = _pick_granularity(ts_for_picker)
    else:
        # Caller forced granularity; no need to inspect the timestamps.
        chosen = granularity
        # Still coerce non-polars/non-pandas inputs downstream consumers may not handle.
        if not (_HAS_POLARS and isinstance(df, pl.DataFrame)) and not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

    # 2. ONE aggregation pass вЂ” polars fastpath multi-agg, pandas
    #    fallback runs per-target (less efficient but correct).
    if _HAS_POLARS and isinstance(df, pl.DataFrame):
        agg = _aggregate_by_time_polars_multi(
            df, timestamp_col, target_specs, chosen,  # type: ignore[arg-type]
        )
    else:
        # Pandas fallback: still N passes, but at least one place to
        # change later if we add a multi-agg pandas path.
        agg = _aggregate_by_time_pandas(
            df, timestamp_col, target_specs[0][0], chosen,  # type: ignore[arg-type]
            target_type=target_specs[0][1],
        )
        agg = agg.rename(columns={"target_rate": target_specs[0][2]})
        for col, ttype, alias in target_specs[1:]:
            sub = _aggregate_by_time_pandas(
                df, timestamp_col, col, chosen, target_type=ttype,  # type: ignore[arg-type]
            )
            agg[alias] = sub["target_rate"].values

    # 3. Per-target downstream pipeline.
    results: dict[str, TemporalAuditResult] = {}
    for name, spec in targets.items():
        col, ttype = (spec, "binary_classification") if isinstance(spec, str) else (spec[0], spec[1])
        alias = name_to_alias[name]

        sub = pd.DataFrame({
            "bin_start": agg["bin_start"].values,
            "n_obs": agg["n_obs"].values,
            "target_rate": agg[alias].values,
        })
        results[name] = _audit_from_agg(
            agg=sub,
            target_name=name,
            target_type=ttype,
            timestamp_col=timestamp_col,
            granularity=chosen,  # type: ignore[arg-type]
            min_bin_fraction=min_bin_fraction,
            method=method,
            pelt_model=pelt_model,
            pelt_penalty=pelt_penalty,
            pelt_min_segment_size=pelt_min_segment_size,
            z_threshold=z_threshold,
            z_window=z_window,
            min_anomaly_run=min_anomaly_run,
            drift_warn_threshold=drift_warn_threshold,
        )
    return results


# Wave 106c (2026-05-21): _audit_from_agg moved to sibling.
from ._target_temporal_audit_from_agg import _audit_from_agg  # noqa: F401, E402

# Wave 106b (2026-05-21): plot_target_over_time moved to sibling.
from ._target_temporal_plot import plot_target_over_time  # noqa: F401, E402

def format_temporal_audit_report(result: TemporalAuditResult) -> str:
    """Compact text rendering for log output."""
    lines = [
        f"target_temporal_audit: {result.target_name} ({result.target_type}, "
        f"{result.granularity}-binned, {len(result.bins)} bins, "
        f"{len(result.segments)} segments)",
    ]
    for s in result.segments:
        lines.append(
            f"  segment {s['start_label']}..{s['end_label']} "
            f"({s['n_bins']} bins, n_obs={s['n_obs']:_}): "
            f"mean_rate={s['mean_rate']:.3f}"
        )
    if result.warnings:
        lines.append("")
        for w in result.warnings:
            if w.startswith("  "):
                lines.append(w)
            else:
                lines.append(f"  WARN: {w}")
    if result.actionable.get("recommendation"):
        lines.append("")
        lines.append("  ACTIONABLE: " + result.actionable["recommendation"])
    return "\n".join(lines)
