"""Temporal target audit — detect P(y) shifts over time, find change points.

The user's production drift incident (Upwork jobs-hired forward-mode
training) revealed that:

- The target rate isn't constant over time. The historical period was
  positive-only (selection-biased) at ~98%; a 4-month window in
  2021-2022 had unbiased ~40% rate; the recent month (full uid scrape)
  is unbiased ~40% again.
- Without a per-bin time-series view of the target, this regime change
  is invisible — the operator sees the AGGREGATE rate (74%) and
  thinks they have a balanced classification problem when in fact
  they have multiple regimes mixed.

This module ships:

- `audit_target_over_time(...)` — group-by-time + change-point
  detection, returns a structured `TemporalAuditResult`.
- `plot_target_over_time(...)` — matplotlib-based time-series plot
  with change points marked, saves to disk (or returns Figure).
- `_pick_granularity(...)` — auto-pick a time bin width that yields
  30-50 non-empty bins (configurable via env / args).
- `find_change_points_zscore(...)` — generic 1-D change-point
  detector. Could move to `pyutilz` later (useful for trading too —
  the user said "будет полезно и для трейдинга позже").

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
- DEFAULT_MIN_BIN_FRACTION_FOR_FILTER  — bin must have ≥ this many obs
  to be plotted / used; default 0.5 of the median bin size.
- DEFAULT_ZSCORE_THRESHOLD — 3.0 (3 MADs from rolling median)
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


DEFAULT_MIN_BIN_FRACTION_FOR_FILTER: float = 0.5
"""A bin is kept (plotted, used in change-point detection) only if its
n_obs >= this fraction × median bin size. Filters out the tiny tail
bins that would dominate the curve with high-variance noise."""

DEFAULT_ZSCORE_THRESHOLD: float = 3.0
"""Default threshold for the z-score change-point detector. A bin is
flagged if |rate - rolling_median| > threshold × rolling_MAD (modified
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
distributional changes — slower, sensitive to penalty)."""

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
            * ``"most_recent_stable"`` — last segment with n_bins ≥ 3
              (the operator-friendly default — most-recent regime,
              skipping single-bin spikes).
            * ``"largest"`` — segment with the most observations.
            * ``"all_stable"`` — all segments with n_bins ≥ 3 (omits
              short transient regimes).
            * ``"first"`` / ``"last"`` — first / last segment by index.

        Returns
        -------
        ndarray of bool
            Same length as ``timestamps``. True for rows whose
            timestamp lies in [segment.start_bin, segment.end_bin)
            (the bin's exclusive end is the next bin's start, so
            inclusion is half-open as Pelt segments are).
        """
        if not self.segments or not self.bins:
            ts = pd.to_datetime(pd.Series(timestamps))
            return np.zeros(len(ts), dtype=bool)

        kept = [b for b in self.bins if b.kept]
        if not kept:
            ts = pd.to_datetime(pd.Series(timestamps))
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

        ts = pd.to_datetime(pd.Series(timestamps))
        mask = np.zeros(len(ts), dtype=bool)
        for s in chosen_segs:
            start_idx = int(s["start_idx"])
            end_idx = int(s["end_idx"])  # exclusive into kept[]
            if start_idx >= len(kept):
                continue
            start_ts = kept[start_idx].bin_start
            if end_idx < len(kept):
                end_ts = kept[end_idx].bin_start
            else:
                # Tail segment — use last bin's left edge + granularity
                # span as the (open) right edge.
                end_ts = kept[-1].bin_start + gran_delta
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
    mean of the range (sqrt(30 * 50) ≈ 38).
    """
    if len(timestamps) == 0:
        return "month"
    ts = pd.to_datetime(pd.Series(timestamps))
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
    # offset aliases — e.g. "QS" works as an offset but not as a Period
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
        rate = s.groupby("__bin")[target_col].apply(
            lambda c: (c.fillna(0) > 0).mean()
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

def find_change_points_pelt(
    rates: np.ndarray,
    *,
    model: str = DEFAULT_PELT_MODEL,
    penalty: float | None = None,
    min_segment_size: int = DEFAULT_PELT_MIN_SEGMENT_SIZE,
    weights: np.ndarray | None = None,
) -> list[int]:
    """Find change points via PELT (Killick et al. 2012) using
    ``ruptures.Pelt``.

    PELT is the canonical algorithm for offline change-point detection:
    given a cost function and a constant penalty per change point, it
    returns the partition that minimises ``total_cost +
    penalty × n_changepoints``. With pruning it runs in O(n) practical
    / O(n²) worst-case.

    Penalty auto-tuning
    -------------------
    When ``penalty`` is ``None`` (default), we use a BIC-style
    estimator: ``pen = log(n) × max(var(rates), eps)``. For binary-rate
    series this lands on a sensible default — empirically detected
    all 4 transitions in the user's production drift pattern. Override
    explicitly if the auto-pen leaves you with too many or too few
    points.

    Parameters
    ----------
    rates : ndarray, shape (n,)
        Per-bin target rate.
    model : str
        ruptures cost model. ``"l2"`` (default; mean-shift) is the
        fast and standard choice. Others: ``"l1"`` (median-shift,
        outlier-robust), ``"rbf"`` (kernel, slow but detects general
        distributional changes).
    penalty : float, optional
        Per-change-point penalty (BIC-like). Higher = fewer points.
        ``None`` = auto-tune (see above).
    min_segment_size : int
        Minimum number of bins per segment. Default 2.
    weights : ndarray, optional
        Currently unused by ruptures' built-in costs; we accept the
        kwarg for signature parity with ``find_change_points_zscore``.
        Future: drop tiny-n bins from the input before fitting.

    Returns
    -------
    list of int
        Boundary indices in pairs ``[start_0, end_0_excl, start_1,
        end_1_excl, ...]`` — same convention as
        ``find_change_points_zscore``. Conversion from ruptures' format
        (which returns ``[end_0, end_1, ..., n]`` excluding 0) to our
        anomaly-pair format is handled internally.
    """
    rates = np.asarray(rates, dtype=np.float64)
    n = len(rates)
    if n < min_segment_size * 2:
        return []

    # Optional weight-based bin filter — drop very-small-n bins so they
    # don't bias the segmentation. We keep the full series shape and
    # only re-weight by replacing dropped bins with the running median
    # so they neither add a "segment" nor force a split.
    rates_for_fit = rates.copy()
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        wpos = weights[weights > 0]
        if wpos.size > 0:
            wmedian = float(np.median(wpos))
            tiny = weights < max(1.0, 0.1 * wmedian)
            if tiny.any():
                # Replace tiny-bin rates with their nearest non-tiny neighbour.
                non_tiny_idx = np.where(~tiny)[0]
                if non_tiny_idx.size > 0:
                    for i in np.where(tiny)[0]:
                        nearest = non_tiny_idx[np.argmin(np.abs(non_tiny_idx - i))]
                        rates_for_fit[i] = rates[nearest]

    # Auto-tune penalty: BIC-style. The within-segment residual variance
    # is unknown, so we use overall data variance as an upper bound;
    # times log(n). A floor protects against degenerate constant series
    # (var ≈ 0 → no segmentation).
    if penalty is None:
        var = float(np.var(rates_for_fit))
        penalty = max(0.05, var * math.log(max(n, 2)))

    rpt = _import_ruptures()
    algo = rpt.Pelt(model=model, min_size=min_segment_size, jump=1)
    algo.fit(rates_for_fit.reshape(-1, 1))
    # ruptures returns ascending end-indices ending at n (e.g. [14, 18, 33]
    # for n=33 means segments [0:14], [14:18], [18:33]).
    bkps = algo.predict(pen=float(penalty))

    # Convert to our [start_0, end_0_excl, start_1, end_1_excl, ...]
    # boundary-pairs format. The "anomaly intervals" are the segments
    # whose mean differs from the global median by more than
    # 0.5 × MAD-equivalent. Operators want a flat list of segment
    # boundaries — convert directly.
    if not bkps or len(bkps) <= 1:
        return []
    boundaries: list[int] = []
    prev = 0
    for end in bkps:
        if end > prev:
            boundaries.append(prev)
            boundaries.append(int(end))
            prev = end
    # The very first / last "segments" cover [0:end_0] and [end_{-2}:n]
    # — these are NOT anomaly intervals, they're the regime-segments.
    # For change-point output we want the *transition points*: the
    # indices where regime switches. So remove the [0, end_0] and
    # [last_start, n] outer pairs to get just the inner transitions.
    # Actually our format expects all segment boundaries; let the
    # caller (`_segments_from_change_points`) interpret them. Return as
    # flat list of inner change points (not the trivial 0 / n).
    inner = [b for b in bkps[:-1]]  # drop the trivial final n
    if not inner:
        return []
    # Build pairs [c0, c1, c1, c2, ...] so segments[0]=[0:c0],
    # segments[1]=[c0:c1], etc.
    pairs: list[int] = []
    for c in inner:
        pairs.append(int(c))
        pairs.append(int(c))
    # Drop the duplicate at the end so we have inner change points
    # exactly (each appears twice = both end of prev seg and start of
    # next seg in our pair format). _segments_from_change_points uses
    # `set([0, n] + boundaries)` to collapse duplicates.
    return pairs


def find_change_points_zscore(
    rates: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    z_threshold: float = DEFAULT_ZSCORE_THRESHOLD,
    min_anomaly_run: int = 1,
    window: int | None = None,
    abs_min_spread: float = 0.05,
) -> list[int]:
    """Find indices where ``rates`` departs from a baseline by more
    than ``z_threshold`` modified-z-score units (or by an absolute
    minimum spread when the baseline MAD is degenerate).

    Two modes:

    * **Global-median baseline** (default, ``window=None``). The
      median of ``rates`` is the baseline; bins are flagged if
      ``|rate - median| > z_threshold × MAD`` (with a fall-through
      ``|rate - median| > abs_min_spread`` for the degenerate
      MAD-near-zero case). Right for "dominant baseline + a few
      anomaly regimes" patterns — the user's selection-bias case
      where ~80% of bins sit at one rate and a contiguous regime
      sits at another.

    * **Local rolling-window baseline** (``window=K``). Centred rolling
      median + MAD over a window of width K. Right when no single
      regime dominates (e.g. multiple roughly-equal-sized regimes).
      Window must be wider than the longest anomaly to detect it
      (median of a centred window inside an anomaly returns the
      anomaly's value). Falls back to the global-median branch for
      bins where the local window is degenerate.

    Modified z-score uses median + MAD (Iglewicz & Hoaglin 1993) so
    a single outlier doesn't inflate the baseline.

    Parameters
    ----------
    rates : ndarray, shape (n,)
        Per-bin target rate.
    weights : ndarray, shape (n,), optional
        Per-bin weight (typically n_obs). When provided, bins with
        n_obs below ``0.1 × median(weights)`` are excluded from the
        baseline median (they're too noisy to anchor it).
    z_threshold : float
        Modified-z-score threshold. 3.0 = strong outlier; 2.0 = loose.
    min_anomaly_run : int
        Minimum consecutive flagged bins to count as a change point.
        ``1`` flags single-bin spikes; ``2+`` filters them out.
    window : int, optional
        If given, use centred rolling-window baseline (must be odd; +1
        applied if even). If None (default), use global median.
    abs_min_spread : float
        Absolute baseline-vs-rate spread above which a bin is flagged
        even when the modified z-score is degenerate (MAD ≈ 0). 0.05
        means "5 percentage points off the baseline" for binary
        targets.

    Returns
    -------
    list of int
        Boundary indices in pairs: ``[start_0, end_0_exclusive,
        start_1, end_1_exclusive, ...]``. For one dip in the middle
        of an otherwise-stable series the result is
        ``[dip_start, dip_end + 1]``.
    """
    rates = np.asarray(rates, dtype=np.float64)
    n = len(rates)
    if n == 0:
        return []

    # Bin-validity mask: drop tiny-n bins from baseline.
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        wpos = weights[weights > 0]
        if wpos.size > 0:
            wmedian = float(np.median(wpos))
            valid_for_baseline = weights >= max(1.0, 0.1 * wmedian)
        else:
            valid_for_baseline = np.ones(n, dtype=bool)
    else:
        valid_for_baseline = np.ones(n, dtype=bool)

    if window is None:
        # ---- Global-median baseline ----
        valid_rates = rates[valid_for_baseline]
        if valid_rates.size < 3:
            return []
        med = float(np.median(valid_rates))
        mad = float(np.median(np.abs(valid_rates - med)))
        if mad < 1e-9:
            flagged = np.abs(rates - med) > abs_min_spread
        else:
            mz = 0.6745 * np.abs(rates - med) / mad
            # Combine z-score + absolute-spread floor (catches narrow
            # but real shifts in series with tiny baseline MAD).
            flagged = (mz > z_threshold) | (np.abs(rates - med) > abs_min_spread * 2)
    else:
        # ---- Local rolling-window baseline ----
        if window % 2 == 0:
            window += 1
        half = window // 2
        flagged = np.zeros(n, dtype=bool)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            window_vals = rates[lo:hi][valid_for_baseline[lo:hi]]
            if window_vals.size < 2:
                continue
            med_i = float(np.median(window_vals))
            mad_i = float(np.median(np.abs(window_vals - med_i)))
            if mad_i < 1e-9:
                flagged[i] = abs(rates[i] - med_i) > abs_min_spread
            else:
                mz_i = 0.6745 * abs(rates[i] - med_i) / mad_i
                flagged[i] = (mz_i > z_threshold) or (
                    abs(rates[i] - med_i) > abs_min_spread * 2
                )

    # Group consecutive flags into runs; report run boundaries.
    boundaries: list[int] = []
    in_run = False
    run_start = -1
    for i, f in enumerate(flagged):
        if f and not in_run:
            run_start = i
            in_run = True
        elif not f and in_run:
            run_len = i - run_start
            if run_len >= min_anomaly_run:
                boundaries.append(run_start)
                boundaries.append(i)
            in_run = False
    if in_run:
        run_len = n - run_start
        if run_len >= min_anomaly_run:
            boundaries.append(run_start)
            boundaries.append(n)
    return boundaries


def find_change_points(
    rates: np.ndarray,
    *,
    method: ChangePointMethod = "pelt",
    weights: np.ndarray | None = None,
    **method_kwargs: Any,
) -> list[int]:
    """Top-level dispatcher: ``method="pelt"`` (default) or
    ``method="zscore"``.

    Pelt is the principled choice (auto-detects K via penalty, optimal
    given cost+penalty, handles balanced regimes). Z-score is the
    simpler / more interpretable alternative for "dominant baseline +
    anomaly" patterns. See module docstring for trade-offs.

    All ``method_kwargs`` are forwarded to the underlying detector.
    Returns the same ``[start_0, end_0, start_1, end_1, ...]`` boundary
    list shape regardless of method.
    """
    if method == "pelt":
        return find_change_points_pelt(rates, weights=weights, **method_kwargs)
    if method == "zscore":
        return find_change_points_zscore(rates, weights=weights, **method_kwargs)
    raise ValueError(f"Unknown change-point method: {method!r}")


def _segments_from_change_points(
    rates: np.ndarray,
    weights: np.ndarray,
    boundaries: list[int],
    bin_labels: list[str],
) -> list[dict[str, Any]]:
    """Split [0..n) into intervals by the change-point boundaries.

    Boundaries from `find_change_points_zscore` come in pairs
    (anomaly_start, anomaly_end). We add 0 and n to get all segment
    edges, then dedup-sort.
    """
    n = len(rates)
    edges = sorted(set([0, n] + list(boundaries)))
    segments: list[dict[str, Any]] = []
    for s, e in zip(edges[:-1], edges[1:]):
        if e <= s:
            continue
        seg_rates = rates[s:e]
        seg_weights = weights[s:e]
        total_w = float(seg_weights.sum())
        wmean = (
            float(np.average(seg_rates, weights=seg_weights))
            if total_w > 0 else float(seg_rates.mean()) if seg_rates.size else float("nan")
        )
        segments.append({
            "start_idx": int(s),
            "end_idx": int(e),
            "start_label": bin_labels[s] if s < len(bin_labels) else "",
            "end_label": bin_labels[e - 1] if e - 1 < len(bin_labels) else "",
            "n_bins": int(e - s),
            "n_obs": int(seg_weights.sum()),
            "mean_rate": wmean,
        })
    return segments


# -----------------------------------------------------------------------------
# Top-level audit
# -----------------------------------------------------------------------------

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
    1. Coerce to pandas (lazy — uses polars groupby if df is pl.DataFrame).
    2. Pick granularity (or use the supplied one) to land in [30, 50] bins.
    3. Aggregate ``target_col`` by time bin.
    4. Filter out sparse bins (< `min_bin_fraction` × median(n_obs)).
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
        supported in this audit yet — fall through to mean(y).
    granularity : str
        ``"auto"`` (default) or one of minute/hour/day/week/month/
        quarter/year.
    min_bin_fraction : float
        Bin kept only if n_obs >= this × median(n_obs). 0.5 by default
        — drops the tail bins that have noisy rates.
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

    # 1. Aggregation — prefer polars path when input is polars.
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
    single time and slices out per-target rates. For 9M rows × 5
    targets this is ~5× faster than the per-target loop because the
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

        * a string — the source column name in ``df``. Target type is
          inferred as ``"binary_classification"`` (the most common
          case). Use the longer form below to override.
        * a tuple ``(column_name, target_type)`` — explicit
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

    # 1. Pick granularity.
    if _HAS_POLARS and isinstance(df, pl.DataFrame):
        ts_for_picker = df[timestamp_col].to_list()
    else:
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        ts_for_picker = df[timestamp_col]
    chosen = (
        _pick_granularity(ts_for_picker)
        if granularity == "auto"
        else granularity
    )

    # 2. ONE aggregation pass — polars fastpath multi-agg, pandas
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


def _audit_from_agg(
    *,
    agg: pd.DataFrame,
    target_name: str,
    target_type: str,
    timestamp_col: str,
    granularity: Granularity,
    min_bin_fraction: float,
    method: ChangePointMethod,
    pelt_model: str,
    pelt_penalty: float | None,
    pelt_min_segment_size: int,
    z_threshold: float,
    z_window: int | None,
    min_anomaly_run: int,
    drift_warn_threshold: float,
) -> TemporalAuditResult:
    """Internal: turn a pre-computed (bin_start, n_obs, target_rate)
    aggregation into a full TemporalAuditResult. Shared between
    ``audit_target_over_time`` (single) and ``audit_targets_over_time``
    (batch) so the post-aggregation pipeline lives in one place.
    """
    if agg.empty:
        return TemporalAuditResult(
            target_name=target_name, target_type=target_type,
            timestamp_col=timestamp_col, granularity=granularity,
            bins=[], change_point_indices=[], segments=[],
            warnings=["empty aggregation - no data after time-binning"],
            actionable={},
        )

    median_n = float(agg["n_obs"].median())
    threshold_n = max(1.0, min_bin_fraction * median_n)
    agg = agg.copy()
    agg["kept"] = agg["n_obs"] >= threshold_n

    bins = [
        TimeBin(
            bin_label=_format_bin_label(row.bin_start, granularity),
            bin_start=row.bin_start,
            n_obs=int(row.n_obs),
            target_rate=float(row.target_rate),
            kept=bool(row.kept),
        )
        for row in agg.itertuples(index=False)
    ]

    kept_bins = [b for b in bins if b.kept]
    if len(kept_bins) < 3:
        return TemporalAuditResult(
            target_name=target_name, target_type=target_type,
            timestamp_col=timestamp_col, granularity=granularity,
            bins=bins, change_point_indices=[], segments=[],
            warnings=[
                f"only {len(kept_bins)} non-sparse bins after the {min_bin_fraction}× median-n_obs filter "
                f"- too few for a temporal audit. Consider a finer granularity or a longer time span.",
            ],
            actionable={},
        )

    rates = np.array([b.target_rate for b in kept_bins])
    weights = np.array([b.n_obs for b in kept_bins], dtype=float)
    labels = [b.bin_label for b in kept_bins]
    if method == "pelt":
        boundaries = find_change_points_pelt(
            rates, weights=weights,
            model=pelt_model, penalty=pelt_penalty,
            min_segment_size=pelt_min_segment_size,
        )
    else:
        boundaries = find_change_points_zscore(
            rates, weights=weights,
            window=z_window,
            z_threshold=z_threshold, min_anomaly_run=min_anomaly_run,
        )
    segments = _segments_from_change_points(rates, weights, boundaries, labels)

    warnings: list[str] = []
    if len(segments) >= 2:
        mean_rates = [s["mean_rate"] for s in segments if s["mean_rate"] == s["mean_rate"]]
        if mean_rates and max(mean_rates) - min(mean_rates) > drift_warn_threshold:
            warnings.append(
                f"target rate is NOT stable over time: detected {len(segments)} segments "
                f"with mean rates ranging {min(mean_rates):.3f}..{max(mean_rates):.3f} "
                f"(spread {(max(mean_rates) - min(mean_rates)):.3f}, threshold {drift_warn_threshold:.2f}). "
                f"Likely causes: (a) selection-bias in your data source over time, "
                f"(b) regime change in the underlying generative process, (c) target "
                f"definition shift. See segment list below for cutoff dates."
            )
            for s in segments:
                warnings.append(
                    f"  segment {s['start_label']}..{s['end_label']} "
                    f"({s['n_bins']} bins, n_obs={s['n_obs']:_}): "
                    f"mean_rate={s['mean_rate']:.3f}"
                )

    n_dropped = sum(1 for b in bins if not b.kept)
    if n_dropped > 0:
        warnings.append(
            f"{n_dropped} bin(s) dropped from the audit (n_obs < "
            f"{int(threshold_n):_} = {min_bin_fraction}× median bin size). "
            "Typically the partial first / last bins of your time range; "
            "if this number is large, consider a wider granularity."
        )

    most_recent_stable: dict[str, Any] | None = None
    if segments:
        for s in reversed(segments):
            if s["n_bins"] >= 3:
                most_recent_stable = s
                break

    actionable: dict[str, Any] = {
        "n_segments": len(segments),
        "most_recent_stable_segment": most_recent_stable,
    }
    if most_recent_stable is not None and len(segments) >= 2:
        actionable["recommendation"] = (
            f"Consider restricting training to the most-recent stable segment "
            f"({most_recent_stable['start_label']}..{most_recent_stable['end_label']}, "
            f"n_obs={most_recent_stable['n_obs']:_}, mean_rate={most_recent_stable['mean_rate']:.3f}) "
            f"or pair the suite with PULearningWrapper if earlier segments are "
            f"selection-biased rather than wrong."
        )

    return TemporalAuditResult(
        target_name=target_name, target_type=target_type,
        timestamp_col=timestamp_col, granularity=granularity,
        bins=bins, change_point_indices=boundaries, segments=segments,
        warnings=warnings, actionable=actionable,
    )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_target_over_time(
    result: TemporalAuditResult,
    *,
    save_path: str | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (12, 4.5),
    plot_outputs: str | None = None,
    base_path: str | None = None,
) -> Any | None:
    """Render the time-series plot of target rate over bins.

    Mirrors the user's polars-rendered chart (year-month index, line
    plot of target rate). Adds:
    - dropped sparse bins shown as faded markers
    - vertical lines at change-point boundaries
    - per-segment mean as a horizontal step

    Parameters
    ----------
    result : TemporalAuditResult
    save_path : str, optional
        If provided, save the figure here and return the path (closes
        the figure to free memory).
    show : bool
        Call ``plt.show()`` instead of saving.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure or None
        Returns the Figure when neither save_path nor show is set.
    """
    # Opt-in DSL render path. When ``plot_outputs`` +
    # ``base_path`` are supplied, route through the spec pipeline
    # (matplotlib + plotly via the same DSL). Default behaviour
    # preserved -- legacy callers see no change.
    if plot_outputs and base_path:
        from mlframe.reporting.charts.temporal import build_temporal_audit_spec
        from mlframe.reporting.output import parse_plot_output_dsl
        from mlframe.reporting.renderers import render_and_save
        spec = build_temporal_audit_spec(result, figsize=figsize)
        render_and_save(spec, parse_plot_output_dsl(plot_outputs), base_path)
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping target-over-time plot.")
        return None

    if not result.bins:
        return None

    kept = [b for b in result.bins if b.kept]
    dropped = [b for b in result.bins if not b.kept]

    fig, ax = plt.subplots(figsize=figsize)
    if kept:
        ax.plot(
            [b.bin_start for b in kept],
            [b.target_rate for b in kept],
            marker="o", linestyle="-", linewidth=1.2, markersize=4,
            label=result.target_name,
        )
    if dropped:
        ax.plot(
            [b.bin_start for b in dropped],
            [b.target_rate for b in dropped],
            marker="x", linestyle=":", linewidth=0.6, markersize=4,
            color="gray", alpha=0.4,
            label=f"sparse (filtered, n={len(dropped)})",
        )

    # Change-point lines + per-segment mean.
    for s in result.segments:
        if s["start_idx"] > 0 and s["start_idx"] < len(kept):
            ax.axvline(
                kept[s["start_idx"]].bin_start,
                color="red", linestyle="--", linewidth=0.8, alpha=0.4,
            )
        if s["start_idx"] < len(kept) and s["end_idx"] - 1 < len(kept):
            x_start = kept[s["start_idx"]].bin_start
            x_end = kept[min(s["end_idx"] - 1, len(kept) - 1)].bin_start
            ax.hlines(
                s["mean_rate"], x_start, x_end,
                color="orange", linestyle="-", linewidth=2.0, alpha=0.6,
            )

    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(f"{result.timestamp_col} ({result.granularity})")
    ax.set_ylabel("target rate")
    ax.set_title(
        f"target_temporal_audit: {result.target_name} "
        f"({result.granularity}-binned, {len(result.segments)} segments)"
    )
    ax.legend(loc="best", framealpha=0.7)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        result.plot_path = save_path
        return None
    if show:
        plt.show()
        return None
    return fig


# -----------------------------------------------------------------------------
# Human-readable text report
# -----------------------------------------------------------------------------

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
