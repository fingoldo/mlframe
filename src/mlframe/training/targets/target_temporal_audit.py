"""Temporal target audit — detect P(y) shifts over time, find change points.

A real-world drift incident (a job-board scraping pipeline running in
forward mode) revealed that:

- The target rate isn't constant over time. The historical period was
  positive-only (selection-biased) at ~98%; a several-month window
  had an unbiased ~40% rate; the most recent month is unbiased ~40% again.
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
from typing import Any, Literal

import pandas as pd

try:
    import polars as pl

    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False

logger = logging.getLogger(__name__)

from ._target_temporal_audit_coerce import (
    _AUDIT_DATETIME_HIGH_NS,  # noqa: F401
    _AUDIT_DATETIME_LOW_NS,  # noqa: F401
    _AUDIT_UNIT_NS_FACTOR,  # noqa: F401
    _import_ruptures,  # noqa: F401
)

DEFAULT_MIN_BIN_FRACTION_FOR_FILTER: float = 0.5
"""A bin is kept (plotted, used in change-point detection) only if its
n_obs >= this fraction × median bin size. Filters out the tiny tail
bins that would dominate the curve with high-variance noise."""

from ._target_temporal_types import (  # noqa: F401 - the records and defaults live in the leaf module; re-exported here
    DEFAULT_ZSCORE_THRESHOLD,
    DEFAULT_ZSCORE_WINDOW,
    DEFAULT_PELT_MODEL,
    DEFAULT_PELT_MIN_SEGMENT_SIZE,
    TimeBin,
    TemporalAuditResult,
)
from ._target_temporal_audit_coerce import coerce_timestamps_for_audit  # noqa: F401 - re-exported: callers import it from this module
from ._target_temporal_audit_aggregate import Granularity, _GRANULARITY_SECONDS  # noqa: F401 - re-exported: callers import it from this module

ChangePointMethod = Literal["pelt", "zscore"]

from ._target_temporal_audit_aggregate import (
    DEFAULT_TARGET_BINS_RANGE,  # noqa: F401
    _GRANULARITY_ORDER,  # noqa: F401
    _POLARS_BIN_TRUNCATE,  # noqa: F401
    _aggregate_by_time_pandas,
    _aggregate_by_time_polars,
    _aggregate_by_time_polars_multi,
    _format_bin_label,  # noqa: F401
    _pick_granularity,
    _polars_rate_expr,  # noqa: F401
)


from ._target_temporal_changepoint import (  # noqa: F401
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
        chosen = _pick_granularity(ts_for_picker) if granularity == "auto" else granularity
        agg = _aggregate_by_time_polars(
            df,
            timestamp_col,
            target_col,
            chosen,
            target_type=target_type,  # type: ignore[arg-type]  # granularity is a validated free-form str; the callee narrows to a Literal set at runtime
        )
    else:
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        chosen = _pick_granularity(df[timestamp_col]) if granularity == "auto" else granularity
        agg = _aggregate_by_time_pandas(
            df,
            timestamp_col,
            target_col,
            chosen,
            target_type=target_type,  # type: ignore[arg-type]  # granularity is a validated free-form str; the callee narrows to a Literal set at runtime
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
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 4000
    >>> shifted = np.arange(n) >= n // 2
    >>> df = pd.DataFrame({
    ...     "job_posted_at": pd.date_range("2020-01-01", periods=n, freq="6h"),
    ...     "cl_act_total_hired": (rng.random(n) < np.where(shifted, 0.5, 0.1)).astype(int),
    ...     "amount_spent": rng.normal(size=n) + np.where(shifted, 5.0, 0.0),
    ... })
    >>> result = audit_targets_over_time(
    ...     df, timestamp_col="job_posted_at",
    ...     targets={
    ...         "hired_above_1": "cl_act_total_hired",
    ...         "spent_amount":  ("amount_spent", "regression"),
    ...     },
    ... )
    >>> [len(result[k].segments) for k in ("hired_above_1", "spent_amount")]
    [2, 2]
    >>> result["hired_above_1"].segments[1]["start_label"]
    '2021-05'
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
            raise ValueError(f"targets[{name!r}] must be str or (col, target_type) tuple; " f"got {type(spec).__name__}")
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
        chosen = granularity  # type: ignore[assignment]  # granularity is a validated free-form str matching the Literal set at runtime
        # Still coerce non-polars/non-pandas inputs downstream consumers may not handle.
        if not (_HAS_POLARS and isinstance(df, pl.DataFrame)) and not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

    # 2. ONE aggregation pass — polars fastpath multi-agg, pandas
    #    fallback runs per-target (less efficient but correct).
    if _HAS_POLARS and isinstance(df, pl.DataFrame):
        agg = _aggregate_by_time_polars_multi(
            df,
            timestamp_col,
            target_specs,
            chosen,
        )
    else:
        # Pandas fallback: still N passes, but at least one place to
        # change later if we add a multi-agg pandas path.
        agg = _aggregate_by_time_pandas(
            df,
            timestamp_col,
            target_specs[0][0],
            chosen,
            target_type=target_specs[0][1],
        )
        agg = agg.rename(columns={"target_rate": target_specs[0][2]})
        for col, ttype, alias in target_specs[1:]:
            sub = _aggregate_by_time_pandas(
                df,
                timestamp_col,
                col,
                chosen,
                target_type=ttype,
            )
            agg[alias] = sub["target_rate"].values

    # 3. Per-target downstream pipeline.
    results: dict[str, TemporalAuditResult] = {}
    for name, spec in targets.items():
        col, ttype = (spec, "binary_classification") if isinstance(spec, str) else (spec[0], spec[1])
        alias = name_to_alias[name]

        sub = pd.DataFrame(
            {
                "bin_start": agg["bin_start"].values,
                "n_obs": agg[f"n_obs__{alias}"].values if f"n_obs__{alias}" in agg.columns else agg["n_obs"].values,
                "target_rate": agg[alias].values,
            }
        )
        results[name] = _audit_from_agg(
            agg=sub,
            target_name=name,
            target_type=ttype,
            timestamp_col=timestamp_col,
            granularity=chosen,
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
from ._target_temporal_audit_from_agg import _audit_from_agg

# Wave 106b (2026-05-21): plot_target_over_time moved to sibling.
from ._target_temporal_plot import plot_target_over_time  # noqa: F401


def format_temporal_audit_report(result: TemporalAuditResult) -> str:
    """Compact text rendering for log output."""
    lines = [
        (
            f"target_temporal_audit: {result.target_name} ({result.target_type}, "
            f"{result.granularity}-binned, {len(result.bins)} bins, "
            f"{len(result.segments)} segments)"
        ),
    ]
    lines.extend(
        f"  segment {s['start_label']}..{s['end_label']} " f"({s['n_bins']} bins, n_obs={s['n_obs']:_}): " f"mean_rate={s['mean_rate']:.3f}"
        for s in result.segments
    )
    if result.warnings:
        lines.append("")
        _listed = {line.strip() for line in lines}
        for w in result.warnings:
            if w.startswith("  "):
                # Indented continuation lines of a warning repeat the per-segment lines already printed above; a
                # production log showed every segment twice.
                if w.strip() not in _listed:
                    lines.append(w)
            else:
                lines.append(f"  WARN: {w}")
    if result.actionable.get("recommendation"):
        lines.append("")
        lines.append("  ACTIONABLE: " + result.actionable["recommendation"])
    return "\n".join(lines)
