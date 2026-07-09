"""Builders for the grouped / temporal engineered-recipe kinds.

Thin ``EngineeredRecipe`` constructors for grouped delta / agg / composite-group
agg / grouped quantile / target-aware group bin / lagged diff; the matching
``_apply_*`` replay helpers live in their generator siblings (``_grouped_agg_fe``,
``_ratio_delta_fe``, ...) and are dispatched lazily by the parent. The
``EngineeredRecipe`` dataclass is lazy-imported in-body to avoid a cycle.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import EngineeredRecipe


def build_grouped_delta_recipe(
    *, name: str, group_col: str, num_col: str, op: str,
    lookup_mean: dict, lookup_std: dict,
    global_mean: float, global_std: float,
) -> EngineeredRecipe:
    """Frozen recipe for a grouped-delta column. ``op='minus_mean'`` emits
    ``x - mean(x | group)``; ``op='div_std'`` emits the per-group z-score
    ``(x - mean(x | group)) / std(x | group)``. Both fall back to the
    train global mean / std when a group is unseen at replay."""
    from . import EngineeredRecipe
    if op not in ("minus_mean", "div_std"):
        raise ValueError(f"grouped_delta op must be 'minus_mean' or 'div_std'; got {op!r}")
    lookup_mean_clean = {str(k): float(v) for k, v in lookup_mean.items()}
    lookup_std_clean = {str(k): float(v) for k, v in lookup_std.items()}
    return EngineeredRecipe(
        name=name,
        kind="grouped_delta",
        src_names=(group_col, num_col),
        extra={
            "group_col": str(group_col),
            "num_col": str(num_col),
            "op": str(op),
            "lookup_mean": lookup_mean_clean,
            "lookup_std": lookup_std_clean,
            "global_mean": float(global_mean),
            "global_std": float(global_std),
        },
    )


def build_grouped_agg_recipe(
    *, name: str, group_col: str, num_col: str, stat: str, op: str,
    group_lookup_dict: dict, global_value: float,
    lookup_mean: dict, lookup_std: dict,
    global_mean: float, global_std: float,
) -> EngineeredRecipe:
    """Layer 87 (2026-06-01): frozen recipe for one grouped multi-stat
    aggregate. ``op='broadcast'`` emits the per-group ``stat`` broadcast back
    to rows; ``op='z_within'`` emits ``(x - mean(x|group)) / std(x|group)``;
    ``op='ratio'`` emits ``x / mean(x|group)``. Unseen groups at replay fall
    back to the fit-time global statistic. Replay reads only X (no y), so
    transform() is leakage-free."""
    from . import EngineeredRecipe
    if op not in ("broadcast", "z_within", "ratio"):
        raise ValueError(f"grouped_agg op must be 'broadcast', 'z_within', or 'ratio'; " f"got {op!r}")
    lookup_clean = {str(k): float(v) for k, v in group_lookup_dict.items()}
    lookup_mean_clean = {str(k): float(v) for k, v in lookup_mean.items()}
    lookup_std_clean = {str(k): float(v) for k, v in lookup_std.items()}
    return EngineeredRecipe(
        name=name,
        kind="grouped_agg",
        src_names=(group_col, num_col),
        extra={
            "group_col": str(group_col),
            "num_col": str(num_col),
            "stat": str(stat),
            "op": str(op),
            "group_lookup_dict": lookup_clean,
            "global_value": float(global_value),
            "lookup_mean": lookup_mean_clean,
            "lookup_std": lookup_std_clean,
            "global_mean": float(global_mean),
            "global_std": float(global_std),
        },
    )


def build_composite_group_agg_recipe(
    *, name: str, group_cols, num_col: str, stat: str, op: str,
    group_lookup_dict: dict, global_value: float,
    lookup_mean: dict, lookup_std: dict,
    global_mean: float, global_std: float,
) -> EngineeredRecipe:
    """Layer 93 (2026-06-01): frozen recipe for one COMPOSITE-key grouped
    multi-stat aggregate. ``group_cols`` is the ORDERED tuple of group columns
    (e.g. ``("region", "month")``); the composite key is rebuilt at replay from
    those columns the same way it was at fit. ``op='broadcast'`` emits the
    per-composite-cell ``stat``; ``op='z_within'`` emits ``(x - mean) / std``
    within the composite cell; ``op='ratio'`` emits ``x / mean``. Unseen
    composite cells fall back to the fit-time global statistic. Replay reads
    only X (no y), so transform() is leakage-free."""
    from . import EngineeredRecipe
    if op not in ("broadcast", "z_within", "ratio"):
        raise ValueError(f"composite_group_agg op must be 'broadcast', 'z_within', or " f"'ratio'; got {op!r}")
    group_cols = tuple(str(c) for c in group_cols)
    if len(group_cols) < 2:
        raise ValueError(f"composite_group_agg recipe '{name}' requires >= 2 group_cols; " f"got {group_cols!r}")
    lookup_clean = {str(k): float(v) for k, v in group_lookup_dict.items()}
    lookup_mean_clean = {str(k): float(v) for k, v in lookup_mean.items()}
    lookup_std_clean = {str(k): float(v) for k, v in lookup_std.items()}
    return EngineeredRecipe(
        name=name,
        kind="composite_group_agg",
        src_names=(*tuple(group_cols), str(num_col)),
        extra={
            "group_cols": group_cols,
            "num_col": str(num_col),
            "stat": str(stat),
            "op": str(op),
            "group_lookup_dict": lookup_clean,
            "global_value": float(global_value),
            "lookup_mean": lookup_mean_clean,
            "lookup_std": lookup_std_clean,
            "global_mean": float(global_mean),
            "global_std": float(global_std),
        },
    )


def build_grouped_quantile_recipe(
    *, name: str, group_col: str, num_col: str, op: str,
    group_sorted: dict, global_sorted: list,
    iqr_lookup: dict, p90p10_lookup: dict,
    global_iqr: float, global_p90p10: float,
    quantiles=(),
) -> EngineeredRecipe:
    """Layer 88 (2026-06-01): frozen recipe for one per-group distributional
    feature. ``op='pct_rank'`` emits the empirical-CDF position of x within its
    group (stored per-group sorted value arrays); ``op='iqr'`` / ``op='p90p10'``
    emit the per-group spread broadcast. Unseen groups at replay fall back to
    the pooled global edges. Replay reads only X (no y), so transform() is
    leakage-free."""
    from . import EngineeredRecipe
    if op not in ("pct_rank", "iqr", "p90p10"):
        raise ValueError(f"grouped_quantile op must be 'pct_rank', 'iqr', or 'p90p10'; " f"got {op!r}")
    group_sorted_clean = {str(k): [float(v) for v in vals] for k, vals in group_sorted.items()}
    return EngineeredRecipe(
        name=name,
        kind="grouped_quantile",
        src_names=(group_col, num_col),
        extra={
            "group_col": str(group_col),
            "num_col": str(num_col),
            "op": str(op),
            "group_sorted": group_sorted_clean,
            "global_sorted": [float(v) for v in global_sorted],
            "iqr_lookup": {str(k): float(v) for k, v in iqr_lookup.items()},
            "p90p10_lookup": {str(k): float(v) for k, v in p90p10_lookup.items()},
            "global_iqr": float(global_iqr),
            "global_p90p10": float(global_p90p10),
            "quantiles": [float(q) for q in quantiles],
        },
    )


def build_target_aware_group_bin_recipe(
    *, name: str, group_col: str, num_col: str,
    group_edges: dict, global_edges: list, n_bins: int,
    op: str = "target_aware_bin",
) -> EngineeredRecipe:
    """Layer 88 (2026-06-01): frozen recipe for one target-aware per-group
    supervised bin index. ``group_edges`` holds, per group key, the inner MDLP
    edges (refit on ALL train rows, maximising ``I(bin; y)`` within the group);
    ``global_edges`` is the pooled fallback for unseen groups. Replay maps a
    row's value through ``searchsorted`` on its group's edges -- a pure function
    of X. The leak-safe OOF assignment used at fit for MI scoring is NOT
    persisted, so transform() carries no y reference."""
    from . import EngineeredRecipe

    group_edges_clean = {str(k): [float(v) for v in edges] for k, edges in group_edges.items()}
    return EngineeredRecipe(
        name=name,
        kind="target_aware_group_bin",
        src_names=(group_col, num_col),
        extra={
            "group_col": str(group_col),
            "num_col": str(num_col),
            "group_edges": group_edges_clean,
            "global_edges": [float(v) for v in global_edges],
            "n_bins": int(n_bins),
        },
    )


def build_lagged_diff_recipe(
    *, name: str, time_col: str, value_col: str, period: int,
) -> EngineeredRecipe:
    """Frozen recipe for ``x_t - x_{t-period}`` after sorting by ``time_col``.
    Replay re-sorts the test frame by ``time_col`` and emits the per-row
    difference; the first ``period`` rows of the sorted order get 0."""
    from . import EngineeredRecipe
    if int(period) < 1:
        raise ValueError(f"lagged_diff period must be >= 1; got {period}")
    return EngineeredRecipe(
        name=name,
        kind="lagged_diff",
        src_names=(time_col, value_col),
        extra={
            "time_col": str(time_col),
            "value_col": str(value_col),
            "period": int(period),
        },
    )
