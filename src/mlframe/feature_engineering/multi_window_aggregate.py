"""Multi-fixed-lookback-horizon aggregation: one aggregate feature set per horizon, in a single call.

Computing a single all-history aggregate per entity discards a real signal: how RECENT the driving events
were. A 9th-place Home-Credit team's fix was to separately aggregate at several fixed lookback horizons
("last 3 months", "last 6 months", "last year", ...) rather than one all-history number -- recent-vs-older
behavior divergence is itself informative (a worsening trend, a recent spike). This helper generalizes that
pattern to an arbitrary horizon list in one call, complementing the existing leakage-safe as-of aggregate.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from mlframe.feature_engineering.as_of_aggregate import leakage_safe_aggregate


def multi_window_aggregate(
    history_df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    as_of: pd.DataFrame,
    agg_funcs: Dict[str, Sequence[str]],
    lookback_horizons: Sequence[float],
    query_entity_col: str = "as_of",
    auto_select: bool = False,
    target: Optional[Union[pd.Series, np.ndarray]] = None,
    cv: int = 5,
    scoring: str = "roc_auc",
    min_lift: float = 0.005,
    estimator: Optional[Any] = None,
    return_selection_info: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """Aggregate ``history_df`` per entity at several fixed lookback horizons before each query's cutoff.

    Parameters
    ----------
    history_df, entity_col, time_col, agg_funcs, query_entity_col
        Same contract as :func:`mlframe.feature_engineering.as_of_aggregate.leakage_safe_aggregate`.
    as_of
        Query frame with the entity key and a per-row cutoff column (name given by ``query_entity_col``).
    lookback_horizons
        Window lengths (same units as ``time_col``), e.g. ``[90, 180, 270, 365]`` for "last 3/6/9/12 months"
        in day units. Each horizon ``h`` aggregates only rows with ``cutoff - h <= time_col < cutoff``
        (strictly before the cutoff, same leakage-safety contract as the full-history version).
    auto_select, target, cv, scoring, min_lift, estimator, return_selection_info
        Opt-in automatic horizon-selection mode (all default to caller-hand-picked-horizons behavior when
        ``auto_select`` is left ``False`` -- the rest of these params are then ignored). When ``auto_select``
        is ``True``, ``target`` (the downstream label/regression target, one value per ``as_of`` row) is
        required: candidate horizons are greedily forward-selected by cross-validated ``scoring`` lift of
        ``estimator`` (default ``LogisticRegression``) on top of the horizons already kept, in the order
        given by ``lookback_horizons``; a horizon is kept only if it improves the CV score by more than
        ``min_lift`` over the best score achievable without it. This directly answers "does this lookback
        window carry incremental signal beyond the others", rather than requiring the caller to guess which
        of a candidate grid of horizons are worth computing downstream. Only the columns of kept horizons are
        returned. Pass ``return_selection_info=True`` to also get back a dict with the per-horizon CV lift
        and the kept/dropped horizon lists (evaluation detail, not part of the default return contract).
    """
    if not lookback_horizons:
        raise ValueError("multi_window_aggregate: lookback_horizons must be non-empty")
    if auto_select and target is None:
        raise ValueError("multi_window_aggregate: auto_select=True requires a non-None target")

    query = as_of.reset_index(drop=True)
    out = pd.DataFrame({entity_col: query[entity_col].to_numpy()})

    for horizon in lookback_horizons:
        windowed_history = history_df.copy()
        # tag each history row with the horizon-specific "window start" so a per-horizon leakage_safe_aggregate
        # call only sees rows within [cutoff - horizon, cutoff) -- reuse the vetted as-of aggregation machinery
        # rather than duplicating its cumsum/searchsorted logic.
        merged_query = query[[entity_col, query_entity_col]].copy()
        merged_query["_window_start"] = merged_query[query_entity_col] - horizon

        # filter history to rows that could POSSIBLY fall in ANY query row's window for this horizon is not
        # correct per-row (different entities/cutoffs), so instead run leakage_safe_aggregate against the
        # window-shifted cutoff, then separately re-run with the window START as an exclusion floor by
        # dropping history rows before window start via a per-entity merge-asof-style filter is unnecessary:
        # leakage_safe_aggregate already computes cumulative sums up to the searchsorted cutoff position; the
        # window-start floor is enforced by subtracting the cumulative aggregate AT the window start.
        upper = leakage_safe_aggregate(
            windowed_history, entity_col=entity_col, time_col=time_col,
            as_of=query[[entity_col, query_entity_col]], agg_funcs=agg_funcs, query_entity_col=query_entity_col,
        )
        lower_query = query[[entity_col]].copy()
        lower_query[query_entity_col] = merged_query["_window_start"].to_numpy()
        lower = leakage_safe_aggregate(
            windowed_history, entity_col=entity_col, time_col=time_col,
            as_of=lower_query, agg_funcs=agg_funcs, query_entity_col=query_entity_col,
        )

        for col, fns in agg_funcs.items():
            for fn in fns:
                colname = f"{col}_{fn}"
                windowed_name = f"{colname}_last_{horizon}"
                if fn in ("sum", "count"):
                    out[windowed_name] = (upper[colname].fillna(0.0) - lower[colname].fillna(0.0)).to_numpy()
                elif fn == "mean":
                    upper_sum = upper.get(f"{col}_sum")
                    upper_count = upper.get(f"{col}_count")
                    if upper_sum is None or upper_count is None:
                        raise ValueError(f"multi_window_aggregate: computing windowed 'mean' for {col!r} requires 'sum' and 'count' also in agg_funcs[{col!r}]")
                    lower_sum = lower.get(f"{col}_sum")
                    lower_count = lower.get(f"{col}_count")
                    win_sum = upper_sum.fillna(0.0) - lower_sum.fillna(0.0)
                    win_count = upper_count.fillna(0.0) - lower_count.fillna(0.0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        out[windowed_name] = np.where(win_count > 0, win_sum / win_count, np.nan)
                else:
                    # non-additive aggs (min/max/median/...) can't be derived by subtracting two cumulative
                    # snapshots; fall back to the direct windowed computation for just this (col, fn, horizon).
                    out[windowed_name] = _direct_window_agg(history_df, entity_col, time_col, query, query_entity_col, horizon, col, fn)

    if not auto_select:
        return out

    horizon_columns: Dict[float, List[str]] = {horizon: [] for horizon in lookback_horizons}
    for horizon in lookback_horizons:
        suffix = f"_last_{horizon}"
        horizon_columns[horizon] = [c for c in out.columns if c != entity_col and c.endswith(suffix)]

    kept_horizons, lifts = _select_predictive_horizons(
        out, lookback_horizons, horizon_columns, target=target, cv=cv, scoring=scoring, min_lift=min_lift, estimator=estimator
    )
    kept_cols = [entity_col] + [c for horizon in kept_horizons for c in horizon_columns[horizon]]
    selected_out = out[kept_cols]

    if return_selection_info:
        dropped_horizons = [h for h in lookback_horizons if h not in kept_horizons]
        info = {"kept_horizons": kept_horizons, "dropped_horizons": dropped_horizons, "lift_by_horizon": lifts}
        return selected_out, info
    return selected_out


def _select_predictive_horizons(
    out: pd.DataFrame,
    lookback_horizons: Sequence[float],
    horizon_columns: Dict[float, List[str]],
    target: Optional[Union[pd.Series, np.ndarray]],
    cv: int,
    scoring: str,
    min_lift: float,
    estimator: Optional[Any],
) -> Tuple[List[float], Dict[float, float]]:
    """Greedy forward-selection of horizons by cross-validated incremental score lift.

    Each horizon is scored against the *already-kept* feature set rather than in isolation, so a horizon that
    duplicates signal already captured by a previously-kept horizon (e.g. two highly-overlapping windows) is
    correctly dropped as non-incremental, while a horizon carrying genuinely new signal is kept even if its
    standalone score is unremarkable.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    model = estimator if estimator is not None else LogisticRegression(max_iter=1000)
    y = np.asarray(target)

    def _score(cols: List[str]) -> float:
        """Cross-validate model against target using only cols, falling back to a no-feature baseline when empty."""
        if not cols:
            # no-feature baseline: a constant/majority predictor, the real floor a horizon must beat.
            X_dummy = np.zeros((len(y), 1))
            return float(np.mean(cross_val_score(DummyClassifier(strategy="prior"), X_dummy, y, cv=cv, scoring=scoring)))
        X = out[cols].fillna(0.0)
        return float(np.mean(cross_val_score(model, X, y, cv=cv, scoring=scoring)))

    kept: List[str] = []
    kept_horizons: List[float] = []
    lifts: Dict[float, float] = {}
    baseline_score = _score([])
    for horizon in lookback_horizons:
        candidate_cols = kept + horizon_columns[horizon]
        candidate_score = _score(candidate_cols)
        lift = candidate_score - baseline_score
        lifts[horizon] = lift
        if lift > min_lift:
            kept.extend(horizon_columns[horizon])
            kept_horizons.append(horizon)
            baseline_score = candidate_score
    return kept_horizons, lifts


def _direct_window_agg(
    history_df: pd.DataFrame, entity_col: str, time_col: str, query: pd.DataFrame, query_entity_col: str, horizon: float, col: str, fn: str
) -> np.ndarray:
    """Directly aggregate col over each query row's trailing horizon window from its entity's history, without caching intermediate windows."""
    history_groups = {entity: grp for entity, grp in history_df.groupby(entity_col, sort=False)}
    out = np.full(len(query), np.nan)
    for entity, entity_queries in query.groupby(entity_col, sort=False):
        entity_history = history_groups.get(entity)
        if entity_history is None or entity_history.empty:
            continue
        times = entity_history[time_col].to_numpy()
        order = np.argsort(times)
        sorted_times = times[order]
        sorted_col = entity_history[col].to_numpy()[order]
        for idx, cutoff in zip(entity_queries.index, entity_queries[query_entity_col]):
            lo = np.searchsorted(sorted_times, cutoff - horizon, side="left")
            hi = np.searchsorted(sorted_times, cutoff, side="left")
            if hi > lo:
                out[idx] = getattr(pd.Series(sorted_col[lo:hi]), fn)()
    return out


__all__ = ["multi_window_aggregate"]
