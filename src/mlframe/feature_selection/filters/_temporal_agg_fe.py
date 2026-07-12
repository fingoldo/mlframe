"""Temporal leak-safe grouped aggregations (Layer 92, 2026-06-01).

Layer 87 ships whole-fold grouped aggregations: ``group_mean(value | entity)``
computed over EVERY train row. For time-series / transaction data this peeks at
the future -- the per-entity mean used to score an early row already contains
that entity's LATER rows. The model trains on a statistic it can never compute
at inference time (you don't know a user's future spend when scoring today),
so the train-CV score is inflated and the forward holdout collapses. Every
Kaggle time-series winner instead uses expanding / rolling-window aggregations
keyed on a time column, which by construction only see the past.

This module provides three leak-safe temporal FE families:

* ``generate_expanding_agg_features`` -- sort by ``time_col`` within entity,
  compute the EXPANDING stat over rows strictly BEFORE the current one
  (``expanding(mean/std/count/min/max).shift(1)``). Row 0 of each entity has no
  history and falls back to the global prior.
* ``generate_rolling_window_agg_features`` -- rolling TIME-window stat (pandas
  rolling on a datetime index), shifted by one row to exclude the current.
* ``generate_lag_features`` -- the entity's value at ``t - lag`` (autoregressive
  signal); unseen / out-of-history positions fall back to the global prior.

Recipe-based replay (CRITICAL leak-safety contract)
----------------------------------------------------

At fit time each recipe stores the per-entity SORTED history snapshot
(timestamps + values) observed on the TRAIN fold. At transform time a test row
computes its expanding / rolling / lag statistic against the TRAIN history only
(all train rows whose timestamp is strictly earlier than the test row), then,
within the test frame, against earlier test rows of the same entity. This means
``transform(X_test)``:

* never peeks at a test row's own future (leak-free within the test set), and
* never leaks train labels (the history carries values, never ``y``).

The ``y`` array is consumed ONLY by the MI gate in :func:`hybrid_temporal_agg_fe`
to rank candidates; the recipes carry no ``y`` reference.

MEMORY: this family is OPT-IN (``fe_temporal_agg_enable=False`` by default) precisely because the leak-safety contract
requires each recipe to persist the FULL per-entity train history (timestamps + values) -- O(n_train) per value column,
carried on the recipe and therefore in any pickled fitted state. There is no cheaper representation: replaying a test row's
expanding / rolling / lag statistic against "the train past" needs the actual past values. Enable it only for genuine
time-series targets, and prefer rolling windows / a bounded lag set over unbounded expanding history on very large frames.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numba
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "EXPANDING_STATS",
    "engineered_name_expanding",
    "engineered_name_rolling",
    "engineered_name_lag",
    "generate_expanding_agg_features",
    "generate_rolling_window_agg_features",
    "generate_lag_features",
    "build_temporal_expanding_recipe",
    "build_temporal_rolling_recipe",
    "build_temporal_lag_recipe",
    "apply_temporal_expanding",
    "apply_temporal_rolling",
    "apply_temporal_lag",
    "hybrid_temporal_agg_fe",
]

# Rolling time-window family lives in its own sibling module (kept the parent under the 1k LOC
# ceiling); re-exported here so existing callers of ``mlframe.feature_selection.filters._temporal_agg_fe``
# see no change.
from ._temporal_agg_fe_rolling import (
    apply_temporal_rolling,
    build_temporal_rolling_recipe,
    engineered_name_rolling,
    generate_rolling_window_agg_features,
    _numeric_window,  # noqa: F401 -- re-exported for tests exercising the rolling njit kernel by its historical module path
    _rolling_stat_past_only,  # noqa: F401 -- re-exported for tests exercising the rolling njit kernel by its historical module path
)

EXPANDING_STATS = ("mean", "std", "count", "min", "max")


def engineered_name_expanding(value_col: str, entity_col: str, stat: str) -> str:
    """Canonical feature name for an expanding-stat feature, e.g. ``texp_mean(amount|user_id)``."""
    return f"texp_{stat}({value_col}|{entity_col})"


def engineered_name_lag(value_col: str, entity_col: str, lag: int) -> str:
    """Canonical feature name for a lag feature, e.g. ``tlag3(amount|user_id)``."""
    return f"tlag{int(lag)}({value_col}|{entity_col})"


# ---------------------------------------------------------------------------
# Helpers shared by the three families
# ---------------------------------------------------------------------------


def _entity_key_series(X: pd.DataFrame, entity_cols: Sequence[str]) -> pd.Series:
    """Collapse one or more entity columns into a single str key per row,
    index-aligned with X. Multi-column keys join with a NUL separator that
    cannot appear in normal string casts."""
    # Route each entity column through the canonical group token (int<->float drift safe): a bare ``.astype(str)``
    # makes fit-int ``1`` -> "1" and predict-float ``1.0`` -> "1.0" DIFFERENT keys, so an entity id that arrives as
    # int64 at fit and float64 at inference (a NaN elsewhere promoting the column, a Parquet round-trip) misses every
    # ``history`` entry and silently routes every test row to the global prior -- the temporal feature becomes a dead
    # constant at serving. group_key_strings collapses integral int/float to the same token (per-unique, still fast).
    from ._internals import group_key_strings

    if len(entity_cols) == 1:
        return pd.Series(group_key_strings(X[entity_cols[0]]), index=X.index)
    parts = [pd.Series(group_key_strings(X[c]), index=X.index) for c in entity_cols]
    key = parts[0]
    for p in parts[1:]:
        key = key.str.cat(p, sep="\x00")
    return key


def _global_prior(values: np.ndarray, stat: str) -> float:
    """Fallback statistic used for rows with no per-entity history (row 0 of an entity, or out-of-history lag positions): the same ``stat`` computed globally over all finite ``values``."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    if stat == "mean":
        return float(np.mean(finite))
    if stat == "std":
        return float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    if stat == "count":
        return 0.0
    if stat == "min":
        return float(np.min(finite))
    if stat == "max":
        return float(np.max(finite))
    if stat == "median":
        return float(np.median(finite))
    raise ValueError(f"temporal_agg: unknown stat {stat!r}")


def _validate(X, entity_cols, value_cols, time_col, fn_name):
    """Type/column-presence gate shared by the three feature generators: X must be a DataFrame, ``time_col`` must exist; ``entity_cols``/``value_cols`` are silently filtered to existing (and, for values, numeric) columns rather than raising."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"{fn_name}: X must be a pandas DataFrame; got {type(X).__name__}")
    entity_cols = [c for c in (entity_cols or []) if c in X.columns]
    value_cols = [c for c in (value_cols or []) if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    if time_col is None or time_col not in X.columns:
        raise KeyError(f"{fn_name}: time_col {time_col!r} not present in X.columns")
    return entity_cols, value_cols


def _stable_time_order(time_vals: np.ndarray) -> np.ndarray:
    """Mergesort (stable) argsort over a time column. Stability guarantees a
    deterministic order for ties so fit / transform agree row-for-row."""
    return np.argsort(time_vals, kind="mergesort")


def _group_row_slices(codes_sorted: np.ndarray, n_codes: int) -> list[np.ndarray]:
    """Return, for each dense group code 0..n_codes-1, the array of row indices
    (into the time-sorted arrays) belonging to that group, in time order.

    Replaces the O(N * cardinality) ``codes_sorted == g_code`` boolean-mask
    rescan (one full-array scan per group) with a single stable argsort + split
    on group boundaries: O(N log N) total. A STABLE sort over the group codes
    preserves the existing within-group time order (rows were sorted by time
    before factorize), so the per-group row order is bit-identical to the mask
    path's ``codes_sorted == g`` selection."""
    if codes_sorted.size == 0 or n_codes == 0:
        return []
    perm = np.argsort(codes_sorted, kind="stable")
    counts = np.bincount(codes_sorted, minlength=n_codes)
    bounds = np.cumsum(counts)
    out: list[np.ndarray] = []
    start = 0
    for g in range(n_codes):
        end = int(bounds[g])
        out.append(perm[start:end])
        start = end
    return out


# ---------------------------------------------------------------------------
# Expanding aggregations
# ---------------------------------------------------------------------------


_EXPANDING_STAT_CODE = {"count": 0, "mean": 1, "std": 2, "min": 3, "max": 4}


@numba.njit(cache=True)
def _expanding_stat_past_only_njit(sorted_vals, group_codes, stat_code, n_groups):
    """Sequential per-entity expanding stat over the strict past. Dense int-code accumulator ARRAYS (indexed by the
    per-entity code) replace the Python dicts of the reference loop -- same running arithmetic, bit-identical incl NaN.
    ``stat_code``: 0=count 1=mean 2=std 3=min 4=max. Not prange (running accumulator is order-dependent)."""
    n = sorted_vals.size
    out = np.full(n, np.nan, dtype=np.float64)
    n_seen = np.zeros(n_groups, dtype=np.int64)
    run_sum = np.zeros(n_groups, dtype=np.float64)
    run_sumsq = np.zeros(n_groups, dtype=np.float64)
    run_min = np.full(n_groups, np.inf, dtype=np.float64)
    run_max = np.full(n_groups, -np.inf, dtype=np.float64)
    for i in range(n):
        g = group_codes[i]
        cnt = n_seen[g]
        if cnt > 0:
            if stat_code == 0:
                out[i] = float(cnt)
            elif stat_code == 1:
                out[i] = run_sum[g] / cnt
            elif stat_code == 2:
                if cnt > 1:
                    mean = run_sum[g] / cnt
                    var = (run_sumsq[g] - cnt * mean * mean) / (cnt - 1)
                    out[i] = np.sqrt(var) if var > 0.0 else 0.0
                else:
                    out[i] = 0.0
            elif stat_code == 3:
                out[i] = run_min[g]
            elif stat_code == 4:
                out[i] = run_max[g]
        v = sorted_vals[i]
        if np.isfinite(v):
            n_seen[g] = cnt + 1
            run_sum[g] += v
            run_sumsq[g] += v * v
            if v < run_min[g]:
                run_min[g] = v
            if v > run_max[g]:
                run_max[g] = v
    return out


def _expanding_stat_past_only(
    sorted_vals: np.ndarray, group_codes: np.ndarray, stat: str,
) -> np.ndarray:
    """Compute the EXPANDING ``stat`` over rows strictly BEFORE the current
    row, within each entity, given values already in (entity, time) order.

    ``group_codes`` is a dense per-entity integer code aligned with
    ``sorted_vals`` (also already in the sorted order). Returns an array the
    same length as ``sorted_vals``; the first row of each entity (no history)
    is left as NaN for the caller to fill with the global prior.

    Delegates to :func:`_expanding_stat_past_only_njit` (dict accumulators -> dense code arrays; ~200-500x over the
    Python loop, bit-identical incl NaN). Empty input returns an empty float array.
    """
    n = sorted_vals.size
    if n == 0:
        return np.full(0, np.nan, dtype=np.float64)
    gc = np.ascontiguousarray(group_codes, dtype=np.int64)
    n_groups = int(gc.max()) + 1 if gc.size else 1
    return np.asarray(_expanding_stat_past_only_njit(
        np.ascontiguousarray(sorted_vals, dtype=np.float64), gc, _EXPANDING_STAT_CODE[stat], n_groups,
    ))


def generate_expanding_agg_features(
    X: pd.DataFrame,
    entity_cols: Sequence[str],
    value_cols: Sequence[str],
    time_col: str,
    stats: Sequence[str] = EXPANDING_STATS,
):
    """Leak-safe expanding aggregations: per-entity stat over the strict past.

    Returns ``(enc_df, raw_recipes)`` where each recipe carries the per-entity
    sorted (time, value) history so :func:`apply_temporal_expanding` can replay
    a test frame against the TRAIN history only.
    """
    entity_cols, value_cols = _validate(
        X, entity_cols, value_cols, time_col, "generate_expanding_agg_features",
    )
    stats = [s for s in stats if s in EXPANDING_STATS]
    encoded: dict[str, np.ndarray] = {}
    raw_recipes: dict[str, dict] = {}
    if not entity_cols or not value_cols or not stats:
        return pd.DataFrame(index=X.index), raw_recipes

    key = _entity_key_series(X, entity_cols).to_numpy()
    time_vals = X[time_col].to_numpy()
    order = _stable_time_order(time_vals)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size)
    key_sorted = key[order]
    codes_sorted, _uniq = pd.factorize(key_sorted, sort=False)
    # n_codes / times_ord / the row-slice split depend only on (entity_cols, time_col), never value_col;
    # hoist out of the value_col loop instead of recomputing (incl. the O(N log N) argsort inside
    # _group_row_slices) for every value column.
    n_codes = int(codes_sorted.max()) + 1 if codes_sorted.size else 0
    times_ord = time_vals[order]
    row_slices = _group_row_slices(codes_sorted, n_codes)

    # Persist the per-entity sorted history (train snapshot) once per value col.
    for value_col in value_cols:
        vals = np.asarray(X[value_col].to_numpy(), dtype=np.float64)
        vals_sorted = vals[order]
        # History snapshot: per-entity sorted (time, value) lists.
        history: dict[str, dict] = {}
        for rows in row_slices:
            ent_key = str(key_sorted[rows[0]])
            history[ent_key] = {
                "t": times_ord[rows].tolist(),
                "v": vals_sorted[rows].tolist(),
            }
        for stat in stats:
            res_sorted = _expanding_stat_past_only(vals_sorted, codes_sorted, stat)
            prior = _global_prior(vals, stat)
            res_sorted = np.where(np.isfinite(res_sorted), res_sorted, prior)
            res = res_sorted[inv_order]
            name = engineered_name_expanding(value_col, entity_cols[0] if len(entity_cols) == 1 else "+".join(entity_cols), stat)
            encoded[name] = res
            raw_recipes[name] = {
                "entity_cols": list(entity_cols),
                "value_col": value_col,
                "time_col": time_col,
                "stat": stat,
                "history": history,
                "global_prior": prior,
            }
    enc_df = pd.DataFrame(encoded, index=X.index)
    return enc_df, raw_recipes


# ---------------------------------------------------------------------------
# Rolling time-window aggregations: carved to ``_temporal_agg_fe_rolling`` (generation, the njit
# two-pointer kernel, the recipe builder, and transform-time replay); re-exported below.
# ---------------------------------------------------------------------------


def _is_datetime_like(arr: np.ndarray) -> bool:
    """True if ``arr`` has a numpy datetime64 dtype; selects whether the rolling window is treated as a wall-clock ``Timedelta`` or a raw numeric span."""
    return np.issubdtype(np.asarray(arr).dtype, np.datetime64)


def _reduce(arr: np.ndarray, stat: str) -> float:
    """Reduce an array of past-window values to the requested statistic (mean/std/count/min/max/median); used by the pandas-fallback rolling path."""
    if stat == "mean":
        return float(np.mean(arr))
    if stat == "std":
        return float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if stat == "count":
        return float(arr.size)
    if stat == "min":
        return float(np.min(arr))
    if stat == "max":
        return float(np.max(arr))
    if stat == "median":
        return float(np.median(arr))
    raise ValueError(f"temporal_agg rolling: unknown stat {stat!r}")


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _lag_ring_buffer_njit(sorted_vals, group_codes, lags_arr, n_groups):
    """Single pass emitting EVERY requested lag via a per-entity ring buffer of size ``max(lags_arr)``, replacing
    an independent per-lag rescan of the identical per-entity sequence (the previous per-lag Python loop redid
    the ``g = int(codes_sorted[i])`` walk + buffer bookkeeping from scratch once per lag).

    ``out[k, i]`` is the group's value ``lags_arr[k]`` observations before row ``i`` (in time order), matching
    the reference per-entity growing list's ``buf[-lag]`` (read BEFORE the current row is appended) exactly:
    at row i, entity g has seen ``cnt[g]`` prior values, sitting at ring position ``(cnt[g]-lag) % max_lag``
    whenever ``cnt[g] >= lag`` (else NaN, no such predecessor yet). Since every requested lag is <= max_lag,
    that position is always still resident in the ring."""
    n = sorted_vals.size
    n_lags = lags_arr.size
    max_lag = 1
    for k in range(n_lags):
        if lags_arr[k] > max_lag:
            max_lag = lags_arr[k]
    out = np.full((n_lags, n), np.nan, dtype=np.float64)
    ring = np.zeros((n_groups, max_lag), dtype=np.float64)
    cnt = np.zeros(n_groups, dtype=np.int64)
    for i in range(n):
        g = group_codes[i]
        c = cnt[g]
        for k in range(n_lags):
            lag = lags_arr[k]
            if c >= lag:
                out[k, i] = ring[g, (c - lag) % max_lag]
        ring[g, c % max_lag] = sorted_vals[i]
        cnt[g] = c + 1
    return out


def generate_lag_features(
    X: pd.DataFrame,
    entity_cols: Sequence[str],
    value_cols: Sequence[str],
    time_col: str,
    lags: Sequence[int] = (1, 2, 3),
):
    """Leak-safe lag features: the entity's value at ``t - lag`` (in sorted
    time order). Positions with no ``lag``-th predecessor fall back to the
    global prior. Returns ``(enc_df, raw_recipes)``.
    """
    entity_cols, value_cols = _validate(
        X, entity_cols, value_cols, time_col, "generate_lag_features",
    )
    lags = [int(lo) for lo in (lags or []) if int(lo) >= 1]
    encoded: dict[str, np.ndarray] = {}
    raw_recipes: dict[str, dict] = {}
    if not entity_cols or not value_cols or not lags:
        return pd.DataFrame(index=X.index), raw_recipes

    key = _entity_key_series(X, entity_cols).to_numpy()
    time_vals = X[time_col].to_numpy()
    order = _stable_time_order(time_vals)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size)
    key_sorted = key[order]
    times_sorted = time_vals[order]
    codes_sorted, _ = pd.factorize(key_sorted, sort=False)
    codes_sorted_i64 = np.ascontiguousarray(codes_sorted, dtype=np.int64)
    ent_label = entity_cols[0] if len(entity_cols) == 1 else "+".join(entity_cols)
    # n_codes / the row-slice split depend only on (entity_cols, time_col), never value_col; hoist out
    # of the value_col loop instead of recomputing per value column.
    n_codes = int(codes_sorted.max()) + 1 if codes_sorted.size else 0
    row_slices = _group_row_slices(codes_sorted, n_codes)
    lags_arr = np.asarray(lags, dtype=np.int64)

    for value_col in value_cols:
        vals = np.asarray(X[value_col].to_numpy(), dtype=np.float64)
        vals_sorted = vals[order]
        prior = _global_prior(vals, "mean")
        history: dict[str, dict] = {}
        # Per-entity ordered value list (for lag-by-position).
        for rows in row_slices:
            ent_key = str(key_sorted[rows[0]])
            t_list = times_sorted[rows]
            history[ent_key] = {
                "t": (t_list.astype("datetime64[ns]").astype(np.int64).tolist() if _is_datetime_like(t_list) else t_list.astype(np.float64).tolist()),
                "v": vals_sorted[rows].astype(np.float64).tolist(),
                "is_datetime": bool(_is_datetime_like(t_list)),
            }
        # One pass over the time-sorted sequence emits EVERY requested lag via a max(lags)-sized per-entity ring
        # buffer, instead of an independent per-lag rescan of the same per-entity sequence (3x fewer row-visits
        # at the default 3 lags).
        lag_outputs = _lag_ring_buffer_njit(vals_sorted, codes_sorted_i64, lags_arr, max(n_codes, 1))
        for k, lag in enumerate(lags):
            res_sorted = np.where(np.isfinite(lag_outputs[k]), lag_outputs[k], prior)
            res = res_sorted[inv_order]
            name = engineered_name_lag(value_col, ent_label, lag)
            encoded[name] = res
            raw_recipes[name] = {
                "entity_cols": list(entity_cols),
                "value_col": value_col,
                "time_col": time_col,
                "lag": lag,
                "history": history,
                "global_prior": prior,
            }
    enc_df = pd.DataFrame(encoded, index=X.index)
    return enc_df, raw_recipes


# ---------------------------------------------------------------------------
# Recipe builders (re-exported from engineered_recipes)
# ---------------------------------------------------------------------------


def build_temporal_expanding_recipe(*, name, entity_cols, value_col, time_col, stat, history, global_prior):
    """Build an ``EngineeredRecipe`` carrying the per-entity TRAIN history + global prior needed to leak-safely replay an expanding-stat feature at transform time."""
    from .engineered_recipes import EngineeredRecipe
    return EngineeredRecipe(
        name=name, kind="temporal_expanding",
        src_names=(*tuple(entity_cols), value_col),
        extra={
            "entity_cols": [str(c) for c in entity_cols],
            "value_col": str(value_col),
            "time_col": str(time_col),
            "stat": str(stat),
            "history": history,
            "global_prior": float(global_prior),
        },
    )


def build_temporal_lag_recipe(*, name, entity_cols, value_col, time_col, lag, history, global_prior):
    """Build an ``EngineeredRecipe`` carrying the per-entity TRAIN history + global prior needed to leak-safely replay a lag feature at transform time."""
    from .engineered_recipes import EngineeredRecipe
    return EngineeredRecipe(
        name=name, kind="temporal_lag",
        src_names=(*tuple(entity_cols), value_col),
        extra={
            "entity_cols": [str(c) for c in entity_cols],
            "value_col": str(value_col),
            "time_col": str(time_col),
            "lag": int(lag),
            "history": history,
            "global_prior": float(global_prior),
        },
    )


# ---------------------------------------------------------------------------
# Replay (transform-time). Reads only X; computes against the stored TRAIN
# history plus earlier rows within the test frame itself.
# ---------------------------------------------------------------------------


def _coerce_replay_frame(X, entity_cols, value_col, time_col, recipe_name):
    """Coerce a transform-time ``X`` (pandas, polars, or a structured numpy array) to a pandas DataFrame exposing exactly the entity/value/time columns the recipe needs; raises for unsupported types."""
    if isinstance(X, pd.DataFrame):
        return X
    cols = [*list(entity_cols), value_col, time_col]
    try:
        import polars as _pl
        if isinstance(X, _pl.DataFrame):
            return pd.DataFrame({c: X[c].to_numpy() for c in cols})
    except ImportError:
        pass
    if isinstance(X, np.ndarray) and X.dtype.names is not None:
        return pd.DataFrame({c: X[c] for c in cols})
    raise TypeError(f"recipe '{recipe_name}': cannot extract temporal columns from X of " f"type {type(X).__name__}")


def _replay_keys_times(X_test, entity_cols, time_col):
    """Extract the per-row entity key array and raw time-column values from a replay frame, ready for time-ordered per-entity accumulation."""
    key = _entity_key_series(X_test, entity_cols).to_numpy()
    times = X_test[time_col].to_numpy()
    return key, times


def apply_temporal_expanding(X_test: pd.DataFrame, recipe_extra: dict) -> np.ndarray:
    """Replay an expanding stat for each test row against the stored TRAIN
    history (all earlier-timestamped train rows of the same entity) PLUS
    earlier rows of the same entity within the test frame -- never the row's
    own future."""
    entity_cols = list(recipe_extra["entity_cols"])
    value_col = recipe_extra["value_col"]
    time_col = recipe_extra["time_col"]
    stat = recipe_extra["stat"]
    prior = float(recipe_extra["global_prior"])
    history = recipe_extra["history"]
    X_test = _coerce_replay_frame(X_test, entity_cols, value_col, time_col, "temporal_expanding")
    key, times = _replay_keys_times(X_test, entity_cols, time_col)
    vals = np.asarray(X_test[value_col].to_numpy(), dtype=np.float64)
    n = len(X_test)
    out = np.full(n, prior, dtype=np.float64)
    # Process test rows in time order so within-test history accumulates.
    order = _stable_time_order(times)
    # Numeric time axis for the strict-past comparison. Datetime is compared in int64-ns (NOT float64: epoch-ns
    # ~1.6e18 exceeds float64's 2^53 exact-integer range, so a float cast loses ~hundreds of ns at the boundary).
    _is_dt = _is_datetime_like(times)
    times_num = times.astype("datetime64[ns]").astype(np.int64) if _is_dt else np.asarray(times, dtype=np.float64)
    # Running per-entity accumulators (count / sum / Welford-M2 / min / max). Train history is MERGED into the
    # time-ordered stream via a per-entity pointer: before scoring a test row at time t, only train rows STRICTLY
    # EARLIER than t are folded in. The previous implementation seeded the accumulator with the entity's ENTIRE train
    # history up front, so a test row whose timestamp fell inside the train time range saw FUTURE train values -- a
    # look-ahead leak / train-serve skew (fit never lets an expanding row see same-entity rows at a later time). This
    # matches apply_temporal_rolling's strict ``all_t < t`` contract. Each row stays O(1): O(N) total. mean uses
    # sum/count, std uses Welford-M2 (reduction-order delta ~1e-9, never selection-altering).
    acc_n: dict[str, int] = {}
    acc_sum: dict[str, float] = {}
    acc_mean: dict[str, float] = {}
    acc_m2: dict[str, float] = {}
    acc_min: dict[str, float] = {}
    acc_max: dict[str, float] = {}
    h_times: dict[str, np.ndarray] = {}
    h_vals: dict[str, np.ndarray] = {}
    h_pos: dict[str, int] = {}
    seeded: set[str] = set()

    def _fold(ent: str, v: float) -> None:
        """Fold one strictly-past value ``v`` (train history or earlier test row) into entity ``ent``'s running count/sum/Welford-M2/min/max accumulators."""
        cnt = acc_n[ent] + 1
        acc_n[ent] = cnt
        acc_sum[ent] += v
        delta = v - acc_mean[ent]
        acc_mean[ent] += delta / cnt
        acc_m2[ent] += delta * (v - acc_mean[ent])
        if v < acc_min[ent]:
            acc_min[ent] = v
        if v > acc_max[ent]:
            acc_max[ent] = v

    def _init(ent: str) -> None:
        """Lazily seed entity ``ent``'s sorted train-history arrays and zeroed accumulators the first time it is seen in the test time-ordered scan."""
        h = history.get(ent, {})
        _hv = np.asarray(h.get("v", []), dtype=np.float64)
        _ht = np.asarray(h.get("t", []))
        if _ht.size != _hv.size:
            # Legacy / hand-built recipe without stored train timestamps: fall back to the historical positional
            # seed-all (times = -inf so every train row folds before the first test row). Leak-safe only under the
            # old non-overlapping assumption; recipes produced by generate_expanding_agg_features carry "t".
            _ht = np.full(_hv.size, -np.inf, dtype=np.float64)
        elif _ht.size and not np.all(_ht[1:] >= _ht[:-1]):
            _o = np.argsort(_ht, kind="mergesort")
            _ht = _ht[_o]
            _hv = _hv[_o]
        h_times[ent] = _ht
        h_vals[ent] = _hv
        h_pos[ent] = 0
        acc_n[ent] = 0
        acc_sum[ent] = 0.0
        acc_mean[ent] = 0.0
        acc_m2[ent] = 0.0
        acc_min[ent] = np.inf
        acc_max[ent] = -np.inf
        seeded.add(ent)

    for idx in order:
        ent = str(key[idx])
        if ent not in seeded:
            _init(ent)
        # Fold train history strictly earlier than this row's timestamp (leak-safe merge, not one-shot seeding).
        t = times_num[idx]
        _ht = h_times[ent]
        _hv = h_vals[ent]
        _p = h_pos[ent]
        while _p < _ht.size and _ht[_p] < t:
            _vv = _hv[_p]
            if np.isfinite(_vv):
                _fold(ent, _vv)
            _p += 1
        h_pos[ent] = _p
        cnt = acc_n[ent]
        if cnt > 0:
            if stat == "count":
                out[idx] = float(cnt)
            elif stat == "mean":
                out[idx] = acc_sum[ent] / cnt
            elif stat == "std":
                out[idx] = float(np.sqrt(acc_m2[ent] / (cnt - 1))) if cnt > 1 else 0.0
            elif stat == "min":
                out[idx] = acc_min[ent]
            elif stat == "max":
                out[idx] = acc_max[ent]
            else:
                out[idx] = _reduce_expanding(np.array([acc_mean[ent]]), stat)
        v = vals[idx]
        if np.isfinite(v):
            _fold(ent, v)
    return out


def _reduce_expanding(arr: np.ndarray, stat: str) -> float:
    """Fallback reducer for expanding stats not handled by the fast running accumulators in :func:`apply_temporal_expanding` (e.g. ``median``); delegates to :func:`_reduce`."""
    if stat == "count":
        return float(arr.size)
    return _reduce(arr, stat)


def apply_temporal_lag(X_test: pd.DataFrame, recipe_extra: dict) -> np.ndarray:
    """Replay a lag feature: the entity's value ``lag`` positions earlier in
    the merged (train history ++ within-test) time order. Falls back to the
    global prior when there is no such predecessor."""
    entity_cols = list(recipe_extra["entity_cols"])
    value_col = recipe_extra["value_col"]
    time_col = recipe_extra["time_col"]
    lag = int(recipe_extra["lag"])
    prior = float(recipe_extra["global_prior"])
    history = recipe_extra["history"]
    X_test = _coerce_replay_frame(X_test, entity_cols, value_col, time_col, "temporal_lag")
    key, times = _replay_keys_times(X_test, entity_cols, time_col)
    vals = np.asarray(X_test[value_col].to_numpy(), dtype=np.float64)
    is_dt = _is_datetime_like(times)
    times_num = times.astype("datetime64[ns]").astype(np.int64) if is_dt else np.asarray(times, dtype=np.float64)
    n = len(X_test)
    out = np.full(n, prior, dtype=np.float64)
    order = _stable_time_order(times_num)
    # Per-entity time-sorted train history + a merge pointer, so a test row's positional lag counts ONLY train rows
    # STRICTLY EARLIER in time (plus earlier within-test rows). The previous implementation pre-seeded the buffer with
    # the entity's ENTIRE train history regardless of the test row's timestamp, so a test row inside the train time
    # range saw FUTURE train values in its positional history -- a look-ahead leak / train-serve skew. Train values
    # (finite or not) are appended in time order to preserve the positional-lag semantics of the fit side.
    h_sorted: dict[str, tuple] = {}
    h_pos: dict[str, int] = {}
    buffers: dict[str, list] = {}

    def _hist(ent: str) -> tuple:
        """Return entity ``ent``'s time-sorted (times, values) train-history arrays, falling back to an all-``-inf`` time axis for legacy recipes without stored timestamps."""
        h = history.get(ent, {})
        _v = np.asarray(h.get("v", []), dtype=np.float64)
        _t = np.asarray(h.get("t", []))
        if _t.size != _v.size:
            # Legacy / hand-built recipe without stored train timestamps: positional seed-all (times = -inf so the
            # whole train buffer folds before the first test row), the historical behaviour. Recipes produced by
            # generate_lag_features carry "t".
            _t = np.full(_v.size, -np.inf, dtype=np.float64)
        elif _t.size and not np.all(_t[1:] >= _t[:-1]):
            _o = np.argsort(_t, kind="mergesort")
            _t = _t[_o]
            _v = _v[_o]
        return _t, _v

    for idx in order:
        ent = str(key[idx])
        if ent not in h_pos:
            h_sorted[ent] = _hist(ent)
            h_pos[ent] = 0
            buffers[ent] = []
        _ht, _hv = h_sorted[ent]
        _p = h_pos[ent]
        t = times_num[idx]
        buf = buffers[ent]
        while _p < _ht.size and _ht[_p] < t:
            buf.append(float(_hv[_p]))
            _p += 1
        h_pos[ent] = _p
        if len(buf) >= lag:
            cand = buf[-lag]
            if np.isfinite(cand):
                out[idx] = cand
        buf.append(vals[idx])
    return out


# ---------------------------------------------------------------------------
# MI gate + end-to-end pipeline
# ---------------------------------------------------------------------------


def _mi_score(col: np.ndarray, y_bin: np.ndarray, n_bins: int = 10) -> float:
    """Quantile-bin a candidate temporal feature and compute its mutual information against the pre-binned target; used by :func:`hybrid_temporal_agg_fe` to rank/gate candidates before they enter the recipe pool."""
    from ._mi_greedy_cmi_fe import _quantile_bin, _cmi_from_binned
    x_bin = _quantile_bin(np.asarray(col, dtype=np.float64), nbins=n_bins)
    return float(_cmi_from_binned(x_bin, y_bin, None))


def hybrid_temporal_agg_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    entity_cols: Sequence[str],
    value_cols: Sequence[str],
    time_col: str,
    stats: Sequence[str] = ("mean", "std", "count"),
    windows: Sequence[str] = (),
    lags: Sequence[int] = (1,),
    top_k: int = 10,
    n_bins: int = 10,
    min_mi: float = 1e-4,
):
    """End-to-end leak-safe temporal FE pipeline.

    1. Materialise expanding (always), rolling (if ``windows``), and lag
       features.
    2. Score each candidate by ``MI(feature; y)`` (Layer 60 binned-MI
       primitive) and keep the top ``top_k`` above ``min_mi``.
    3. Append survivors to X; return ``(X_aug, appended, recipes, scores)``.

    ``y`` is consumed only by the MI gate; the recipes carry no y reference, so
    transform-time replay is leak-free.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"hybrid_temporal_agg_fe: X must be a pandas DataFrame; got " f"{type(X).__name__}")
    entity_cols = [c for c in (entity_cols or []) if c in X.columns]
    value_cols = [c for c in (value_cols or []) if c in X.columns]
    if not entity_cols or not value_cols or time_col not in X.columns:
        return X.copy(), [], [], pd.DataFrame()

    enc_exp, rec_exp = generate_expanding_agg_features(
        X, entity_cols, value_cols, time_col, stats=stats,
    )
    enc_parts = [enc_exp]
    raw_recipes: dict[str, dict] = {}
    raw_recipes.update({k: ("exp", v) for k, v in rec_exp.items()})
    if windows:
        enc_roll, rec_roll = generate_rolling_window_agg_features(
            X, entity_cols, value_cols, time_col, windows=windows,
            stats=[s for s in stats if s in ("mean", "std", "count", "min", "max")] or ("mean", "count"),
        )
        enc_parts.append(enc_roll)
        raw_recipes.update({k: ("roll", v) for k, v in rec_roll.items()})
    if lags:
        enc_lag, rec_lag = generate_lag_features(
            X, entity_cols, value_cols, time_col, lags=lags,
        )
        enc_parts.append(enc_lag)
        raw_recipes.update({k: ("lag", v) for k, v in rec_lag.items()})

    enc_df = pd.concat([p for p in enc_parts if p.shape[1]], axis=1) if any(p.shape[1] for p in enc_parts) else pd.DataFrame(index=X.index)
    if enc_df.empty:
        return X.copy(), [], [], pd.DataFrame()

    y_arr = np.asarray(y)
    if not np.issubdtype(y_arr.dtype, np.integer):
        # Continuous y must be quantile-binned, never int-truncated: astype(int64) collapses 0.7->0 and destroys the MI gate for regression targets.
        if y_arr.dtype.kind in "fc" and int(np.unique(y_arr).size) > 32:
            try:
                y_arr = pd.qcut(y_arr, q=10, labels=False, duplicates="drop").to_numpy()
            except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                logger.debug("suppressed in _temporal_agg_fe.py:967: %s", e)
                pass
        _, y_arr = np.unique(y_arr, return_inverse=True)
    y_bin = y_arr.astype(np.int64)

    rows = [{"engineered_col": col, "mi": _mi_score(enc_df[col].to_numpy(), y_bin, n_bins)} for col in enc_df.columns]
    scores = pd.DataFrame(rows).sort_values("mi", ascending=False, kind="mergesort").reset_index(drop=True)
    keep = scores[scores["mi"] >= float(min_mi)]
    winners = list(keep["engineered_col"].head(int(top_k)))
    if not winners:
        return X.copy(), [], [], scores

    from .engineered_recipes import (
        build_temporal_expanding_recipe,
        build_temporal_rolling_recipe,
        build_temporal_lag_recipe,
    )
    recipes = []
    for name in winners:
        kind, payload = raw_recipes[name]
        if kind == "exp":
            recipes.append(build_temporal_expanding_recipe(name=name, **payload))
        elif kind == "roll":
            recipes.append(build_temporal_rolling_recipe(name=name, **payload))
        else:
            recipes.append(build_temporal_lag_recipe(name=name, **payload))
    X_aug = pd.concat([X, enc_df[winners]], axis=1)
    return X_aug, winners, recipes, scores
