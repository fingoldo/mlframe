"""Rolling time-window family of ``_temporal_agg_fe`` (Layer 92 temporal FE).

Carved out of ``_temporal_agg_fe.py`` to keep that file under the 1k LOC ceiling. Holds the
rolling-time-window leak-safe aggregation: generation (``generate_rolling_window_agg_features``),
the fast per-entity two-pointer njit kernel (``_rolling_stat_past_only_njit`` /
``_rolling_stat_past_only``), the recipe builder (``build_temporal_rolling_recipe``), and the
transform-time replay (``apply_temporal_rolling``). See the parent module's docstring for the
shared leak-safety contract (train-history replay, never peeking at a row's own future).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numba
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "engineered_name_rolling",
    "generate_rolling_window_agg_features",
    "build_temporal_rolling_recipe",
    "apply_temporal_rolling",
]

_EXPANDING_STAT_CODE = {"count": 0, "mean": 1, "std": 2, "min": 3, "max": 4}


def engineered_name_rolling(value_col: str, entity_col: str, window: str, stat: str) -> str:
    """Canonical feature name for a rolling time-window-stat feature, e.g. ``troll_mean_7D(amount|user_id)``."""
    return f"troll_{stat}_{window}({value_col}|{entity_col})"


@numba.njit(cache=True)
def _rolling_stat_past_only_njit(times, vals, group_codes, td, stat_code, n_groups):
    """Rolling time-window stat over the strict past within each entity. Per-entity history is a newest->oldest
    linked list (``prev``/``last`` int arrays); for each row we walk back while ``times[j] >= t - td`` (ascending
    per-entity time -> the walk stops at the window edge), accumulating count/sum/sumsq/min/max over the in-window,
    strictly-past, finite values. Replaces the Python per-entity lists + the per-row list-comprehension window filter
    (O(n*window) with Python objects) -- same reduction, bit-identical incl NaN. ``times``/``td`` are int64 ns for a
    datetime column or float for numeric (numba specialises); ``stat_code``: 0=count 1=mean 2=std 3=min 4=max."""
    n = vals.size
    out = np.full(n, np.nan, dtype=np.float64)
    prev = np.full(n, -1, dtype=np.int64)
    last = np.full(n_groups, -1, dtype=np.int64)
    for i in range(n):
        g = group_codes[i]
        t = times[i]
        lo = t - td
        cnt = 0
        s = 0.0
        ss = 0.0
        mn = np.inf
        mx = -np.inf
        j = last[g]
        while j != -1 and times[j] >= lo:
            if times[j] < t:
                v = vals[j]
                if np.isfinite(v):
                    cnt += 1
                    s += v
                    ss += v * v
                    if v < mn:
                        mn = v
                    if v > mx:
                        mx = v
            j = prev[j]
        if cnt > 0:
            if stat_code == 0:
                out[i] = float(cnt)
            elif stat_code == 1:
                out[i] = s / cnt
            elif stat_code == 2:
                if cnt > 1:
                    mean = s / cnt
                    var = (ss - cnt * mean * mean) / (cnt - 1)
                    out[i] = np.sqrt(var) if var > 0.0 else 0.0
                else:
                    out[i] = 0.0
            elif stat_code == 3:
                out[i] = mn
            elif stat_code == 4:
                out[i] = mx
        prev[i] = last[g]
        last[g] = i
    return out


def _numeric_window(window: str) -> float:
    """Parse a numeric rolling window. Accepts a bare number ('7') or a pandas
    offset string whose leading integer is treated as a raw numeric span."""
    try:
        return float(window)
    except (TypeError, ValueError):
        pass
    # Strip a trailing unit letter from offset-like strings ('7D' -> 7).
    digits = "".join(ch for ch in str(window) if (ch.isdigit() or ch == "."))
    return float(digits) if digits else 1.0


def _rolling_stat_past_only(
    times: np.ndarray, vals: np.ndarray, group_codes: np.ndarray,
    window: str, stat: str,
) -> np.ndarray:
    """Rolling TIME-window stat over rows strictly BEFORE the current one,
    within each entity. ``times`` must already be in (entity, time) order and
    convertible to datetime / numeric. Window is a pandas offset string ('7D').

    Delegates to :func:`_rolling_stat_past_only_njit` (per-entity linked-list two-pointer walk; ~300-380x over the
    Python per-entity lists + per-row window-filter listcomp, bit-identical incl NaN).
    """
    from ._temporal_agg_fe import _is_datetime_like

    n = vals.size
    if n == 0:
        return np.full(0, np.nan, dtype=np.float64)
    td_num: Any
    if _is_datetime_like(times):
        # int64 nanoseconds so the window comparison keeps full datetime precision (float64 would lose ns on
        # ~1e18-scale timestamps). td is the window as ns.
        t_num = np.ascontiguousarray(np.asarray(times, dtype="datetime64[ns]").view("int64"))
        td_num = np.int64(pd.Timedelta(window).value)
    else:
        t_num = np.ascontiguousarray(times, dtype=np.float64)
        td_num = float(_numeric_window(window))
    gc = np.ascontiguousarray(group_codes, dtype=np.int64)
    n_groups = int(gc.max()) + 1 if gc.size else 1
    return np.asarray(_rolling_stat_past_only_njit(
        t_num, np.ascontiguousarray(vals, dtype=np.float64), gc, td_num, _EXPANDING_STAT_CODE[stat], n_groups,
    ))


def generate_rolling_window_agg_features(
    X: pd.DataFrame,
    entity_cols: Sequence[str],
    value_cols: Sequence[str],
    time_col: str,
    windows: Sequence[str] = ("7D", "30D"),
    stats: Sequence[str] = ("mean", "count"),
) -> tuple:
    """Leak-safe rolling time-window aggregations (one column per
    window x stat x value_col). Each row sees only earlier rows of its entity
    within the offset window. Returns ``(enc_df, raw_recipes)``.
    """
    from ._temporal_agg_fe import _entity_key_series, _group_row_slices, _global_prior, _is_datetime_like, _stable_time_order, _validate

    entity_cols, value_cols = _validate(
        X, entity_cols, value_cols, time_col,
        "generate_rolling_window_agg_features",
    )
    windows = [str(w) for w in (windows or [])]
    stats = [s for s in stats if s in ("mean", "std", "count", "min", "max", "median")]
    encoded: dict[str, np.ndarray] = {}
    raw_recipes: dict[str, dict] = {}
    if not entity_cols or not value_cols or not windows or not stats:
        return pd.DataFrame(index=X.index), raw_recipes

    key = _entity_key_series(X, entity_cols).to_numpy()
    time_vals = X[time_col].to_numpy()
    order = _stable_time_order(time_vals)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size)
    key_sorted = key[order]
    times_sorted = time_vals[order]
    codes_sorted, _ = pd.factorize(key_sorted, sort=False)
    ent_label = entity_cols[0] if len(entity_cols) == 1 else "+".join(entity_cols)
    # n_codes / the row-slice split depend only on (entity_cols, time_col), never value_col; hoist out
    # of the value_col loop instead of recomputing per value column.
    n_codes = int(codes_sorted.max()) + 1 if codes_sorted.size else 0
    row_slices = _group_row_slices(codes_sorted, n_codes)

    for value_col in value_cols:
        vals = np.asarray(X[value_col].to_numpy(), dtype=np.float64)
        vals_sorted = vals[order]
        history: dict[str, dict] = {}
        for rows in row_slices:
            ent_key = str(key_sorted[rows[0]])
            t_list = times_sorted[rows]
            history[ent_key] = {
                "t": (t_list.astype("datetime64[ns]").astype(np.int64).tolist() if _is_datetime_like(t_list) else t_list.astype(np.float64).tolist()),
                "v": vals_sorted[rows].astype(np.float64).tolist(),
                "is_datetime": bool(_is_datetime_like(t_list)),
            }
        for window in windows:
            for stat in stats:
                res_sorted = _rolling_stat_past_only(
                    times_sorted, vals_sorted, codes_sorted, window, stat,
                )
                prior = _global_prior(vals, stat)
                res_sorted = np.where(np.isfinite(res_sorted), res_sorted, prior)
                res = res_sorted[inv_order]
                name = engineered_name_rolling(value_col, ent_label, window, stat)
                encoded[name] = res
                raw_recipes[name] = {
                    "entity_cols": list(entity_cols),
                    "value_col": value_col,
                    "time_col": time_col,
                    "window": window,
                    "stat": stat,
                    "history": history,
                    "global_prior": prior,
                }
    enc_df = pd.DataFrame(encoded, index=X.index)
    return enc_df, raw_recipes


def build_temporal_rolling_recipe(
    *,
    name: str,
    entity_cols: Sequence[str],
    value_col: str,
    time_col: str,
    window: str,
    stat: str,
    history: Any,
    global_prior: float,
) -> Any:
    """Build an ``EngineeredRecipe`` carrying the per-entity TRAIN history + global prior needed to leak-safely replay a rolling time-window-stat feature at transform time."""
    from .engineered_recipes import EngineeredRecipe
    return EngineeredRecipe(
        name=name, kind="temporal_rolling",
        src_names=(*tuple(entity_cols), value_col),
        extra={
            "entity_cols": [str(c) for c in entity_cols],
            "value_col": str(value_col),
            "time_col": str(time_col),
            "window": str(window),
            "stat": str(stat),
            "history": history,
            "global_prior": float(global_prior),
        },
    )


def apply_temporal_rolling(X_test: pd.DataFrame, recipe_extra: dict) -> np.ndarray:
    """Replay a rolling time-window stat for each test row against the stored
    TRAIN history plus earlier within-test rows of the same entity, restricted
    to the offset window ending strictly before the row's timestamp."""
    from ._temporal_agg_fe import _coerce_replay_frame, _is_datetime_like, _reduce, _replay_keys_times, _stable_time_order

    entity_cols = list(recipe_extra["entity_cols"])
    value_col = recipe_extra["value_col"]
    time_col = recipe_extra["time_col"]
    window = recipe_extra["window"]
    stat = recipe_extra["stat"]
    prior = float(recipe_extra["global_prior"])
    history = recipe_extra["history"]
    X_test = _coerce_replay_frame(X_test, entity_cols, value_col, time_col, "temporal_rolling")
    key, times = _replay_keys_times(X_test, entity_cols, time_col)
    vals = np.asarray(X_test[value_col].to_numpy(), dtype=np.float64)
    is_dt = _is_datetime_like(times)
    td: Any
    if is_dt:
        times_num = times.astype("datetime64[ns]").astype(np.int64)
        td = int(pd.Timedelta(window).value)
    else:
        times_num = np.asarray(times, dtype=np.float64)
        td = _numeric_window(window)
    n = len(X_test)
    out = np.full(n, prior, dtype=np.float64)
    order = _stable_time_order(times_num)
    # FUTURE (perf): this per-row concatenate+mask reduce is O(N^2) per entity,
    # same shape as the expanding replay that was converted to O(N) accumulators.
    # A two-pointer sliding window (monotone left bound, since `t` is
    # non-decreasing within an entity's processing order) would make it O(N) for
    # mean/count/min/max, but train/test time interleaving + window eviction of
    # min/max needs a monotonic deque -- deferred as the rolling path is opt-in
    # (windows default empty) and correctness risk is higher than the win here.
    test_hist: dict[str, dict] = {}
    for idx in order:
        ent = str(key[idx])
        t = times_num[idx]
        h = history.get(ent, {})
        # Keep the time axis in its NATIVE dtype (int64-ns for datetime): a float64 cast of an epoch-ns value
        # (~1.6e18 > 2^53) loses ~hundreds of ns, so rows within a few hundred ns of the window bound ``lo``/``t``
        # could be mis-included/excluded. times_num above is already int64-ns for datetime; match it here.
        h_t = np.asarray(h.get("t", []))
        h_v = np.asarray(h.get("v", []), dtype=np.float64)
        seen = test_hist.get(ent, {"t": [], "v": []})
        all_t = np.concatenate([h_t, np.asarray(seen["t"])]) if seen["t"] else h_t
        all_v = np.concatenate([h_v, np.asarray(seen["v"], dtype=np.float64)]) if seen["v"] else h_v
        if all_t.size:
            lo = t - td
            mask = (all_t >= lo) & (all_t < t) & np.isfinite(all_v)
            if mask.any():
                out[idx] = _reduce(all_v[mask], stat)
        if np.isfinite(vals[idx]):
            d = test_hist.setdefault(ent, {"t": [], "v": []})
            d["t"].append(t)  # native scalar (int64-ns for datetime): no float round-trip precision loss
            d["v"].append(float(vals[idx]))
    return out
