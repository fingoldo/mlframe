"""Layer 38: CROSS-FEATURE RATIO + GROUPED-DELTA + LAGGED-DIFF FE.

Three patterns every real-world prod tabular ML pipeline uses:

* Ratio features: ``debt / income``, ``click / impression``, ``value / cost``.
  Either the ratio or its log-ratio is the natural signal; the raw pair
  rarely is.
* Grouped statistics: per-row deviation from a per-group baseline
  (``x - mean(x | g)``) and per-row z-score within group
  (``x / std(x | g)``). Captures "above/below average for this segment"
  patterns that no marginal feature can express.
* Lag-style differences: ``x_t - x_{t-1}`` (and longer lags), the time-series
  delta. The raw level often carries far less information than the change.

DESIGN: all three are CLOSED-FORM functions of X at replay. The fit-time
state is small:

* ``pairwise_ratio``  : empty extra (replay re-runs ``a / max(b, eps)``).
* ``grouped_delta``   : ``{group_col, num_col, lookup_mean, lookup_std,
                          global_mean, global_std}``.
* ``lagged_diff``     : ``{time_col, value_col, period}``. Replay sorts by
                        ``time_col`` AND emits diff per ``period``; row order
                        of the OUTPUT matches the input order (resort via the
                        permutation captured implicitly through pandas index).

extra layout:
* pairwise_ratio     : {kind: "div" | "log_div", eps: float}
* grouped_delta      : {group_col: str, num_col: str, op: "minus_mean" |
                       "div_std", lookup_mean: dict[str, float],
                       lookup_std: dict[str, float], global_mean: float,
                       global_std: float}
* lagged_diff        : {time_col: str, value_col: str, period: int}


Backends and memory (2026-11-16): the ordered-pair redundancy screen of the ratio and log-ratio pools decides on a per-task candidate buffer (host, ``_ratio_delta_screen``) or one block per pair
(device, ``_ratio_delta_screen_gpu``, tuned choice ``ratio_pair_screen_backend``), so only accepted columns are materialised and the device holds the (p, n) source matrix, read in row blocks per
thread pass. The MI gate runs through ``local_mi_gate``. ``audits/2026-10-22/profiles/_ratio_delta_fe.md`` has the measured figures."""

from __future__ import annotations

from ._grouped_coerce_shared import coerce_X_for_grouped

import logging
from typing import Any, Callable, Optional, Sequence

import numpy as np
from ._ratio_delta_screen import LOG_RATIO, RATIO, ratio_pair_screen
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
import pandas as pd

from ._internals import group_key_strings
from typing import TYPE_CHECKING
from mlframe.feature_selection.filters._fe_family_timing import fe_timed
if TYPE_CHECKING:
    from .engineered_recipes import EngineeredRecipe

logger = logging.getLogger(__name__)

__all__ = [
    "engineered_name_ratio",
    "engineered_name_log_ratio",
    "engineered_name_grouped_delta_mean",
    "engineered_name_grouped_delta_std",
    "engineered_name_lagged_diff",
    "pairwise_ratio_features",
    "pairwise_log_ratio_features",
    "grouped_delta_features",
    "lagged_diff_features",
    "apply_ratio",
    "apply_log_ratio",
    "apply_grouped_delta",
    "apply_lagged_diff",
    "pairwise_ratio_with_recipes",
    "pairwise_log_ratio_with_recipes",
    "grouped_delta_with_recipes",
    "lagged_diff_with_recipes",
]

# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def engineered_name_ratio(a: str, b: str) -> str:
    """Canonical column name for the ``a / b`` ratio feature; must match the name recipes replay under."""
    return f"ratio__{a}__{b}"

def engineered_name_log_ratio(a: str, b: str) -> str:
    """Canonical column name for the ``log1p(|a|) - log1p(|b|)`` log-ratio feature."""
    return f"log_ratio__{a}__{b}"

def engineered_name_grouped_delta_mean(num_col: str, group_col: str) -> str:
    """Canonical column name for the ``num_col - mean(num_col | group_col)`` grouped-delta feature."""
    return f"grouped_delta_{num_col}__{group_col}"

def engineered_name_grouped_delta_std(num_col: str, group_col: str) -> str:
    """Canonical column name for the grouped z-score ``(num_col - mean) / std(num_col | group_col)`` feature."""
    return f"grouped_zscore_{num_col}__{group_col}"

def engineered_name_lagged_diff(value_col: str, period: int) -> str:
    """Canonical column name for the ``value_col_t - value_col_{t-period}`` lagged-difference feature."""
    return f"lagged_diff_{value_col}__period{int(period)}"

def _map_group_keys(
    keys, lookup: dict, global_value: float, dtype: Any = None,
) -> np.ndarray:
    """Map each row key through ``lookup`` (str-keyed), unseen -> ``global_value``.

    Bit-identical to the per-row ``[lookup.get(str(k), global) for k in keys]``
    listcomp it replaces, but resolves the mapping with pandas' HASH-based
    ``Series.map`` instead of a Python per-row loop. The group keys arrive as an
    object-string array (from ``group_key_strings``); a sort-based
    ``np.unique`` gather is ~7x SLOWER on object strings than the per-row loop
    (it sorts n strings), so we hash-join here - ~2x faster than the loop at
    n=200k (microbench). The ``str()`` is a no-op on the already-string keys
    (kept for parity with callers that pass raw labels).

    Unseen keys -> ``global_value``; a key PRESENT with a NaN stored value keeps
    that NaN (matching ``dict.get``), so we fill only the genuinely-absent rows
    rather than blanket ``fillna`` (which would also overwrite present-NaN).
    """
    keys = np.asarray(keys)
    # Keys arrive as object strings from group_key_strings; only cast the rare
    # non-string carrier (str(k) is a no-op on already-string keys, matching the
    # original per-row str()).
    if keys.dtype != object:
        keys = keys.astype(str)
    s = pd.Series(keys, copy=False)
    # copy=True: some pandas versions return a read-only view from to_numpy() when the
    # requested dtype already matches the mapped Series' dtype (no-op astype short-circuit),
    # which then raises "assignment destination is read-only" on the absent-key fill below.
    out = s.map(lookup).to_numpy(dtype=_crit_np_dtype() if dtype is None else dtype, copy=True)
    absent = ~s.isin(lookup.keys()).to_numpy()
    if absent.any():
        out[absent] = global_value
    return np.asarray(out)

# ---------------------------------------------------------------------------
# Pairwise ratio: a / (sign(b) * max(|b|, eps))
# ---------------------------------------------------------------------------

def _safe_div(a: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    """Sign-preserving safe division: ``a / b`` with the denominator floored
    in absolute value at ``eps`` so the output never blows up to inf/nan even
    when ``b`` contains zero / very small values.
    """
    a = np.asarray(a, dtype=_crit_np_dtype())
    b = np.asarray(b, dtype=_crit_np_dtype())
    sign_b = np.where(b >= 0.0, 1.0, -1.0)
    denom = sign_b * np.maximum(np.abs(b), eps)
    return np.asarray(a / denom)

def pairwise_ratio_features(
    X: pd.DataFrame,
    cols: Sequence[str],
    *,
    eps: float = 1e-9,
    redundancy_corr_threshold: float = 0.99,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Emit ``ratio__{a}__{b}`` for every ordered (a, b) pair from ``cols``
    whose absolute Pearson correlation with BOTH ``a`` and ``b`` is below
    ``redundancy_corr_threshold`` (default 0.99).

    Returns
    -------
    enc_df : pd.DataFrame
        One column per surviving pair.
    accepted_pairs : list[tuple[str, str]]
        The pairs that survived the redundancy check, same order as columns.
    """
    if len(X) == 0:
        raise ValueError("pairwise_ratio_features: X is empty")
    cols_present = [c for c in cols if c in X.columns]
    encoded: dict[str, np.ndarray] = {}
    accepted: list[tuple[str, str]] = []
    cols_vals = [np.asarray(X[c].to_numpy(), dtype=_crit_np_dtype()) for c in cols_present]
    ok = ratio_pair_screen(cols_vals, RATIO, float(eps), float(redundancy_corr_threshold)) if cols_vals else np.zeros((0, 0), dtype=bool)
    for ia, a in enumerate(cols_present):
        for ib, b in enumerate(cols_present):
            if ok[ia, ib]:
                encoded[engineered_name_ratio(a, b)] = np.nan_to_num(_safe_div(cols_vals[ia], cols_vals[ib], float(eps)), nan=0.0, posinf=0.0, neginf=0.0)
                accepted.append((a, b))
    enc_df = pd.DataFrame(encoded, index=X.index, copy=False) if encoded else pd.DataFrame(index=X.index, copy=False)
    return enc_df, accepted

def pairwise_log_ratio_features(
    X: pd.DataFrame,
    cols: Sequence[str],
    *,
    eps: float = 1e-9,
    redundancy_corr_threshold: float = 0.99,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Emit ``log_ratio__{a}__{b} = log1p(|a|+eps) - log1p(|b|+eps)`` for
    every ordered pair (a, b) from ``cols``. The ``log1p(|.|)`` form handles
    negative values gracefully (collapses sign to magnitude) at the cost of
    losing sign information; use ``pairwise_ratio_features`` when sign
    matters.
    """
    if len(X) == 0:
        raise ValueError("pairwise_log_ratio_features: X is empty")
    cols_present = [c for c in cols if c in X.columns]
    encoded: dict[str, np.ndarray] = {}
    accepted: list[tuple[str, str]] = []
    cols_vals = [np.asarray(X[c].to_numpy(), dtype=_crit_np_dtype()) for c in cols_present]
    ok = ratio_pair_screen(cols_vals, LOG_RATIO, float(eps), float(redundancy_corr_threshold)) if cols_vals else np.zeros((0, 0), dtype=bool)
    for ia, a in enumerate(cols_present):
        for ib, b in enumerate(cols_present):
            if ok[ia, ib]:
                lr = np.log1p(np.abs(cols_vals[ia]) + float(eps)) - np.log1p(np.abs(cols_vals[ib]) + float(eps))
                encoded[engineered_name_log_ratio(a, b)] = np.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0)
                accepted.append((a, b))
    enc_df = pd.DataFrame(encoded, index=X.index, copy=False) if encoded else pd.DataFrame(index=X.index, copy=False)
    return enc_df, accepted

def apply_ratio(X_test: pd.DataFrame, a: str, b: str, eps: float) -> np.ndarray:
    """Replay a fitted ratio recipe on new data: recompute ``a / b`` with the same sign-preserving safe division as fit time."""
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"apply_ratio: X_test must be a DataFrame; got {type(X_test).__name__}")
    if a not in X_test.columns or b not in X_test.columns:
        raise KeyError(f"apply_ratio: missing source column(s) {a!r}/{b!r} from X_test")
    a_vals = np.asarray(X_test[a].to_numpy(), dtype=_crit_np_dtype())
    b_vals = np.asarray(X_test[b].to_numpy(), dtype=_crit_np_dtype())
    r = _safe_div(a_vals, b_vals, float(eps))
    return np.asarray(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0))

def apply_log_ratio(X_test: pd.DataFrame, a: str, b: str, eps: float) -> np.ndarray:
    """Replay a fitted log-ratio recipe on new data: recompute ``log1p(|a|+eps) - log1p(|b|+eps)``."""
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"apply_log_ratio: X_test must be a DataFrame; got {type(X_test).__name__}")
    if a not in X_test.columns or b not in X_test.columns:
        raise KeyError(f"apply_log_ratio: missing source column(s) {a!r}/{b!r} from X_test")
    a_vals = np.asarray(X_test[a].to_numpy(), dtype=_crit_np_dtype())
    b_vals = np.asarray(X_test[b].to_numpy(), dtype=_crit_np_dtype())
    lr = np.log1p(np.abs(a_vals) + float(eps)) - np.log1p(np.abs(b_vals) + float(eps))
    return np.asarray(np.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0))

# ---------------------------------------------------------------------------
# Grouped delta: x - mean(x | g) and x / std(x | g)
# ---------------------------------------------------------------------------

def grouped_delta_features(
    X: pd.DataFrame,
    group_col: str,
    num_cols: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Per-row grouped delta: for each ``num_col``, emit two engineered cols:

    * ``grouped_delta_{num_col}__{group_col}``  = x - mean(x | group)
    * ``grouped_zscore_{num_col}__{group_col}`` = (x - mean(x | group))
                                                  / std(x | group)

    Both use TRAIN per-group statistics at replay; unseen groups at test
    time fall back to the train global mean / std.
    """
    if len(X) == 0:
        raise ValueError("grouped_delta_features: X is empty")
    if group_col not in X.columns:
        raise ValueError(f"grouped_delta_features: group_col {group_col!r} missing from X")
    num_cols = [c for c in num_cols if c in X.columns and c != group_col]
    if not num_cols:
        return pd.DataFrame(index=X.index), {}

    g = pd.Series(group_key_strings(X[group_col]), index=X.index)
    encoded: dict[str, np.ndarray] = {}
    recipes: dict[str, dict] = {}
    # The grouper (hash of the group key) depends only on group_col, not num_col: build ONE frame holding every
    # num_col's crit-dtype values alongside "_g" and group it ONCE, instead of rebuilding a fresh 2-column temp
    # frame + re-hashing the identical "_g" key on every num_col iteration (num_cols is uncapped by the caller).
    x_by_col = {c: np.asarray(X[c].to_numpy(), dtype=np.float64) for c in num_cols}  # Intentional float64: the deviation from a group mean / the lag difference is a subtraction of nearly equal numbers, float32 operands lose it for a column with a large offset
    df_aux = pd.DataFrame({"_g": g.values, **x_by_col}, copy=False)
    gb = df_aux.groupby("_g")
    row_codes, uniq_keys = pd.factorize(g.values)
    for num_col in num_cols:
        x = x_by_col[num_col]
        # Per-group mean / std (sample std for unbiased; treat NaNs sanely).
        agg = gb[num_col].agg(["mean", "std"])
        # NaN std (single-row group) -> 1.0 fallback; std == 0 also -> 1.0.
        agg_std = agg["std"].fillna(0.0).to_numpy(dtype=np.float64)  # Intentional float64: same operands as the deviation below
        agg_std = np.where(agg_std > 0.0, agg_std, 1.0)
        lookup_mean = {str(k): float(v) for k, v in agg["mean"].items()}
        lookup_std = {str(k): float(s) for k, s in zip(agg.index.astype(str), agg_std)}
        finite_mask = np.isfinite(x)
        global_mean = float(np.nanmean(x[finite_mask])) if finite_mask.any() else 0.0
        global_std_raw = float(np.nanstd(x[finite_mask])) if finite_mask.any() else 1.0
        global_std = global_std_raw if global_std_raw > 0.0 else 1.0
        # Build the engineered columns.
        # Every training row's group is in the aggregate, so the per-row statistics are a gather of the per-group values by the once-factorised row codes (no per-column hash join of the n key strings).
        slot = pd.Index(agg.index).get_indexer(uniq_keys)
        per_row_mean = np.asarray(agg["mean"].to_numpy(dtype=np.float64)[slot][row_codes])  # Intentional float64: same operands as the deviation below
        per_row_std = np.asarray(agg_std[slot][row_codes])
        delta = x - per_row_mean
        zscore = delta / per_row_std
        delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0).astype(_crit_np_dtype(), copy=False)  # the stored column is narrowed AFTER the double-precision subtraction
        zscore = np.nan_to_num(zscore, nan=0.0, posinf=0.0, neginf=0.0).astype(_crit_np_dtype(), copy=False)
        name_delta = engineered_name_grouped_delta_mean(num_col, group_col)
        name_z = engineered_name_grouped_delta_std(num_col, group_col)
        encoded[name_delta] = delta
        encoded[name_z] = zscore
        recipes[name_delta] = {
            "group_col": group_col,
            "num_col": num_col,
            "op": "minus_mean",
            "lookup_mean": lookup_mean,
            "lookup_std": lookup_std,
            "global_mean": global_mean,
            "global_std": global_std,
        }
        recipes[name_z] = {
            "group_col": group_col,
            "num_col": num_col,
            "op": "div_std",
            "lookup_mean": lookup_mean,
            "lookup_std": lookup_std,
            "global_mean": global_mean,
            "global_std": global_std,
        }
    enc_df = pd.DataFrame(encoded, index=X.index, copy=False)
    return enc_df, recipes

def apply_grouped_delta(X_test: pd.DataFrame, recipe: dict) -> np.ndarray:
    """Replay a fitted grouped-delta/z-score recipe on new data using the TRAIN-time per-group mean/std lookups; unseen groups fall back to the stored global mean/std."""
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"apply_grouped_delta: X_test must be a DataFrame; got " f"{type(X_test).__name__}")
    group_col = recipe["group_col"]
    num_col = recipe["num_col"]
    op = recipe["op"]
    lookup_mean = dict(recipe["lookup_mean"])
    lookup_std = dict(recipe["lookup_std"])
    global_mean = float(recipe["global_mean"])
    # `or 1.0` would silently rewrite ANY falsy stored value, not just a genuine zero-std guard case
    # (fit-time already stores global_std as strictly > 0 -- see the mirrored `> 0.0 else 1.0` guard
    # where the recipe is built); an explicit comparison keeps the same zero-std fallback without that trap.
    _global_std_raw = float(recipe["global_std"])
    global_std = _global_std_raw if _global_std_raw > 0.0 else 1.0
    if group_col not in X_test.columns or num_col not in X_test.columns:
        raise KeyError(f"apply_grouped_delta: missing column(s) {group_col!r}/{num_col!r} " f"from X_test")
    g_vals = group_key_strings(X_test[group_col])
    x = np.asarray(X_test[num_col].to_numpy(), dtype=np.float64)  # Intentional float64: the deviation from a group mean / the lag difference is a subtraction of nearly equal numbers, float32 operands lose it for a column with a large offset
    per_row_mean = _map_group_keys(g_vals, lookup_mean, global_mean, np.float64)  # Intentional float64: subtracted from the float64 column
    delta = x - per_row_mean
    if op == "minus_mean":
        out = delta
    elif op == "div_std":
        per_row_std = _map_group_keys(g_vals, lookup_std, global_std, np.float64)  # Intentional float64: divides the float64 deviation
        per_row_std = np.where(per_row_std > 0.0, per_row_std, 1.0)
        out = delta / per_row_std
    else:
        raise ValueError(f"apply_grouped_delta: unknown op {op!r}")
    return np.asarray(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)).astype(_crit_np_dtype(), copy=False)

# ---------------------------------------------------------------------------
# Lagged diff: sort by time_col, compute x - x.shift(period)
# ---------------------------------------------------------------------------

def lagged_diff_features(
    X: pd.DataFrame,
    time_col: str,
    value_cols: Sequence[str],
    periods: Sequence[int] = (1, 2),
    entity_cols: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Emit ``lagged_diff_{value_col}__period{p}`` for each (value_col, p).

    The frame is sorted by ``time_col`` (stable mergesort) before the lag is
    computed; the output is reordered back to the input row order so the
    appended columns line up with X row-by-row.

    Recipes store ``{time_col, value_col, period, entity_cols}``; replay reproduces the
    sort + lag from X alone (no fit-time state needed - the operation is
    a pure function of the test frame).

    ``entity_cols``: optional per-entity scoping,
    mirroring ``_temporal_agg_fe.generate_lag_features``'s entity-scoped design. Without it (the default,
    unchanged behaviour), the diff is computed on the GLOBAL time-sorted order - on a panel/multi-entity
    dataset this can silently compute a cross-entity diff wherever the global time-sort places one entity's
    row immediately after a different entity's row. When given, the frame is sorted by ``(entity_key,
    time_col)`` instead, and a lag whose ``period``-back neighbour in the sorted order belongs to a
    DIFFERENT entity is zeroed (the same "no cross-boundary lag" contract ``generate_lag_features``'s
    per-entity ring buffer enforces), so single-series callers are byte-for-byte unaffected while panel
    callers get correct per-entity lags.
    """
    if len(X) == 0:
        raise ValueError("lagged_diff_features: X is empty")
    if time_col not in X.columns:
        raise ValueError(f"lagged_diff_features: time_col {time_col!r} missing from X")
    value_cols = [c for c in value_cols if c in X.columns and c != time_col]
    if not value_cols:
        return pd.DataFrame(index=X.index), {}
    periods = tuple(int(p) for p in periods if int(p) >= 1)
    if not periods:
        return pd.DataFrame(index=X.index), {}
    entity_cols = [c for c in (entity_cols or []) if c in X.columns] or None

    if entity_cols:
        from ._temporal_agg_fe import _entity_key_series

        entity_key = _entity_key_series(X, entity_cols).to_numpy()
        # Sort by (entity_key, time_col) - lexsort's LAST key is primary.
        sort_idx = np.lexsort((X[time_col].to_numpy(), entity_key))
        entity_sorted = entity_key[sort_idx]
    else:
        sort_idx = np.argsort(X[time_col].to_numpy(), kind="mergesort")
        entity_sorted = None
    inv_perm = np.empty_like(sort_idx)
    inv_perm[sort_idx] = np.arange(len(X))

    encoded: dict[str, np.ndarray] = {}
    recipes: dict[str, dict] = {}
    for value_col in value_cols:
        x = np.asarray(X[value_col].to_numpy(), dtype=np.float64)  # Intentional float64: the deviation from a group mean / the lag difference is a subtraction of nearly equal numbers, float32 operands lose it for a column with a large offset
        x_sorted = x[sort_idx]
        try:
            _time_num, _time_kind = _numeric_time(X[time_col].to_numpy())
            time_sorted = _time_num[sort_idx]
        except TypeError:
            time_sorted, _time_kind = None, ""
        for p in periods:
            diff_sorted = np.empty_like(x_sorted)
            diff_sorted[:p] = 0.0
            diff_sorted[p:] = x_sorted[p:] - x_sorted[:-p]
            if entity_sorted is not None:
                # Zero any diff whose period-back neighbour belongs to a different entity.
                cross_boundary = entity_sorted[p:] != entity_sorted[:-p]
                diff_sorted[p:][cross_boundary] = 0.0
            diff = diff_sorted[inv_perm]
            diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0).astype(_crit_np_dtype(), copy=False)  # narrowed AFTER the double-precision difference
            name = engineered_name_lagged_diff(value_col, p)
            encoded[name] = diff
            recipes[name] = {
                "time_col": time_col,
                "value_col": value_col,
                "period": int(p),
                "entity_cols": tuple(entity_cols) if entity_cols else None,
                "history": None if time_sorted is None else lag_history(time_sorted, _time_kind, x_sorted, entity_sorted, int(p)),
            }
    enc_df = pd.DataFrame(encoded, index=X.index, copy=False)
    return enc_df, recipes

#: Entities whose last observations the lagged-diff recipe keeps: beyond this many the recipe carries no history (the replay then starts every frame at 0 as before) so a panel with a huge entity count does not bloat it.
_LAG_HISTORY_MAX_ENTITIES = 20000


def _numeric_time(values: np.ndarray) -> "tuple[np.ndarray, str]":
    """``values`` of a time column as numbers (datetimes as int64 nanoseconds) and the kind tag (``'M'`` or ``'f'``) that turns them back; a column that is not numeric or datetime raises ``TypeError``."""
    values = np.asarray(values)
    if values.dtype.kind == "M":
        return values.astype("datetime64[ns]").astype(np.int64), "M"
    if values.dtype.kind in "iufb":
        # Intentional float64: numeric times stay in double precision so the strictly-earlier test of the lag replay is exact
        return values.astype(np.float64), "f"
    raise TypeError(f"time column of dtype {values.dtype} has no numeric form")


def lag_history(time_sorted: np.ndarray, kind: str, x_sorted: np.ndarray, entity_sorted: "np.ndarray | None", period: int) -> "dict | None":
    """The last ``period`` observations of every entity (of the whole series without entities) of a fit frame already sorted by (entity, time), the time as numbers of ``kind`` (see ``_numeric_time``): ``{"kind", "keys", "times", "values"}`` with plain lists, or
    ``None`` when the time column is not numeric / datetime or there are more than ``_LAG_HISTORY_MAX_ENTITIES`` entities. A replay frame that starts after these observations takes its first lags from them."""
    n = len(x_sorted)
    if n == 0:
        return None
    if entity_sorted is None:
        starts, ends = [0], [n]
        keys = [""]
    else:
        change = np.flatnonzero(entity_sorted[1:] != entity_sorted[:-1]) + 1
        starts = [0, *change.tolist()]
        ends = [*change.tolist(), n]
        if len(starts) > _LAG_HISTORY_MAX_ENTITIES:
            return None
        keys = [str(entity_sorted[s]) for s in starts]
    out_keys: list = []
    out_times: list = []
    out_values: list = []
    for key, s, e in zip(keys, starts, ends):
        lo = max(s, e - period)
        out_keys.extend([key] * (e - lo))
        out_times.extend(np.asarray(time_sorted[lo:e]).tolist())
        out_values.extend(np.asarray(x_sorted[lo:e]).tolist())
    return {"kind": kind, "keys": out_keys, "times": out_times, "values": out_values}


def _with_history(time_vals: np.ndarray, x: np.ndarray, entity_key: "np.ndarray | None", history: "dict | None") -> "tuple[np.ndarray, np.ndarray, np.ndarray | None, int]":
    """The replay arrays extended by the fit-time observations that precede each entity's first replay row (strictly earlier time); returns ``(time, x, entity, n_replay_rows)`` with the replay rows first."""
    n = len(x)
    if not history or not history.get("keys"):
        return time_vals, x, entity_key, n
    h_keys = np.asarray(history["keys"], dtype=object)
    # Intentional float64: the fit-time history must carry the same double-precision times and values the fit used, so the strictly-earlier test is exact
    h_times = np.asarray(history["times"], dtype=np.float64) if time_vals.dtype.kind == "f" else np.asarray(history["times"], dtype=np.int64)
    # Intentional float64: the fit-time values the replay extends, kept in double precision with their times
    h_vals = np.asarray(history["values"], dtype=np.float64)
    if entity_key is None:
        first = time_vals.min() if n else None
        keep = (h_times < first) if first is not None else np.zeros(len(h_times), dtype=bool)
    else:
        first_by_entity = pd.Series(time_vals).groupby(np.asarray([str(k) for k in entity_key], dtype=object)).min()
        limit = pd.Series(h_keys).map(first_by_entity)
        keep = (h_times < limit.to_numpy(dtype=time_vals.dtype, na_value=time_vals.dtype.type(0))) & limit.notna().to_numpy()
    if not keep.any():
        return time_vals, x, entity_key, n
    ent_ext = None if entity_key is None else np.concatenate([np.asarray([str(k) for k in entity_key], dtype=object), h_keys[keep]])  # one key type: the fit-time keys are strings
    return np.concatenate([time_vals, h_times[keep]]), np.concatenate([x, h_vals[keep]]), ent_ext, n


def apply_lagged_diff(X_test: pd.DataFrame, recipe: dict) -> np.ndarray:
    """Replay a fitted lagged-diff recipe on new data: resort ``X_test`` by ``time_col`` (or ``(entity_cols,
    time_col)`` when the recipe carries ``entity_cols``),
    compute the ``period``-step diff (zeroed across an entity boundary when entity-scoped), then unsort
    back to input row order (pure function of X_test alone, no fit-time state needed)."""
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"apply_lagged_diff: X_test must be a DataFrame; got " f"{type(X_test).__name__}")
    time_col = recipe["time_col"]
    value_col = recipe["value_col"]
    period = int(recipe["period"])
    entity_cols = recipe.get("entity_cols") or None
    if time_col not in X_test.columns or value_col not in X_test.columns:
        raise KeyError(f"apply_lagged_diff: missing column(s) {time_col!r}/{value_col!r} " f"from X_test")
    if entity_cols:
        missing_entity = [c for c in entity_cols if c not in X_test.columns]
        if missing_entity:
            raise KeyError(f"apply_lagged_diff: missing entity column(s) {missing_entity!r} from X_test; a global-order diff would cross entity boundaries")
    time_vals = X_test[time_col].to_numpy()
    entity_key = None
    if entity_cols:
        from ._temporal_agg_fe import _entity_key_series

        entity_key = _entity_key_series(X_test, entity_cols).to_numpy()
    x = np.asarray(X_test[value_col].to_numpy(), dtype=np.float64)  # Intentional float64: the deviation from a group mean / the lag difference is a subtraction of nearly equal numbers, float32 operands lose it for a column with a large offset
    n_rows = len(x)
    history = recipe.get("history")
    if history:
        # the fit-time observations that precede the first replay row of each entity serve as its first lags (a live batch of one row would otherwise be 0 in every lag column)
        try:
            time_num, kind = _numeric_time(time_vals)
        except TypeError:
            time_num, kind = None, ""
        if time_num is not None and kind == history.get("kind"):
            time_vals, x, entity_key, n_rows = _with_history(time_num, x, entity_key, history)
    if entity_key is not None:
        sort_idx = np.lexsort((time_vals, entity_key))
        entity_sorted = entity_key[sort_idx]
    else:
        sort_idx = np.argsort(time_vals, kind="mergesort")
        entity_sorted = None
    inv_perm = np.empty_like(sort_idx)
    inv_perm[sort_idx] = np.arange(len(x))
    x_sorted = x[sort_idx]
    diff_sorted = np.empty_like(x_sorted)
    diff_sorted[:period] = 0.0
    diff_sorted[period:] = x_sorted[period:] - x_sorted[:-period]
    if entity_sorted is not None:
        cross_boundary = entity_sorted[period:] != entity_sorted[:-period]
        diff_sorted[period:][cross_boundary] = 0.0
    out = diff_sorted[inv_perm][:n_rows]
    return np.asarray(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)).astype(_crit_np_dtype(), copy=False)

# ---------------------------------------------------------------------------
# End-to-end wrappers for MRMR.fit auto-wiring
# ---------------------------------------------------------------------------

def _gate_ratio_pairs(enc_df, accepted, name_fn, y, raw_X, mi_gate, mi_gate_top_k, reject_sink=None):
    """Tier-1 local MI floor over an ordered-pair ratio pool. Returns the
    pruned ``(enc_df, accepted)`` keeping only pairs whose engineered column
    clears the raw-baseline noise floor (top-K by MI). No-op when ``mi_gate``
    is False or ``y`` is None."""
    if not mi_gate or y is None or enc_df is None or enc_df.empty:
        return enc_df, accepted
    from ._unified_fe_gate import local_mi_gate

    keep = set(local_mi_gate(enc_df, y, raw_X=raw_X, top_k=mi_gate_top_k, reject_sink=reject_sink))
    if not keep:
        return enc_df.iloc[:, :0], []
    accepted = [(a, b) for (a, b) in accepted if name_fn(a, b) in keep]
    enc_df = enc_df[[name_fn(a, b) for (a, b) in accepted]]
    return enc_df, accepted

@fe_timed("pairwise_ratio")
def pairwise_ratio_with_recipes(
    X: pd.DataFrame,
    *,
    cols: Optional[Sequence[str]] = None,
    eps: float = 1e-9,
    mi_gate: bool = False,
    mi_gate_top_k: Optional[int] = None,
    y: Optional[np.ndarray] = None,
    reject_sink: Optional[Callable[..., None]] = None,
) -> tuple[pd.DataFrame, list[str], list[EngineeredRecipe]]:
    """Append ratio columns to X and emit one recipe per accepted pair.

    ``mi_gate=True`` (with ``y``) applies the Tier-1 local MI floor (Layer 91)
    over the O(p^2) ordered-pair ratio pool - the most explosion-prone L38
    emitter.
    """
    from .engineered_recipes import build_pairwise_ratio_recipe

    if not cols:
        return X, [], []
    cols = [c for c in cols if c in X.columns]
    if len(cols) < 2:
        return X, [], []
    enc_df, accepted = pairwise_ratio_features(X, cols, eps=float(eps))
    if not accepted:
        return X, [], []
    enc_df, accepted = _gate_ratio_pairs(
        enc_df, accepted, engineered_name_ratio, y, X, mi_gate, mi_gate_top_k,
        reject_sink=reject_sink,
    )
    if not accepted:
        return X, [], []
    X_aug = pd.concat([X, enc_df], axis=1)
    appended = list(enc_df.columns)
    recipes = [
        build_pairwise_ratio_recipe(
            name=engineered_name_ratio(a, b), src_a_name=a, src_b_name=b,
            kind="div", eps=float(eps),
        )
        for (a, b) in accepted
    ]
    return X_aug, appended, recipes

@fe_timed("pairwise_log_ratio")
def pairwise_log_ratio_with_recipes(
    X: pd.DataFrame,
    *,
    cols: Optional[Sequence[str]] = None,
    eps: float = 1e-9,
    mi_gate: bool = False,
    mi_gate_top_k: Optional[int] = None,
    y: Optional[np.ndarray] = None,
    reject_sink: Optional[Callable[..., None]] = None,
) -> tuple[pd.DataFrame, list[str], list[EngineeredRecipe]]:
    """Append log-ratio columns and emit one recipe per accepted pair.

    ``mi_gate=True`` (with ``y``) applies the Tier-1 local MI floor (Layer 91).
    """
    from .engineered_recipes import build_pairwise_ratio_recipe

    if not cols:
        return X, [], []
    cols = [c for c in cols if c in X.columns]
    if len(cols) < 2:
        return X, [], []
    enc_df, accepted = pairwise_log_ratio_features(X, cols, eps=float(eps))
    if not accepted:
        return X, [], []
    enc_df, accepted = _gate_ratio_pairs(
        enc_df, accepted, engineered_name_log_ratio, y, X, mi_gate, mi_gate_top_k,
        reject_sink=reject_sink,
    )
    if not accepted:
        return X, [], []
    X_aug = pd.concat([X, enc_df], axis=1)
    appended = list(enc_df.columns)
    recipes = [
        build_pairwise_ratio_recipe(
            name=engineered_name_log_ratio(a, b), src_a_name=a, src_b_name=b,
            kind="log_div", eps=float(eps),
        )
        for (a, b) in accepted
    ]
    return X_aug, appended, recipes

@fe_timed("grouped_delta")
def grouped_delta_with_recipes(
    X: pd.DataFrame,
    *,
    group_col: Optional[str] = None,
    num_cols: Optional[Sequence[str]] = None,
    mi_gate: bool = False,
    mi_gate_top_k: Optional[int] = None,
    y: Optional[np.ndarray] = None,
    reject_sink: Optional[Callable[..., None]] = None,
) -> tuple[pd.DataFrame, list[str], list[EngineeredRecipe]]:
    """Append grouped-delta + grouped-zscore columns and emit one recipe per
    engineered column.

    ``mi_gate=True`` (with ``y``) applies the Tier-1 local MI floor (Layer 91)
    over the 2*|num_cols| grouped-stat pool.
    """
    from .engineered_recipes import build_grouped_delta_recipe

    if not group_col or group_col not in X.columns or not num_cols:
        return X, [], []
    num_cols = [c for c in num_cols if c in X.columns and c != group_col]
    if not num_cols:
        return X, [], []
    enc_df, raw_recipes = grouped_delta_features(X, group_col, num_cols)
    if enc_df.empty:
        return X, [], []
    if mi_gate and y is not None:
        from ._unified_fe_gate import local_mi_gate

        keep = set(local_mi_gate(enc_df, y, raw_X=X, top_k=mi_gate_top_k, reject_sink=reject_sink))
        if not keep:
            return X, [], []
        enc_df = enc_df[[c for c in enc_df.columns if c in keep]]
    X_aug = pd.concat([X, enc_df], axis=1)
    appended = list(enc_df.columns)
    recipes = [build_grouped_delta_recipe(name=name, **raw_recipes[name]) for name in appended]
    return X_aug, appended, recipes

@fe_timed("lagged_diff")
def lagged_diff_with_recipes(
    X: pd.DataFrame,
    *,
    time_col: Optional[str] = None,
    value_cols: Optional[Sequence[str]] = None,
    periods: Sequence[int] = (1, 2),
    entity_cols: Optional[Sequence[str]] = None,
    mi_gate: bool = False,
    mi_gate_top_k: Optional[int] = None,
    y: Optional[np.ndarray] = None,
    reject_sink: Optional[Callable[..., None]] = None,
) -> tuple[pd.DataFrame, list[str], list[EngineeredRecipe]]:
    """Append lagged-diff columns and emit one recipe per (value_col, period).

    ``mi_gate=True`` (with ``y``) applies the Tier-1 local MI floor (Layer 91)
    over the |value_cols| * |periods| lag pool.

    ``entity_cols``: optional per-entity scoping - see
    :func:`lagged_diff_features`'s docstring. ``None`` (the default) preserves the exact prior global-sort
    behaviour.
    """
    from .engineered_recipes import build_lagged_diff_recipe

    if not time_col or time_col not in X.columns or not value_cols:
        return X, [], []
    value_cols = [c for c in value_cols if c in X.columns and c != time_col]
    if not value_cols:
        return X, [], []
    periods = tuple(int(p) for p in periods if int(p) >= 1)
    if not periods:
        return X, [], []
    enc_df, raw_recipes = lagged_diff_features(X, time_col, value_cols, periods, entity_cols=entity_cols)
    if enc_df.empty:
        return X, [], []
    if mi_gate and y is not None:
        from ._unified_fe_gate import local_mi_gate

        keep = set(local_mi_gate(enc_df, y, raw_X=X, top_k=mi_gate_top_k, reject_sink=reject_sink))
        if not keep:
            return X, [], []
        enc_df = enc_df[[c for c in enc_df.columns if c in keep]]
    X_aug = pd.concat([X, enc_df], axis=1)
    appended = list(enc_df.columns)
    recipes = [build_lagged_diff_recipe(name=name, **raw_recipes[name]) for name in appended]
    return X_aug, appended, recipes

# ---------------------------------------------------------------------------
# Recipe-apply adapters consumed by engineered_recipes.apply_recipe
# ---------------------------------------------------------------------------

def _coerce_X_for_pair(X, a: str, b: str, recipe_name: str) -> pd.DataFrame:
    """Extract the two source columns ``a``/``b`` from an arbitrary carrier (pandas, polars, or structured ndarray) as a pandas DataFrame so ``apply_ratio``/``apply_log_ratio`` can run at recipe-replay time."""
    if isinstance(X, pd.DataFrame):
        return X
    try:
        import polars as _pl
        if isinstance(X, _pl.DataFrame):
            return pd.DataFrame({a: X[a].to_numpy(), b: X[b].to_numpy()}, copy=False)
    except ImportError:
        pass
    if isinstance(X, np.ndarray) and X.dtype.names is not None:
        return pd.DataFrame({a: X[a], b: X[b]}, copy=False)
    raise TypeError(f"recipe '{recipe_name}': cannot extract columns {a!r}/{b!r} from " f"X of type {type(X).__name__}")

def _apply_pairwise_ratio_recipe(recipe, X) -> np.ndarray:
    """Recipe-apply adapter (consumed by ``engineered_recipes.apply_recipe``): dispatches to ``apply_ratio`` or ``apply_log_ratio`` based on the recipe's stored ``kind``."""
    if len(recipe.src_names) != 2:
        raise ValueError(f"pairwise_ratio recipe '{recipe.name}' must have exactly 2 " f"src_names; got {len(recipe.src_names)}")
    a, b = recipe.src_names
    kind = str(recipe.extra.get("kind", "div"))
    eps = float(recipe.extra.get("eps", 1e-9))
    X_view = _coerce_X_for_pair(X, a, b, recipe.name)
    if kind == "div":
        return apply_ratio(X_view, a, b, eps)
    if kind == "log_div":
        return apply_log_ratio(X_view, a, b, eps)
    raise ValueError(f"pairwise_ratio recipe '{recipe.name}': unknown kind {kind!r}")

def _apply_grouped_delta_recipe(recipe, X) -> np.ndarray:
    """Recipe-apply adapter (consumed by ``engineered_recipes.apply_recipe``): rebuilds the ``apply_grouped_delta`` recipe dict from the frozen ``recipe.extra`` payload and replays it on ``X``."""
    group_col = str(recipe.extra["group_col"])
    num_col = str(recipe.extra["num_col"])
    X_view = coerce_X_for_grouped(X, group_col, num_col, recipe.name)
    return apply_grouped_delta(
        X_view,
        {
            "group_col": group_col,
            "num_col": num_col,
            "op": str(recipe.extra.get("op", "minus_mean")),
            "lookup_mean": dict(recipe.extra.get("lookup_mean", {})),
            "lookup_std": dict(recipe.extra.get("lookup_std", {})),
            "global_mean": float(recipe.extra.get("global_mean", 0.0)),
            "global_std": float(recipe.extra.get("global_std", 1.0)),
        },
    )

def _coerce_X_for_lagged_diff(X, time_col: str, value_col: str, recipe_name: str, entity_cols: "tuple[str, ...] | None" = None) -> pd.DataFrame:
    """Extract ``time_col``/``value_col``/``entity_cols`` from an arbitrary carrier (pandas, polars, or structured ndarray) as a pandas DataFrame for ``apply_lagged_diff`` at recipe-replay time."""
    if isinstance(X, pd.DataFrame):
        return X
    _entity_cols = entity_cols or ()
    try:
        import polars as _pl
        if isinstance(X, _pl.DataFrame):
            cols = {time_col: X[time_col].to_numpy(), value_col: X[value_col].to_numpy()}
            for c in _entity_cols:
                cols[c] = X[c].to_numpy()
            return pd.DataFrame(cols, copy=False)
    except ImportError:
        pass
    if isinstance(X, np.ndarray) and X.dtype.names is not None:
        cols_np = {time_col: X[time_col], value_col: X[value_col]}
        for c in _entity_cols:
            cols_np[c] = X[c]
        return pd.DataFrame(cols_np, copy=False)
    raise TypeError(f"recipe '{recipe_name}': cannot extract {time_col!r}/{value_col!r} " f"from X of type {type(X).__name__}")

def _apply_lagged_diff_recipe(recipe, X) -> np.ndarray:
    """Recipe-apply adapter (consumed by ``engineered_recipes.apply_recipe``): rebuilds the ``apply_lagged_diff`` recipe dict from the frozen ``recipe.extra`` payload and replays it on ``X``."""
    time_col = str(recipe.extra["time_col"])
    value_col = str(recipe.extra["value_col"])
    period = int(recipe.extra["period"])
    entity_cols = recipe.extra.get("entity_cols") or None
    X_view = _coerce_X_for_lagged_diff(X, time_col, value_col, recipe.name, entity_cols=entity_cols)
    return apply_lagged_diff(
        X_view, {"time_col": time_col, "value_col": value_col, "period": period, "entity_cols": entity_cols, "history": recipe.extra.get("history")},
    )
