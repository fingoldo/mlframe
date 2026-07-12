"""Layer 38 (2026-05-31): CROSS-FEATURE RATIO + GROUPED-DELTA + LAGGED-DIFF FE.

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
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from ._internals import group_key_strings

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
    keys, lookup: dict, global_value: float,
) -> np.ndarray:
    """Map each row key through ``lookup`` (str-keyed), unseen -> ``global_value``.

    Bit-identical to the per-row ``[lookup.get(str(k), global) for k in keys]``
    listcomp it replaces, but resolves the mapping with pandas' HASH-based
    ``Series.map`` instead of a Python per-row loop. The group keys arrive as an
    object-string array (from ``group_key_strings``); a sort-based
    ``np.unique`` gather is ~7x SLOWER on object strings than the per-row loop
    (it sorts n strings), so we hash-join here -- ~2x faster than the loop at
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
    out = s.map(lookup).to_numpy(dtype=np.float64)
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
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
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
    # bench-attempt-rejected (2026-06-23): hoisting per-column to_numpy() into an
    # outer dict to avoid the in-loop re-extract was 0.98x (13.3s->13.6s, p=40
    # n=100k) -- the O(p^2) _passes_redundancy corrcoef dominates the loop, the
    # to_numpy() re-extract is noise. Bench: _benchmarks/bench_ratio_delta_fe.py.
    encoded: dict[str, np.ndarray] = {}
    accepted: list[tuple[str, str]] = []
    for a in cols_present:
        for b in cols_present:
            if a == b:
                continue
            a_vals = np.asarray(X[a].to_numpy(), dtype=np.float64)
            b_vals = np.asarray(X[b].to_numpy(), dtype=np.float64)
            r = _safe_div(a_vals, b_vals, float(eps))
            r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
            # Redundancy gate: drop ratios that are nearly linearly dependent
            # on either source column (no info gain over the raw pair).
            if not _passes_redundancy(r, a_vals, b_vals, redundancy_corr_threshold):
                continue
            encoded[engineered_name_ratio(a, b)] = r
            accepted.append((a, b))
    enc_df = pd.DataFrame(encoded, index=X.index) if encoded else pd.DataFrame(index=X.index)
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
    # bench-attempt-rejected (2026-06-23): per-column to_numpy() hoist was 1.00x
    # here too (16.8s, p=40 n=100k) -- corrcoef redundancy gate dominates, not
    # the re-extract. See pairwise_ratio_features note + bench_ratio_delta_fe.py.
    encoded: dict[str, np.ndarray] = {}
    accepted: list[tuple[str, str]] = []
    for a in cols_present:
        for b in cols_present:
            if a == b:
                continue
            a_vals = np.asarray(X[a].to_numpy(), dtype=np.float64)
            b_vals = np.asarray(X[b].to_numpy(), dtype=np.float64)
            lr = np.log1p(np.abs(a_vals) + float(eps)) - np.log1p(np.abs(b_vals) + float(eps))
            lr = np.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0)
            if not _passes_redundancy(lr, a_vals, b_vals, redundancy_corr_threshold):
                continue
            encoded[engineered_name_log_ratio(a, b)] = lr
            accepted.append((a, b))
    enc_df = pd.DataFrame(encoded, index=X.index) if encoded else pd.DataFrame(index=X.index)
    return enc_df, accepted


def _passes_redundancy(
    candidate: np.ndarray, a_vals: np.ndarray, b_vals: np.ndarray, threshold: float,
) -> bool:
    """Reject the candidate column if |Pearson(candidate, a)| > threshold or
    |Pearson(candidate, b)| > threshold -- i.e. it would carry no new info
    over the source pair."""
    cand = candidate
    if cand.std() <= 1e-12:
        return False
    for src in (a_vals, b_vals):
        if src.std() <= 1e-12:
            continue
        # Pearson on the finite mask only.
        mask = np.isfinite(cand) & np.isfinite(src)
        if not mask.any():
            continue
        c2 = cand[mask]
        s2 = src[mask]
        if c2.std() <= 1e-12 or s2.std() <= 1e-12:
            continue
        rho = float(np.corrcoef(c2, s2)[0, 1])
        if abs(rho) > float(threshold):
            return False
    return True


def apply_ratio(X_test: pd.DataFrame, a: str, b: str, eps: float) -> np.ndarray:
    """Replay a fitted ratio recipe on new data: recompute ``a / b`` with the same sign-preserving safe division as fit time."""
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"apply_ratio: X_test must be a DataFrame; got {type(X_test).__name__}")
    if a not in X_test.columns or b not in X_test.columns:
        raise KeyError(f"apply_ratio: missing source column(s) {a!r}/{b!r} from X_test")
    a_vals = np.asarray(X_test[a].to_numpy(), dtype=np.float64)
    b_vals = np.asarray(X_test[b].to_numpy(), dtype=np.float64)
    r = _safe_div(a_vals, b_vals, float(eps))
    return np.asarray(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0))


def apply_log_ratio(X_test: pd.DataFrame, a: str, b: str, eps: float) -> np.ndarray:
    """Replay a fitted log-ratio recipe on new data: recompute ``log1p(|a|+eps) - log1p(|b|+eps)``."""
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"apply_log_ratio: X_test must be a DataFrame; got {type(X_test).__name__}")
    if a not in X_test.columns or b not in X_test.columns:
        raise KeyError(f"apply_log_ratio: missing source column(s) {a!r}/{b!r} from X_test")
    a_vals = np.asarray(X_test[a].to_numpy(), dtype=np.float64)
    b_vals = np.asarray(X_test[b].to_numpy(), dtype=np.float64)
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
    # num_col's float64-cast values alongside "_g" and group it ONCE, instead of rebuilding a fresh 2-column temp
    # frame + re-hashing the identical "_g" key on every num_col iteration (num_cols is uncapped by the caller).
    x_by_col = {c: np.asarray(X[c].to_numpy(), dtype=np.float64) for c in num_cols}
    df_aux = pd.DataFrame({"_g": g.values, **x_by_col})
    gb = df_aux.groupby("_g")
    for num_col in num_cols:
        x = x_by_col[num_col]
        # Per-group mean / std (sample std for unbiased; treat NaNs sanely).
        agg = gb[num_col].agg(["mean", "std"])
        # NaN std (single-row group) -> 1.0 fallback; std == 0 also -> 1.0.
        agg_std = agg["std"].fillna(0.0).to_numpy(dtype=np.float64)
        agg_std = np.where(agg_std > 0.0, agg_std, 1.0)
        lookup_mean = {str(k): float(v) for k, v in agg["mean"].items()}
        lookup_std = {str(k): float(s) for k, s in zip(agg.index.astype(str), agg_std)}
        finite_mask = np.isfinite(x)
        global_mean = float(np.nanmean(x[finite_mask])) if finite_mask.any() else 0.0
        global_std_raw = float(np.nanstd(x[finite_mask])) if finite_mask.any() else 1.0
        global_std = global_std_raw if global_std_raw > 0.0 else 1.0
        # Build the engineered columns.
        per_row_mean = _map_group_keys(g.values, lookup_mean, global_mean)
        per_row_std = _map_group_keys(g.values, lookup_std, global_std)
        delta = x - per_row_mean
        zscore = delta / per_row_std
        delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        zscore = np.nan_to_num(zscore, nan=0.0, posinf=0.0, neginf=0.0)
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
    enc_df = pd.DataFrame(encoded, index=X.index)
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
    global_std = float(recipe["global_std"]) or 1.0
    if group_col not in X_test.columns or num_col not in X_test.columns:
        raise KeyError(f"apply_grouped_delta: missing column(s) {group_col!r}/{num_col!r} " f"from X_test")
    g_vals = group_key_strings(X_test[group_col])
    x = np.asarray(X_test[num_col].to_numpy(), dtype=np.float64)
    per_row_mean = _map_group_keys(g_vals, lookup_mean, global_mean)
    delta = x - per_row_mean
    if op == "minus_mean":
        out = delta
    elif op == "div_std":
        per_row_std = _map_group_keys(g_vals, lookup_std, global_std)
        per_row_std = np.where(per_row_std > 0.0, per_row_std, 1.0)
        out = delta / per_row_std
    else:
        raise ValueError(f"apply_grouped_delta: unknown op {op!r}")
    return np.asarray(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0))


# ---------------------------------------------------------------------------
# Lagged diff: sort by time_col, compute x - x.shift(period)
# ---------------------------------------------------------------------------


def lagged_diff_features(
    X: pd.DataFrame,
    time_col: str,
    value_cols: Sequence[str],
    periods: Sequence[int] = (1, 2),
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Emit ``lagged_diff_{value_col}__period{p}`` for each (value_col, p).

    The frame is sorted by ``time_col`` (stable mergesort) before the lag is
    computed; the output is reordered back to the input row order so the
    appended columns line up with X row-by-row.

    Recipes store ``{time_col, value_col, period}``; replay reproduces the
    sort + lag from X alone (no fit-time state needed -- the operation is
    a pure function of the test frame).
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

    # Sort by time_col; preserve a permutation back to the input order.
    sort_idx = np.argsort(X[time_col].to_numpy(), kind="mergesort")
    inv_perm = np.empty_like(sort_idx)
    inv_perm[sort_idx] = np.arange(len(X))

    encoded: dict[str, np.ndarray] = {}
    recipes: dict[str, dict] = {}
    for value_col in value_cols:
        x = np.asarray(X[value_col].to_numpy(), dtype=np.float64)
        x_sorted = x[sort_idx]
        for p in periods:
            diff_sorted = np.empty_like(x_sorted)
            diff_sorted[:p] = 0.0
            diff_sorted[p:] = x_sorted[p:] - x_sorted[:-p]
            diff = diff_sorted[inv_perm]
            diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
            name = engineered_name_lagged_diff(value_col, p)
            encoded[name] = diff
            recipes[name] = {
                "time_col": time_col,
                "value_col": value_col,
                "period": int(p),
            }
    enc_df = pd.DataFrame(encoded, index=X.index)
    return enc_df, recipes


def apply_lagged_diff(X_test: pd.DataFrame, recipe: dict) -> np.ndarray:
    """Replay a fitted lagged-diff recipe on new data: resort ``X_test`` by ``time_col``, compute the ``period``-step diff, then unsort back to input row order (pure function of X_test alone, no fit-time state needed)."""
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"apply_lagged_diff: X_test must be a DataFrame; got " f"{type(X_test).__name__}")
    time_col = recipe["time_col"]
    value_col = recipe["value_col"]
    period = int(recipe["period"])
    if time_col not in X_test.columns or value_col not in X_test.columns:
        raise KeyError(f"apply_lagged_diff: missing column(s) {time_col!r}/{value_col!r} " f"from X_test")
    sort_idx = np.argsort(X_test[time_col].to_numpy(), kind="mergesort")
    inv_perm = np.empty_like(sort_idx)
    inv_perm[sort_idx] = np.arange(len(X_test))
    x = np.asarray(X_test[value_col].to_numpy(), dtype=np.float64)
    x_sorted = x[sort_idx]
    diff_sorted = np.empty_like(x_sorted)
    diff_sorted[:period] = 0.0
    diff_sorted[period:] = x_sorted[period:] - x_sorted[:-period]
    out = diff_sorted[inv_perm]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


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


def pairwise_ratio_with_recipes(
    X: pd.DataFrame,
    *,
    cols: Optional[Sequence[str]] = None,
    eps: float = 1e-9,
    mi_gate: bool = False,
    mi_gate_top_k: Optional[int] = None,
    y: Optional[np.ndarray] = None,
    reject_sink: Optional[Callable[..., None]] = None,
):
    """Append ratio columns to X and emit one recipe per accepted pair.

    ``mi_gate=True`` (with ``y``) applies the Tier-1 local MI floor (Layer 91)
    over the O(p^2) ordered-pair ratio pool -- the most explosion-prone L38
    emitter.
    """
    from .engineered_recipes import build_pairwise_ratio_recipe

    if not cols:
        return X.copy(), [], []
    cols = [c for c in cols if c in X.columns]
    if len(cols) < 2:
        return X.copy(), [], []
    enc_df, accepted = pairwise_ratio_features(X, cols, eps=float(eps))
    if not accepted:
        return X.copy(), [], []
    enc_df, accepted = _gate_ratio_pairs(
        enc_df, accepted, engineered_name_ratio, y, X, mi_gate, mi_gate_top_k,
        reject_sink=reject_sink,
    )
    if not accepted:
        return X.copy(), [], []
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


def pairwise_log_ratio_with_recipes(
    X: pd.DataFrame,
    *,
    cols: Optional[Sequence[str]] = None,
    eps: float = 1e-9,
    mi_gate: bool = False,
    mi_gate_top_k: Optional[int] = None,
    y: Optional[np.ndarray] = None,
    reject_sink: Optional[Callable[..., None]] = None,
):
    """Append log-ratio columns and emit one recipe per accepted pair.

    ``mi_gate=True`` (with ``y``) applies the Tier-1 local MI floor (Layer 91).
    """
    from .engineered_recipes import build_pairwise_ratio_recipe

    if not cols:
        return X.copy(), [], []
    cols = [c for c in cols if c in X.columns]
    if len(cols) < 2:
        return X.copy(), [], []
    enc_df, accepted = pairwise_log_ratio_features(X, cols, eps=float(eps))
    if not accepted:
        return X.copy(), [], []
    enc_df, accepted = _gate_ratio_pairs(
        enc_df, accepted, engineered_name_log_ratio, y, X, mi_gate, mi_gate_top_k,
        reject_sink=reject_sink,
    )
    if not accepted:
        return X.copy(), [], []
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


def grouped_delta_with_recipes(
    X: pd.DataFrame,
    *,
    group_col: Optional[str] = None,
    num_cols: Optional[Sequence[str]] = None,
    mi_gate: bool = False,
    mi_gate_top_k: Optional[int] = None,
    y: Optional[np.ndarray] = None,
    reject_sink: Optional[Callable[..., None]] = None,
):
    """Append grouped-delta + grouped-zscore columns and emit one recipe per
    engineered column.

    ``mi_gate=True`` (with ``y``) applies the Tier-1 local MI floor (Layer 91)
    over the 2*|num_cols| grouped-stat pool.
    """
    from .engineered_recipes import build_grouped_delta_recipe

    if not group_col or group_col not in X.columns or not num_cols:
        return X.copy(), [], []
    num_cols = [c for c in num_cols if c in X.columns and c != group_col]
    if not num_cols:
        return X.copy(), [], []
    enc_df, raw_recipes = grouped_delta_features(X, group_col, num_cols)
    if enc_df.empty:
        return X.copy(), [], []
    if mi_gate and y is not None:
        from ._unified_fe_gate import local_mi_gate

        keep = set(local_mi_gate(enc_df, y, raw_X=X, top_k=mi_gate_top_k, reject_sink=reject_sink))
        if not keep:
            return X.copy(), [], []
        enc_df = enc_df[[c for c in enc_df.columns if c in keep]]
    X_aug = pd.concat([X, enc_df], axis=1)
    appended = list(enc_df.columns)
    recipes = [build_grouped_delta_recipe(name=name, **raw_recipes[name]) for name in appended]
    return X_aug, appended, recipes


def lagged_diff_with_recipes(
    X: pd.DataFrame,
    *,
    time_col: Optional[str] = None,
    value_cols: Optional[Sequence[str]] = None,
    periods: Sequence[int] = (1, 2),
    mi_gate: bool = False,
    mi_gate_top_k: Optional[int] = None,
    y: Optional[np.ndarray] = None,
    reject_sink: Optional[Callable[..., None]] = None,
):
    """Append lagged-diff columns and emit one recipe per (value_col, period).

    ``mi_gate=True`` (with ``y``) applies the Tier-1 local MI floor (Layer 91)
    over the |value_cols| * |periods| lag pool.
    """
    from .engineered_recipes import build_lagged_diff_recipe

    if not time_col or time_col not in X.columns or not value_cols:
        return X.copy(), [], []
    value_cols = [c for c in value_cols if c in X.columns and c != time_col]
    if not value_cols:
        return X.copy(), [], []
    periods = tuple(int(p) for p in periods if int(p) >= 1)
    if not periods:
        return X.copy(), [], []
    enc_df, raw_recipes = lagged_diff_features(X, time_col, value_cols, periods)
    if enc_df.empty:
        return X.copy(), [], []
    if mi_gate and y is not None:
        from ._unified_fe_gate import local_mi_gate

        keep = set(local_mi_gate(enc_df, y, raw_X=X, top_k=mi_gate_top_k, reject_sink=reject_sink))
        if not keep:
            return X.copy(), [], []
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
            return pd.DataFrame({a: X[a].to_numpy(), b: X[b].to_numpy()})
    except ImportError:
        pass
    if isinstance(X, np.ndarray) and X.dtype.names is not None:
        return pd.DataFrame({a: X[a], b: X[b]})
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


def _coerce_X_for_grouped_delta(X, group_col: str, num_col: str, recipe_name: str) -> pd.DataFrame:
    """Extract ``group_col``/``num_col`` from an arbitrary carrier (pandas, polars, or structured ndarray) as a pandas DataFrame for ``apply_grouped_delta`` at recipe-replay time."""
    if isinstance(X, pd.DataFrame):
        return X
    try:
        import polars as _pl
        if isinstance(X, _pl.DataFrame):
            return pd.DataFrame({
                group_col: X[group_col].to_numpy(),
                num_col: X[num_col].to_numpy(),
            })
    except ImportError:
        pass
    if isinstance(X, np.ndarray) and X.dtype.names is not None:
        return pd.DataFrame({group_col: X[group_col], num_col: X[num_col]})
    raise TypeError(f"recipe '{recipe_name}': cannot extract {group_col!r}/{num_col!r} " f"from X of type {type(X).__name__}")


def _apply_grouped_delta_recipe(recipe, X) -> np.ndarray:
    """Recipe-apply adapter (consumed by ``engineered_recipes.apply_recipe``): rebuilds the ``apply_grouped_delta`` recipe dict from the frozen ``recipe.extra`` payload and replays it on ``X``."""
    group_col = str(recipe.extra["group_col"])
    num_col = str(recipe.extra["num_col"])
    X_view = _coerce_X_for_grouped_delta(X, group_col, num_col, recipe.name)
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


def _coerce_X_for_lagged_diff(X, time_col: str, value_col: str, recipe_name: str) -> pd.DataFrame:
    """Extract ``time_col``/``value_col`` from an arbitrary carrier (pandas, polars, or structured ndarray) as a pandas DataFrame for ``apply_lagged_diff`` at recipe-replay time."""
    if isinstance(X, pd.DataFrame):
        return X
    try:
        import polars as _pl
        if isinstance(X, _pl.DataFrame):
            return pd.DataFrame({
                time_col: X[time_col].to_numpy(),
                value_col: X[value_col].to_numpy(),
            })
    except ImportError:
        pass
    if isinstance(X, np.ndarray) and X.dtype.names is not None:
        return pd.DataFrame({time_col: X[time_col], value_col: X[value_col]})
    raise TypeError(f"recipe '{recipe_name}': cannot extract {time_col!r}/{value_col!r} " f"from X of type {type(X).__name__}")


def _apply_lagged_diff_recipe(recipe, X) -> np.ndarray:
    """Recipe-apply adapter (consumed by ``engineered_recipes.apply_recipe``): rebuilds the ``apply_lagged_diff`` recipe dict from the frozen ``recipe.extra`` payload and replays it on ``X``."""
    time_col = str(recipe.extra["time_col"])
    value_col = str(recipe.extra["value_col"])
    period = int(recipe.extra["period"])
    X_view = _coerce_X_for_lagged_diff(X, time_col, value_col, recipe.name)
    return apply_lagged_diff(
        X_view, {"time_col": time_col, "value_col": value_col, "period": period},
    )
