"""Composite-key grouped multi-stat aggregator with CMI gate (Layer 93, 2026-06-01).

Extension of Layer 87 (``_grouped_agg_fe``) from SINGLE group columns to
COMPOSITE (multi-column) group keys. Real-world aggregations key on more than
one column at once - ``groupby([region, month])``, ``groupby([store,
category])`` - and the interaction is frequently where the signal lives: the
per-(region, month) mean of a value can carry information that neither the
per-region mean nor the per-month mean exposes on its own (a seasonal effect
that differs by region, say).

The mechanics reuse Layer 87 wholesale: a composite key ``(c1, c2, ...)`` is
factorized into a single integer-coded group via a string join of the per-row
tuple, and the existing per-group stat / z-within / ratio-to-group logic runs
against that synthetic group column. Each surviving aggregate is stored as a
``composite_group_agg`` recipe carrying the ORDERED tuple of group columns plus
the composite-key -> stat lookup. Replay reads only X: each test row rebuilds
its composite key the same way and maps it through the stored lookup; an unseen
composite key falls back to the fit-time global statistic. No y reference is
captured at fit, so ``transform`` is leakage-free by construction.

Cardinality guard (Layer 29 lesson)
------------------------------------

A composite key explodes cardinality multiplicatively: ``groupby([a, b])`` with
100 distinct ``a`` and 100 distinct ``b`` can reach 10k cells. When the number
of distinct composite cells exceeds ``max_card_frac * n`` (default 0.5) the key
is REFUSED - every group would hold ~1 row, the per-group statistic would just
re-encode the row's own value (target-leakage-shaped overfitting), and the
broadcast would carry no generalisable signal. Refused keys emit nothing.
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from ._group_stats_njit import group_mean_std_count

from ._fe_usability_signal import _crit_np_dtype
from ._grouped_coerce_shared import auto_detect_num_cols_plain as _auto_detect_num_cols
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .engineered_recipes import EngineeredRecipe

logger = logging.getLogger(__name__)

__all__ = [
    "COMPOSITE_STAT_NAMES",
    "engineered_name_composite_agg",
    "engineered_name_composite_z",
    "engineered_name_composite_ratio",
    "composite_key_label",
    "build_composite_keys",
    "generate_composite_group_agg_features",
    "composite_group_agg_with_recipes",
    "hybrid_composite_group_agg_fe",
]

# The default broadcast stats for composite keys. mean / std / count are the
# three that matter most on sparse composite cells (count is the cell support,
# a strong signal on its own when cell occupancy is informative).
COMPOSITE_STAT_NAMES = ("mean", "std", "count")

# Stat names understood by the per-group computation.
_VALID_STATS = ("mean", "std", "min", "max", "median", "skew", "nunique", "count")

# Stats whose per-group lookup doubles as the residual base (mean / std).
_RESIDUAL_STATS = ("mean", "std")

def composite_key_label(group_cols: Sequence[str]) -> str:
    """Human-readable composite-key label, e.g. ``"region,month"``."""
    return ",".join(str(c) for c in group_cols)

def engineered_name_composite_agg(
    num_col: str, group_cols: Sequence[str], stat: str,
) -> str:
    """Column name for a per-composite-key ``stat`` broadcast of ``num_col``, e.g. ``"cgrpagg_mean(x|region,month)"``."""
    return f"cgrpagg_{stat}({num_col}|{composite_key_label(group_cols)})"

def engineered_name_composite_z(num_col: str, group_cols: Sequence[str]) -> str:
    """Column name for the z-score-within-composite-group residual of ``num_col``."""
    return f"cgrpz({num_col}|{composite_key_label(group_cols)})"

def engineered_name_composite_ratio(num_col: str, group_cols: Sequence[str]) -> str:
    """Column name for the ratio-to-composite-group-mean residual of ``num_col``."""
    return f"cgrpratio({num_col}|{composite_key_label(group_cols)})"

# ---------------------------------------------------------------------------
# Composite-key construction
# ---------------------------------------------------------------------------

def build_composite_keys(X: pd.DataFrame, group_cols: Sequence[str]) -> np.ndarray:
    """Build a single string-coded composite key per row from ``group_cols``.

    Each row's key is the ordered ``str(c1)\\x1f str(c2) ...`` join (a unit-
    separator byte that never appears in normal categorical labels keeps the
    join unambiguous). The result is an object ndarray usable as a group key by
    the Layer 87 per-group machinery and as a dict key in the replay lookup.
    """
    from ._internals import canonical_group_token, canonical_group_tokens

    parts = []
    for c in group_cols:
        ser = X[c]
        # Each part is canonicalised per UNIQUE value so an integral int/float
        # collapses to the same token (``1`` and ``1.0`` -> ``'1'``): a fit-int /
        # predict-float dtype drift on any component column would otherwise
        # change the whole composite key and miss every per-cell lookup. The
        # per-unique map keeps the low-cardinality hot path cheap. Nulls keep the
        # explicit branch (None -> "" while NaN -> the canonical 'nan').
        if not ser.isna().any():
            arr = ser.to_numpy()
            try:
                uniq, inv = np.unique(arr, return_inverse=True)
                toks = canonical_group_tokens(uniq)
                parts.append(toks[np.asarray(inv).reshape(-1)])
            except (TypeError, ValueError):
                # Unorderable mixed-type object array: per-value canonical.
                parts.append(ser.astype(object).map(canonical_group_token).to_numpy())
        else:
            arr = ser.to_numpy()
            if arr.dtype.kind == "f":
                # Float column with NaN: a float array's missing is NaN (never Python None), and every NaN maps to the
                # canonical 'nan' token whether np.unique merges NaNs or leaves them distinct - so the per-unique path
                # is bit-identical to the per-row map AND runs canonical_group_token per-unique instead of per-row
                # (~13x at 100k / few-hundred uniques; canonical_group_token was the fit's #1 tottime at 5.77M calls).
                uniq, inv = np.unique(arr, return_inverse=True)
                toks = canonical_group_tokens(uniq)
                parts.append(toks[np.asarray(inv).reshape(-1)])
            else:
                parts.append(ser.astype(object).map(lambda v: "" if v is None else canonical_group_token(v)).to_numpy())
    if not parts:
        return np.empty(len(X), dtype=object)
    # \x1f = ASCII unit separator; vanishingly unlikely inside real labels.
    sep = "\x1f"
    out = parts[0].astype(object)
    if len(parts) == 1:
        return np.asarray(out, dtype=object)
    # numpy object-array elementwise string concat (``out + sep + p``) instead
    # of the per-row ``[sep.join(t) for t in zip(*parts)]`` listcomp: same
    # result, ~4% faster, and no Python-level zip/join per row.
    keys = out
    for p in parts[1:]:
        keys = keys + sep + p
    return np.asarray(keys, dtype=object)

def _global_value_for_stat(x: np.ndarray, stat: str) -> float:
    """Global fallback statistic for unseen composite cells at replay time."""
    from ._agg_stat_helpers import global_value_for_stat

    return global_value_for_stat(stat, x, "composite_group_agg")

def _ratio_denominator(per_row_mean: np.ndarray, global_mean: float, global_std: float) -> np.ndarray:
    """Per-row group mean with near-zero cells set to NaN; "near zero" is 1e-12 of the column scale, capped at the absolute 1e-12 for scale >= 1."""
    scale = max(abs(global_mean), global_std if global_std > 0.0 else 1.0)
    tiny = 1e-12 * min(1.0, scale)
    return np.where(np.abs(per_row_mean) > tiny, per_row_mean, np.nan)

def _agg_func_for_stat(stat: str) -> str:
    """Pandas groupby-agg name for ``stat``; this module's set includes ``count``."""
    from ._agg_stat_helpers import agg_func_for_stat

    return agg_func_for_stat(stat, _VALID_STATS, "composite_group_agg")

def _unique_inverse(keys: np.ndarray):
    """``np.unique(keys, return_inverse=True)`` with a 1-D inverse, or
    ``(None, None)`` when the key array is not orderable (mixed-type object).
    Computed ONCE per key array and reused across every stat / residual so the
    O(n log n) argsort inside ``np.unique`` is not repaid per column."""
    keys = np.asarray(keys, dtype=object)
    try:
        uniq, inverse = np.unique(keys, return_inverse=True)
        return uniq, np.asarray(inverse).reshape(-1)
    except (TypeError, ValueError) as _exc:
        logger.debug("_unique_inverse fell back to its default after %s", type(_exc).__name__)
        return None, None

def _broadcast_lookup(
    keys: np.ndarray, lookup: dict, global_value: float,
    *, uniq=None, inverse=None,
) -> np.ndarray:
    """Map each row's composite key through ``lookup`` (str-keyed), unseen ->
    global. Resolved once per UNIQUE key then broadcast via the inverse index.

    ``uniq`` / ``inverse`` may be supplied (from :func:`_unique_inverse`) so the
    de-duplication is shared across every stat for the same key array.
    """
    keys = np.asarray(keys, dtype=object)
    if uniq is None or inverse is None:
        uniq, inverse = _unique_inverse(keys)
    if uniq is not None and inverse is not None:
        uniq_vals = np.array(
            [lookup.get(str(_k), global_value) for _k in uniq],
            # Intentional float64: group statistics come from the f64 recipe lookup; the output column stays f64 to match the stored values.
            dtype=np.float64,
        )
        out = uniq_vals[inverse]
    else:
        out = np.array(
            [lookup.get(str(_k), global_value) for _k in keys],
            # Intentional float64: group statistics come from the f64 recipe lookup; the output column stays f64 to match the stored values.
            dtype=np.float64,
        )
    return np.asarray(np.nan_to_num(
        out, nan=global_value, posinf=global_value, neginf=global_value,
    ))

# ---------------------------------------------------------------------------
# Per-composite-key statistic computation
# ---------------------------------------------------------------------------

def composite_cardinality_ok(
    n_distinct: int, n_rows: int, max_card_frac: float = 0.5,
) -> bool:
    """L29 guard: refuse a composite key whose distinct-cell count exceeds
    ``max_card_frac * n_rows`` (every cell would hold ~1 row -> the per-group
    stat re-encodes the row's own value, no generalisable signal)."""
    if n_rows <= 0:
        return False
    return n_distinct <= int(max_card_frac * n_rows)

def _group_lookup(series: pd.Series, label: Callable[[Any], str], zero_nonfinite: bool) -> dict:
    """``{label(group): float(value)}`` of a per-group statistic Series. One array conversion and one ``zip`` instead of ``Series.items()`` with a ``float()`` per group;
    ``zero_nonfinite`` replaces NaN / inf by 0.0 (the std and aggregate lookups), exactly as ``float(v) if np.isfinite(v) else 0.0`` did."""
    values = series.to_numpy(dtype=np.float64)  # float32 -> float64 is exact, so tolist() gives the same Python floats as float(v)
    if zero_nonfinite:
        values = np.where(np.isfinite(values), values, 0.0)
    return dict(zip(map(label, series.index.tolist()), values.tolist()))


def _array_lookup(labels: list, values: np.ndarray, zero_nonfinite: bool) -> dict:
    """:func:`_group_lookup` for a per-group-code array: ``{labels[code]: value}``."""
    vals = np.asarray(values, dtype=np.float64)  # float32 -> float64 is exact
    if zero_nonfinite:
        vals = np.where(np.isfinite(vals), vals, 0.0)
    return dict(zip(labels, vals.tolist()))


def _emit_num_col_features(
    X: pd.DataFrame,
    num_col: str,
    keys: Any,
    key_uniq: Any,
    key_inverse: Any,
    group_cols: tuple,
    stats: Sequence[str],
    encoded: dict,
    raw_recipes: dict,
) -> None:
    """Compute one numeric column's per-composite-group broadcasts plus its z-within-group and ratio residuals into ``encoded`` / ``raw_recipes``."""
    x = np.asarray(X[num_col].to_numpy(), dtype=_crit_np_dtype())
    # Group on the integer codes of the key when the caller has them (the key strings are then hashed once per column set, not once per numeric column and statistic);
    # the recipe lookups stay keyed by the key string, taken from the unique keys.
    label: Callable[[Any], str]
    kernel = key_uniq is not None and key_inverse is not None and x.dtype in (np.float32, np.float64)
    grouped = None
    if key_uniq is not None and key_inverse is not None:
        key_labels = [str(u) for u in key_uniq]  # the key string of group code c is key_labels[c]
        label = key_labels.__getitem__
    else:
        label = str
    if kernel:
        # mean / std / count over the group codes in one pass, bit-equal to the pandas groupby of the other branch (see ``_group_stats_njit``)
        mean_arr, std_arr, count_arr = group_mean_std_count(np.ascontiguousarray(key_inverse, dtype=np.int64), np.ascontiguousarray(x), len(key_labels))
        lookup_mean = _array_lookup(key_labels, mean_arr, zero_nonfinite=False)
        lookup_std = _array_lookup(key_labels, std_arr, zero_nonfinite=True)
    else:
        if key_uniq is not None and key_inverse is not None:
            tmp = pd.DataFrame({"_g": np.asarray(key_inverse), "_v": x}, index=X.index)
        else:
            tmp = pd.DataFrame({"_g": keys, "_v": x}, index=X.index)
        grouped = tmp.groupby("_g", observed=True, sort=False)["_v"]
        mean_series = grouped.mean()
        std_series = grouped.std(ddof=1)
        lookup_mean = _group_lookup(mean_series, label, zero_nonfinite=False)
        lookup_std = _group_lookup(std_series, label, zero_nonfinite=True)
    global_mean = _global_value_for_stat(x, "mean")
    global_std = _global_value_for_stat(x, "std")

    for stat in stats:
        # mean / std are already materialised above for the z / ratio residuals; reuse them instead of running the groupby a second time
        if kernel and stat in ("mean", "std", "count"):
            agg_values = {"mean": mean_arr, "std": std_arr, "count": count_arr}[stat]
            lookup = _array_lookup(key_labels, agg_values, zero_nonfinite=True)
        else:
            if grouped is None:  # a statistic the kernel does not cover (min / max / median / skew / nunique)
                tmp = pd.DataFrame({"_g": np.asarray(key_inverse), "_v": x}, index=X.index)
                grouped = tmp.groupby("_g", observed=True, sort=False)["_v"]
            if stat == "mean" and not kernel:
                agg_series = mean_series
            elif stat == "std" and not kernel:
                agg_series = std_series
            else:
                agg_series = grouped.agg(_agg_func_for_stat(stat))
            lookup = _group_lookup(agg_series, label, zero_nonfinite=True)
        if stat in ("count", "nunique"):
            # The whole-population fallback
            # (finite.size / np.unique(finite).size) used to emit a WILDLY out-of-distribution
            # value for an unseen composite key - confirmed by direct execution: real per-cell
            # count values were 14-25 (n=2000) vs a global fallback of 2000 (the entire training
            # set, ~100x). Every OTHER stat's fallback stays on the SAME SCALE as a per-cell value;
            # use the median PER-CELL count/nunique across fit-time cells instead.
            cell_values = count_arr if (kernel and stat == "count") else agg_series.to_numpy()
            global_value = float(np.median(cell_values)) if len(cell_values) else 0.0
        else:
            global_value = _global_value_for_stat(x, stat)
        broadcast = _broadcast_lookup(
            keys, lookup, global_value,
            uniq=key_uniq, inverse=key_inverse,
        )
        name = engineered_name_composite_agg(num_col, group_cols, stat)
        encoded[name] = broadcast
        raw_recipes[name] = {
            "group_cols": tuple(group_cols),
            "num_col": num_col,
            "stat": stat,
            "op": "broadcast",
            "group_lookup_dict": lookup,
            "global_value": global_value,
            "lookup_mean": lookup_mean,
            "lookup_std": lookup_std,
            "global_mean": global_mean,
            "global_std": global_std,
        }

    # ---- z-within-composite-group residual ----
    per_row_mean = _broadcast_lookup(
        keys, lookup_mean, global_mean,
        uniq=key_uniq, inverse=key_inverse,
    )
    per_row_std = _broadcast_lookup(
        keys, lookup_std, global_std,
        uniq=key_uniq, inverse=key_inverse,
    )
    per_row_std = np.where(per_row_std > 0.0, per_row_std, 1.0)
    zscore = np.nan_to_num(
        (x - per_row_mean) / per_row_std, nan=0.0, posinf=0.0, neginf=0.0,
    )
    z_name = engineered_name_composite_z(num_col, group_cols)
    encoded[z_name] = zscore
    raw_recipes[z_name] = {
        "group_cols": tuple(group_cols),
        "num_col": num_col,
        "stat": "mean",
        "op": "z_within",
        "group_lookup_dict": lookup_mean,
        "global_value": global_mean,
        "lookup_mean": lookup_mean,
        "lookup_std": lookup_std,
        "global_mean": global_mean,
        "global_std": global_std if global_std > 0.0 else 1.0,
    }

    # ---- ratio-to-composite-group residual ----
    denom = _ratio_denominator(per_row_mean, global_mean, global_std)
    ratio = np.nan_to_num(x / denom, nan=1.0, posinf=1.0, neginf=1.0)
    r_name = engineered_name_composite_ratio(num_col, group_cols)
    encoded[r_name] = ratio
    raw_recipes[r_name] = {
        "group_cols": tuple(group_cols),
        "num_col": num_col,
        "stat": "mean",
        "op": "ratio",
        "group_lookup_dict": lookup_mean,
        "global_value": global_mean,
        "lookup_mean": lookup_mean,
        "lookup_std": lookup_std,
        "global_mean": global_mean,
        "global_std": global_std if global_std > 0.0 else 1.0,
    }


def generate_composite_group_agg_features(
    X: pd.DataFrame,
    group_col_sets: Sequence[Sequence[str]],
    num_cols: Sequence[str],
    stats: Sequence[str] = COMPOSITE_STAT_NAMES,
    *,
    max_card_frac: float = 0.5,
    _precomputed_keys: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Compute per-(composite_key, num_col, stat) broadcasts plus the
    z-within-composite-group and ratio-to-composite-group residuals.

    Returns ``(enc_df, raw_recipes)`` where ``raw_recipes[name]`` is the kwargs
    dict for :func:`engineered_recipes.build_composite_group_agg_recipe`.

    A composite key whose distinct-cell count exceeds ``max_card_frac * n`` is
    refused (Layer 29 guard): it emits no columns.

    ``_precomputed_keys``, when supplied by an auto-detect caller (see
    :func:`auto_detect_key_sets`'s ``_key_cache``), maps ``group_cols -> (keys, key_uniq, key_inverse,
    n_distinct)`` for combos already built while ranking candidates - skips rebuilding + re-hashing the
    identical composite key for every surviving combo.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"generate_composite_group_agg_features: X must be a pandas " f"DataFrame; got {type(X).__name__}")
    stats = [s for s in stats if s in _VALID_STATS]
    encoded: dict[str, np.ndarray] = {}
    raw_recipes: dict[str, dict] = {}
    if not stats:
        return pd.DataFrame(index=X.index), raw_recipes

    n_rows = len(X)
    # Normalise + validate the requested key sets.
    norm_sets: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for gset in group_col_sets:
        cols = tuple(c for c in gset if c in X.columns)
        # A "composite" key needs >= 2 columns; single columns are Layer 87.
        if len(cols) < 2:
            continue
        if cols in seen:
            continue
        seen.add(cols)
        norm_sets.append(cols)
    if not norm_sets:
        return pd.DataFrame(index=X.index), raw_recipes

    for group_cols in norm_sets:
        _pre = _precomputed_keys.get(group_cols) if _precomputed_keys is not None else None
        if _pre is not None:
            keys, key_uniq, key_inverse, n_distinct = _pre
        else:
            keys = build_composite_keys(X, group_cols)
            # De-duplicate the composite keys ONCE per key-set; reused by every
            # num_col / stat / residual broadcast below (the np.unique argsort was
            # the dominant cProfile hotspot before this hoist), and for the
            # cardinality guard itself (avoids a second full np.unique pass).
            key_uniq, key_inverse = _unique_inverse(keys)
            n_distinct = int(key_uniq.size) if key_uniq is not None else int(np.unique(keys).size)
        if not composite_cardinality_ok(n_distinct, n_rows, max_card_frac):
            logger.debug(
                "composite_group_agg: refusing key %s (cardinality %d > %.2f*n=%d)",
                composite_key_label(group_cols), n_distinct,
                max_card_frac, int(max_card_frac * n_rows),
            )
            continue

        cur_num_cols = [c for c in num_cols if c in X.columns and c not in set(group_cols) and pd.api.types.is_numeric_dtype(X[c])]
        # FUTURE (perf): this still rebuilds a temp frame + re-hashes "_g" (already known via key_uniq/key_inverse
        # above) for every num_col. A bincount-style mean/std accumulation keyed by key_inverse would skip the
        # pandas groupby entirely, mirroring _binned_numeric_agg_fe's njit raw-moment kernel - deferred because
        # replicating pandas' NaN-skipping mean/Welford-based std exactly (not just mathematically) needs the
        # same empirical bit-identity verification as the K-fold subtraction rewrite, and this file's num_cols
        # loop is uncapped (unlike the capped max_pairs paths that make that verification cheap to scope).
        for num_col in cur_num_cols:
            _emit_num_col_features(X, num_col, keys, key_uniq, key_inverse, group_cols, stats, encoded, raw_recipes)

    enc_df = pd.DataFrame(encoded, index=X.index)
    return enc_df, raw_recipes

# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def apply_composite_group_agg(X_test: pd.DataFrame, recipe: dict) -> np.ndarray:
    """Replay one composite-group-agg column from the stored recipe dict,
    reading only ``X_test``. ``op`` selects the variant (``broadcast`` /
    ``z_within`` / ``ratio``)."""
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"apply_composite_group_agg: X_test must be a DataFrame; got " f"{type(X_test).__name__}")
    group_cols = list(recipe["group_cols"])
    num_col = recipe["num_col"]
    op = recipe.get("op", "broadcast")
    missing = [c for c in group_cols if c not in X_test.columns]
    if missing or num_col not in X_test.columns:
        raise KeyError(f"apply_composite_group_agg: missing column(s) {missing or num_col!r} " f"from X_test")
    keys = build_composite_keys(X_test, group_cols)
    x = np.asarray(X_test[num_col].to_numpy(), dtype=_crit_np_dtype())

    if op == "broadcast":
        lookup = dict(recipe["group_lookup_dict"])
        global_value = float(recipe["global_value"])
        return _broadcast_lookup(keys, lookup, global_value)

    lookup_mean = dict(recipe["lookup_mean"])
    global_mean = float(recipe["global_mean"])
    per_row_mean = _broadcast_lookup(keys, lookup_mean, global_mean)

    if op == "z_within":
        lookup_std = dict(recipe["lookup_std"])
        global_std = float(recipe.get("global_std", 1.0)) or 1.0
        per_row_std = _broadcast_lookup(keys, lookup_std, global_std)
        per_row_std = np.where(per_row_std > 0.0, per_row_std, 1.0)
        return np.asarray(np.nan_to_num(
            (x - per_row_mean) / per_row_std, nan=0.0, posinf=0.0, neginf=0.0,
        ))
    if op == "ratio":
        denom = _ratio_denominator(per_row_mean, global_mean, float(recipe.get("global_std", 1.0)))
        return np.asarray(np.nan_to_num(x / denom, nan=1.0, posinf=1.0, neginf=1.0))
    raise ValueError(f"apply_composite_group_agg: unknown op {op!r}")

def _coerce_X_for_composite_agg(X, group_cols, num_col, recipe_name: str) -> pd.DataFrame:
    """Extract only the columns a replay needs (``group_cols`` + ``num_col``) as a pandas frame, accepting polars or a structured ndarray so recipe replay is format-agnostic without materialising the full input."""
    if isinstance(X, pd.DataFrame):
        return X
    try:
        import polars as _pl
        if isinstance(X, _pl.DataFrame):
            cols = {c: X[c].to_numpy() for c in group_cols}
            cols[num_col] = X[num_col].to_numpy()
            return pd.DataFrame(cols)
    except ImportError:
        pass
    if isinstance(X, np.ndarray) and X.dtype.names is not None:
        cols = {c: X[c] for c in group_cols}
        cols[num_col] = X[num_col]
        return pd.DataFrame(cols)
    raise TypeError(f"recipe '{recipe_name}': cannot extract {group_cols}/{num_col!r} " f"from X of type {type(X).__name__}")

def _apply_composite_group_agg_recipe(recipe, X) -> np.ndarray:
    """Adapter consumed by ``engineered_recipes.apply_recipe``."""
    group_cols = [str(c) for c in recipe.extra["group_cols"]]
    num_col = str(recipe.extra["num_col"])
    X_view = _coerce_X_for_composite_agg(X, group_cols, num_col, recipe.name)
    return apply_composite_group_agg(
        X_view,
        {
            "group_cols": group_cols,
            "num_col": num_col,
            "stat": str(recipe.extra.get("stat", "mean")),
            "op": str(recipe.extra.get("op", "broadcast")),
            "group_lookup_dict": dict(recipe.extra.get("group_lookup_dict", {})),
            "global_value": float(recipe.extra.get("global_value", 0.0)),
            "lookup_mean": dict(recipe.extra.get("lookup_mean", {})),
            "lookup_std": dict(recipe.extra.get("lookup_std", {})),
            "global_mean": float(recipe.extra.get("global_mean", 0.0)),
            "global_std": float(recipe.extra.get("global_std", 1.0)),
        },
    )

def composite_group_agg_with_recipes(
    X: pd.DataFrame,
    *,
    group_col_sets: Optional[Sequence[Sequence[str]]] = None,
    num_cols: Optional[Sequence[str]] = None,
    stats: Sequence[str] = COMPOSITE_STAT_NAMES,
    max_card_frac: float = 0.5,
) -> tuple[pd.DataFrame, list[str], list[EngineeredRecipe]]:
    """Append composite-group-agg columns and emit one recipe per column."""
    from .engineered_recipes import build_composite_group_agg_recipe

    if not group_col_sets or not num_cols:
        return X, [], []
    enc_df, raw_recipes = generate_composite_group_agg_features(
        X, group_col_sets, num_cols, stats=stats, max_card_frac=max_card_frac,
    )
    if enc_df.empty:
        return X, [], []
    X_aug = pd.concat([X, enc_df], axis=1)
    appended = list(enc_df.columns)
    recipes = [build_composite_group_agg_recipe(name=name, **raw_recipes[name]) for name in appended]
    return X_aug, appended, recipes

# ---------------------------------------------------------------------------
# Auto-detect composite key sets
# ---------------------------------------------------------------------------

def _auto_detect_group_cols(X: pd.DataFrame, max_cols: int = 12) -> list[str]:
    """Reuse the Layer 87 / composite_auto_detect int-as-cat detector."""
    # This used to try a two-dot
    # `from .._grouped_agg_fe import ...` FIRST (wrong depth - `_grouped_agg_fe.py` is a SIBLING, one dot,
    # not a parent-package member) inside a bare `except Exception`, which always failed and silently fell
    # through to the correct single-dot import below - functionally masked, but dead/misleading code.
    try:
        from ._grouped_coerce_shared import auto_detect_group_cols as _l87_detect
        return list(_l87_detect(X, max_cols=max_cols, caller='composite_group_agg'))
    except Exception as exc:  # nosec B110 - optional dependency import guard
        logger.debug("_auto_detect_group_cols: Layer-87 detector unavailable; falling back to inline heuristic: %r", exc)
    out: list[str] = []
    n = len(X)
    for c in X.columns:
        col = X[c]
        if pd.api.types.is_float_dtype(col):
            continue
        nun = int(col.nunique(dropna=True))
        if 3 <= nun <= min(500, max(3, n // 2)):
            out.append(str(c))
    return out[:max_cols]

def _target_codes(y: Any, n_rows: int) -> Optional[np.ndarray]:
    """Dense class codes of ``y`` (a target of at most 20 distinct values keeps its classes, a continuous one is cut into ten quantile bins); None when ``y`` has the wrong length."""
    arr = np.asarray(y).ravel()
    if len(arr) != n_rows:
        return None
    uniq, inverse = np.unique(arr, return_inverse=True)
    if len(uniq) <= 20:
        codes: np.ndarray = inverse.astype(np.int64)
        return codes
    edges = np.quantile(arr.astype(np.float64), np.linspace(0.0, 1.0, 11)[1:-1])  # Intentional float64: quantile edges of the target
    return np.searchsorted(edges, arr, side="right").astype(np.int64)


def _corrected_key_mi(key_inverse: np.ndarray, n_keys: int, y_codes: np.ndarray) -> float:
    """Mutual information (nats) of a composite key with the target codes minus its chance level ``(K - 1)(C - 1) / (2 n)``."""
    n_classes = int(y_codes.max()) + 1
    n = len(y_codes)
    joint = np.bincount(np.asarray(key_inverse, dtype=np.int64) * n_classes + y_codes, minlength=n_keys * n_classes).reshape(n_keys, n_classes).astype(np.float64)  # Intentional float64: count table for the entropy sums
    pj = joint / n
    pk = pj.sum(axis=1, keepdims=True)
    pc = pj.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(pj > 0, pj * np.log(pj / (pk * pc)), 0.0)
    return float(terms.sum() - (n_keys - 1) * (n_classes - 1) / (2.0 * n))


def auto_detect_key_sets(
    X: pd.DataFrame,
    *,
    max_arity: int = 2,
    max_card_frac: float = 0.5,
    detected_group_cols: Optional[Sequence[str]] = None,
    max_sets: int = 12,
    _key_cache: Optional[dict] = None,
    y: Optional[np.ndarray] = None,
) -> list[tuple[str, ...]]:
    """Enumerate composite key candidates: all r-combinations (2..max_arity) of
    the detected single group columns whose composite cardinality clears the
    L29 guard. Returns ordered tuples: with ``y`` supplied, by the bias-corrected mutual information of the
    composite key with the target (the ``max_sets`` kept are the keys that carry the most about y; the lowest-cardinality
    ones kept before said nothing about relevance, and a wider column pool would have filled the list with ties);
    without ``y``, lowest-cardinality first.

    ``_key_cache``, when supplied, is populated with ``{combo: (keys, key_uniq, key_inverse, n_distinct)}``
    for every candidate combo built here - lets :func:`hybrid_composite_group_agg_fe` thread the already-built
    composite keys into :func:`generate_composite_group_agg_features` for the SURVIVING combos instead of
    rebuilding + re-hashing them from scratch."""
    cols = list(detected_group_cols) if detected_group_cols is not None else _auto_detect_group_cols(X)
    cols = [c for c in cols if c in X.columns]
    n_rows = len(X)
    cand: list[tuple[tuple[str, ...], int]] = []
    y_codes = _target_codes(y, n_rows) if y is not None else None
    relevance: dict = {}
    for r in range(2, max(2, int(max_arity)) + 1):
        for combo in combinations(cols, r):
            keys = build_composite_keys(X, combo)
            key_uniq, key_inverse = _unique_inverse(keys)
            n_distinct = int(key_uniq.size) if key_uniq is not None else int(np.unique(keys).size)
            if composite_cardinality_ok(n_distinct, n_rows, max_card_frac):
                cand.append((tuple(combo), n_distinct))
                if y_codes is not None:
                    relevance[tuple(combo)] = _corrected_key_mi(key_inverse if key_inverse is not None else np.unique(keys, return_inverse=True)[1], n_distinct, y_codes)
                if _key_cache is not None:
                    _key_cache[tuple(combo)] = (keys, key_uniq, key_inverse, n_distinct)
    if relevance:
        cand.sort(key=lambda t: (-relevance[t[0]], t[1]))
    else:
        cand.sort(key=lambda t: t[1])
    return [c for c, _ in cand[:max_sets]]

def hybrid_composite_group_agg_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    group_col_sets: Optional[Sequence[Sequence[str]]] = None,
    num_cols: Optional[Sequence[str]] = None,
    stats: Sequence[str] = COMPOSITE_STAT_NAMES,
    max_arity: int = 2,
    max_card_frac: float = 0.5,
    top_k: int = 10,
    n_bins: int = 10,
    min_cmi: float = 1e-4,
    min_uplift: float = 0.0,
) -> tuple[pd.DataFrame, list[str], list[EngineeredRecipe], pd.DataFrame]:
    """End-to-end composite-group-agg FE pipeline.

    1. Auto-detect ``group_col_sets`` (combinations of detected group columns up
       to ``max_arity`` whose composite cardinality clears the L29 guard) and
       ``num_cols`` (continuous) when not supplied.
    2. Materialise every (composite_key, num, stat) broadcast + z / ratio.
    3. Score by ``CMI(agg; y | base_cols)`` and gate by uplift over the source
       num_col marginal MI; keep the top ``top_k`` survivors.
    4. Append survivors to X; return ``(X_aug, appended, recipes, scores)``.

    ``y`` is consumed ONLY by the CMI scorer / gate; recipes carry no y, so
    transform-time replay is leakage-free.
    """
    from ._grouped_agg_fe import score_grouped_agg_by_cmi_uplift

    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"hybrid_composite_group_agg_fe: X must be a pandas DataFrame; got " f"{type(X).__name__}")
    # Populated by auto_detect_key_sets with the composite keys it already built while ranking candidates;
    # threaded into generate_composite_group_agg_features below so the SURVIVING combos' keys aren't rebuilt
    # from scratch. None for a caller-supplied group_col_sets (nothing was pre-built to reuse).
    _key_cache: Optional[dict] = None
    if group_col_sets is None or len(group_col_sets) == 0:
        _key_cache = {}
        group_col_sets = auto_detect_key_sets(
            X, max_arity=max_arity, max_card_frac=max_card_frac, _key_cache=_key_cache, y=y,
        )
    else:
        group_col_sets = [tuple(c for c in gset if c in X.columns) for gset in group_col_sets]
        group_col_sets = [gset for gset in group_col_sets if len(gset) >= 2]
    if not group_col_sets:
        return X, [], [], pd.DataFrame()

    if num_cols is None or len(num_cols) == 0:
        all_group_cols = sorted({c for gset in group_col_sets for c in gset})
        num_cols = _auto_detect_num_cols(X, all_group_cols)
    else:
        num_cols = [c for c in num_cols if c in X.columns]
    if not num_cols:
        return X, [], [], pd.DataFrame()

    enc_df, raw_recipes = generate_composite_group_agg_features(
        X, group_col_sets, num_cols, stats=stats, max_card_frac=max_card_frac,
        _precomputed_keys=_key_cache,
    )
    if enc_df.empty:
        return X, [], [], pd.DataFrame()

    base_cols = list(num_cols)
    eng_to_source = {name: raw_recipes[name]["num_col"] for name in enc_df.columns}
    scores = score_grouped_agg_by_cmi_uplift(
        X, enc_df, y, base_cols, n_bins=n_bins, eng_to_source=eng_to_source,
    )
    keep = scores[(scores["cmi"] >= float(min_cmi)) & (scores["uplift"] >= float(min_uplift))]
    winners = list(keep["engineered_col"].head(int(top_k)))
    if not winners:
        return X, [], [], scores

    from .engineered_recipes import build_composite_group_agg_recipe

    X_aug = pd.concat([X, enc_df[winners]], axis=1)
    recipes = [build_composite_group_agg_recipe(name=name, **raw_recipes[name]) for name in winners]
    return X_aug, winners, recipes, scores
