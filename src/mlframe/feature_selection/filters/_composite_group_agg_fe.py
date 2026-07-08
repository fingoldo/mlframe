"""Composite-key grouped multi-stat aggregator with CMI gate (Layer 93, 2026-06-01).

Extension of Layer 87 (``_grouped_agg_fe``) from SINGLE group columns to
COMPOSITE (multi-column) group keys. Real-world aggregations key on more than
one column at once -- ``groupby([region, month])``, ``groupby([store,
category])`` -- and the interaction is frequently where the signal lives: the
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
is REFUSED -- every group would hold ~1 row, the per-group statistic would just
re-encode the row's own value (target-leakage-shaped overfitting), and the
broadcast would carry no generalisable signal. Refused keys emit nothing.
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import pandas as pd

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
    return f"cgrpagg_{stat}({num_col}|{composite_key_label(group_cols)})"


def engineered_name_composite_z(num_col: str, group_cols: Sequence[str]) -> str:
    return f"cgrpz({num_col}|{composite_key_label(group_cols)})"


def engineered_name_composite_ratio(num_col: str, group_cols: Sequence[str]) -> str:
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
    from ._internals import canonical_group_token

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
                toks = np.array([canonical_group_token(u) for u in uniq], dtype=object)
                parts.append(toks[np.asarray(inv).reshape(-1)])
            except (TypeError, ValueError):
                # Unorderable mixed-type object array: per-value canonical.
                parts.append(ser.astype(object).map(canonical_group_token).to_numpy())
        else:
            arr = ser.to_numpy()
            if arr.dtype.kind == "f":
                # Float column with NaN: a float array's missing is NaN (never Python None), and every NaN maps to the
                # canonical 'nan' token whether np.unique merges NaNs or leaves them distinct -- so the per-unique path
                # is bit-identical to the per-row map AND runs canonical_group_token per-unique instead of per-row
                # (~13x at 100k / few-hundred uniques; canonical_group_token was the fit's #1 tottime at 5.77M calls).
                uniq, inv = np.unique(arr, return_inverse=True)
                toks = np.array([canonical_group_token(u) for u in uniq], dtype=object)
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
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return 0.0
    if stat == "mean":
        return float(np.mean(finite))
    if stat == "std":
        return float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    if stat == "min":
        return float(np.min(finite))
    if stat == "max":
        return float(np.max(finite))
    if stat == "median":
        return float(np.median(finite))
    if stat == "nunique":
        return float(np.unique(finite).size)
    if stat == "count":
        return float(finite.size)
    if stat == "skew":
        return float(pd.Series(finite).skew()) if finite.size > 2 else 0.0
    raise ValueError(f"composite_group_agg: unknown stat {stat!r}")


def _agg_func_for_stat(stat: str):
    if stat in ("mean", "std", "min", "max", "median", "skew", "nunique", "count"):
        return stat
    raise ValueError(f"composite_group_agg: unknown stat {stat!r}; valid: {_VALID_STATS}")


def _unique_inverse(keys: np.ndarray):
    """``np.unique(keys, return_inverse=True)`` with a 1-D inverse, or
    ``(None, None)`` when the key array is not orderable (mixed-type object).
    Computed ONCE per key array and reused across every stat / residual so the
    O(n log n) argsort inside ``np.unique`` is not repaid per column."""
    keys = np.asarray(keys, dtype=object)
    try:
        uniq, inverse = np.unique(keys, return_inverse=True)
        return uniq, np.asarray(inverse).reshape(-1)
    except (TypeError, ValueError):
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
            [lookup.get(str(_k), global_value) for _k in uniq], dtype=np.float64,
        )
        out = uniq_vals[inverse]
    else:
        out = np.array(
            [lookup.get(str(_k), global_value) for _k in keys], dtype=np.float64,
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


def generate_composite_group_agg_features(
    X: pd.DataFrame,
    group_col_sets: Sequence[Sequence[str]],
    num_cols: Sequence[str],
    stats: Sequence[str] = COMPOSITE_STAT_NAMES,
    *,
    max_card_frac: float = 0.5,
):
    """Compute per-(composite_key, num_col, stat) broadcasts plus the
    z-within-composite-group and ratio-to-composite-group residuals.

    Returns ``(enc_df, raw_recipes)`` where ``raw_recipes[name]`` is the kwargs
    dict for :func:`engineered_recipes.build_composite_group_agg_recipe`.

    A composite key whose distinct-cell count exceeds ``max_card_frac * n`` is
    refused (Layer 29 guard): it emits no columns.
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
        keys = build_composite_keys(X, group_cols)
        n_distinct = int(np.unique(keys).size)
        if not composite_cardinality_ok(n_distinct, n_rows, max_card_frac):
            logger.debug(
                "composite_group_agg: refusing key %s (cardinality %d > %.2f*n=%d)",
                composite_key_label(group_cols), n_distinct,
                max_card_frac, int(max_card_frac * n_rows),
            )
            continue

        cur_num_cols = [c for c in num_cols if c in X.columns and c not in set(group_cols) and pd.api.types.is_numeric_dtype(X[c])]
        # De-duplicate the composite keys ONCE per key-set; reused by every
        # num_col / stat / residual broadcast below (the np.unique argsort was
        # the dominant cProfile hotspot before this hoist).
        key_uniq, key_inverse = _unique_inverse(keys)
        for num_col in cur_num_cols:
            x = np.asarray(X[num_col].to_numpy(), dtype=np.float64)
            tmp = pd.DataFrame({"_g": keys, "_v": x}, index=X.index)
            grouped = tmp.groupby("_g", observed=True, sort=False)["_v"]
            mean_series = grouped.mean()
            std_series = grouped.std(ddof=1)
            lookup_mean = {str(k): float(v) for k, v in mean_series.items()}
            lookup_std = {str(k): (float(v) if np.isfinite(v) else 0.0) for k, v in std_series.items()}
            global_mean = _global_value_for_stat(x, "mean")
            global_std = _global_value_for_stat(x, "std")

            for stat in stats:
                # mean / std are already materialised above for the z / ratio
                # residuals; reuse them instead of re-running the (O(n)) cython
                # groupby a second time. ``grouped.mean()`` == ``grouped.agg("mean")``
                # and ``grouped.std(ddof=1)`` == ``grouped.agg("std")`` bit-for-bit.
                if stat == "mean":
                    agg_series = mean_series
                elif stat == "std":
                    agg_series = std_series
                else:
                    agg_series = grouped.agg(_agg_func_for_stat(stat))
                lookup = {str(k): (float(v) if np.isfinite(v) else 0.0) for k, v in agg_series.items()}
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
            denom = np.where(np.abs(per_row_mean) > 1e-12, per_row_mean, np.nan)
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
    x = np.asarray(X_test[num_col].to_numpy(), dtype=np.float64)

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
        denom = np.where(np.abs(per_row_mean) > 1e-12, per_row_mean, np.nan)
        return np.asarray(np.nan_to_num(x / denom, nan=1.0, posinf=1.0, neginf=1.0))
    raise ValueError(f"apply_composite_group_agg: unknown op {op!r}")


def _coerce_X_for_composite_agg(X, group_cols, num_col, recipe_name: str) -> pd.DataFrame:
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
):
    """Append composite-group-agg columns and emit one recipe per column."""
    from .engineered_recipes import build_composite_group_agg_recipe

    if not group_col_sets or not num_cols:
        return X.copy(), [], []
    enc_df, raw_recipes = generate_composite_group_agg_features(
        X, group_col_sets, num_cols, stats=stats, max_card_frac=max_card_frac,
    )
    if enc_df.empty:
        return X.copy(), [], []
    X_aug = pd.concat([X, enc_df], axis=1)
    appended = list(enc_df.columns)
    recipes = [build_composite_group_agg_recipe(name=name, **raw_recipes[name]) for name in appended]
    return X_aug, appended, recipes


# ---------------------------------------------------------------------------
# Auto-detect composite key sets
# ---------------------------------------------------------------------------


def _auto_detect_group_cols(X: pd.DataFrame, max_cols: int = 6) -> list[str]:
    """Reuse the Layer 87 / composite_auto_detect int-as-cat detector."""
    try:
        from .._grouped_agg_fe import _auto_detect_group_cols as _l87_detect  # type: ignore
    except Exception:
        _l87_detect = None
    if _l87_detect is not None:
        try:
            return list(_l87_detect(X, max_cols=max_cols))
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed in _composite_group_agg_fe.py:503: %s", e)
            pass
    try:
        from ._grouped_agg_fe import _auto_detect_group_cols as _l87_detect2
        return _l87_detect2(X, max_cols=max_cols)
    except Exception:  # nosec B110 - optional dependency import guard
        pass
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


def _auto_detect_num_cols(
    X: pd.DataFrame, group_cols: Sequence[str], max_cols: int = 8,
) -> list[str]:
    group_set = set(group_cols)
    out: list[str] = []
    for c in X.columns:
        if c in group_set:
            continue
        col = X[c]
        if not pd.api.types.is_numeric_dtype(col):
            continue
        if pd.api.types.is_float_dtype(col):
            out.append(str(c))
            continue
        if int(col.nunique(dropna=True)) > 500:
            out.append(str(c))
    return out[:max_cols]


def auto_detect_key_sets(
    X: pd.DataFrame,
    *,
    max_arity: int = 2,
    max_card_frac: float = 0.5,
    detected_group_cols: Optional[Sequence[str]] = None,
    max_sets: int = 12,
) -> list[tuple[str, ...]]:
    """Enumerate composite key candidates: all r-combinations (2..max_arity) of
    the detected single group columns whose composite cardinality clears the
    L29 guard. Returns ordered tuples, lowest-cardinality first."""
    cols = list(detected_group_cols) if detected_group_cols is not None else _auto_detect_group_cols(X)
    cols = [c for c in cols if c in X.columns]
    n_rows = len(X)
    cand: list[tuple[tuple[str, ...], int]] = []
    for r in range(2, max(2, int(max_arity)) + 1):
        for combo in combinations(cols, r):
            keys = build_composite_keys(X, combo)
            n_distinct = int(np.unique(keys).size)
            if composite_cardinality_ok(n_distinct, n_rows, max_card_frac):
                cand.append((tuple(combo), n_distinct))
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
):
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
    if group_col_sets is None or len(group_col_sets) == 0:
        group_col_sets = auto_detect_key_sets(
            X, max_arity=max_arity, max_card_frac=max_card_frac,
        )
    else:
        group_col_sets = [tuple(c for c in gset if c in X.columns) for gset in group_col_sets]
        group_col_sets = [gset for gset in group_col_sets if len(gset) >= 2]
    if not group_col_sets:
        return X.copy(), [], [], pd.DataFrame()

    if num_cols is None or len(num_cols) == 0:
        all_group_cols = sorted({c for gset in group_col_sets for c in gset})
        num_cols = _auto_detect_num_cols(X, all_group_cols)
    else:
        num_cols = [c for c in num_cols if c in X.columns]
    if not num_cols:
        return X.copy(), [], [], pd.DataFrame()

    enc_df, raw_recipes = generate_composite_group_agg_features(
        X, group_col_sets, num_cols, stats=stats, max_card_frac=max_card_frac,
    )
    if enc_df.empty:
        return X.copy(), [], [], pd.DataFrame()

    base_cols = list(num_cols)
    eng_to_source = {name: raw_recipes[name]["num_col"] for name in enc_df.columns}
    scores = score_grouped_agg_by_cmi_uplift(
        X, enc_df, y, base_cols, n_bins=n_bins, eng_to_source=eng_to_source,
    )
    keep = scores[(scores["cmi"] >= float(min_cmi)) & (scores["uplift"] >= float(min_uplift))]
    winners = list(keep["engineered_col"].head(int(top_k)))
    if not winners:
        return X.copy(), [], [], scores

    from .engineered_recipes import build_composite_group_agg_recipe

    X_aug = pd.concat([X, enc_df[winners]], axis=1)
    recipes = [build_composite_group_agg_recipe(name=name, **raw_recipes[name]) for name in winners]
    return X_aug, winners, recipes, scores
