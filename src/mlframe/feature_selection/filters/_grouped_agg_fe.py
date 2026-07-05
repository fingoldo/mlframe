"""Grouped multi-stat aggregator with CMI gate (Layer 87, 2026-06-01).

The single most-critical FE family in tabular ML competitions (NVIDIA cuDF
Kaggle Grandmaster blog, technique #1): for a categorical / low-cardinality
*group* key and a continuous *numeric* column, compute a per-group statistic
(mean / std / min / max / nunique / skew / median) and broadcast it back to
every row in the group. Two derived "anomaly-from-norm" residuals are also
emitted per (group, num) pair:

* ``z-within-group``   ``(x - mean(x | group)) / std(x | group)`` -- how many
  group-local standard deviations a row sits from its group centre.
* ``ratio-to-group``   ``x / mean(x | group)`` -- multiplicative deviation.

Why this matters
----------------

A raw column ``x`` can carry near-zero marginal MI about ``y`` while the
*group-conditioned* statistic of ``x`` is the real signal. Example: customer
spend ``x`` is noisy, but ``mean(spend | region)`` (the region's typical
spend) drives ``y``. Broadcasting the per-group mean recovers the signal that
raw ``x`` hides.

The CMI gate (reusing the Layer 60 conditional-MI primitives in
``_mi_greedy_cmi_fe``) ranks each engineered aggregate by ``CMI(agg; y |
base_cols)`` -- i.e. the NEW information the aggregate adds on top of the
already-present raw columns. An aggregate that merely re-expresses a raw
column already in the support (e.g. a group-mean that just shifts ``x`` when
the group is degenerate) scores near zero CMI and is gated out. The gate
additionally requires the aggregate's CMI to exceed the source num_col's own
marginal MI by a margin (uplift), so redundant broadcasts are dropped.

Recipe-based replay
-------------------

Each surviving aggregate is stored as a ``grouped_agg`` recipe carrying
``{group_col, num_col, stat, group_lookup_dict, global_value}`` (plus the
mean/std lookups needed by the two residual variants). Replay reads ONLY X:
each test row maps its group key through the stored lookup; unseen groups
fall back to the global statistic computed at fit. No y reference is captured
at fit time, so ``transform`` is leakage-free by construction.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._internals import canonical_group_token, group_key_strings

logger = logging.getLogger(__name__)

__all__ = [
    "STAT_NAMES",
    "engineered_name_grouped_agg",
    "engineered_name_grouped_z",
    "engineered_name_grouped_ratio",
    "generate_grouped_agg_features",
    "grouped_agg_with_recipes",
    "score_grouped_agg_by_cmi_uplift",
    "hybrid_grouped_agg_fe",
]

STAT_NAMES = ("mean", "std", "min", "max", "nunique", "skew", "median")

# Stats whose per-group lookup also doubles as the residual base (mean/std).
_RESIDUAL_STATS = ("mean", "std")


def engineered_name_grouped_agg(num_col: str, group_col: str, stat: str) -> str:
    return f"grpagg_{stat}({num_col}|{group_col})"


def engineered_name_grouped_z(num_col: str, group_col: str) -> str:
    return f"grpz({num_col}|{group_col})"


def engineered_name_grouped_ratio(num_col: str, group_col: str) -> str:
    return f"grpratio({num_col}|{group_col})"


# ---------------------------------------------------------------------------
# Per-group statistic computation
# ---------------------------------------------------------------------------


def _agg_func_for_stat(stat: str):
    """Return a pandas-groupby-compatible aggregator for ``stat``.

    ``nunique`` / ``skew`` are named methods; the rest map straight through.
    All operate NaN-skipping per pandas default, matching the global fallback
    computed with ``np.nan*`` below.
    """
    if stat in ("mean", "std", "min", "max", "median", "skew", "nunique"):
        return stat
    raise ValueError(f"grouped_agg: unknown stat {stat!r}; valid: {STAT_NAMES}")


def _global_value_for_stat(x: np.ndarray, stat: str) -> float:
    """Global fallback statistic for unseen groups at replay time."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return 0.0
    if stat == "mean":
        return float(np.mean(finite))
    if stat == "std":
        s = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        return s
    if stat == "min":
        return float(np.min(finite))
    if stat == "max":
        return float(np.max(finite))
    if stat == "median":
        return float(np.median(finite))
    if stat == "nunique":
        return float(np.unique(finite).size)
    if stat == "skew":
        return float(pd.Series(finite).skew()) if finite.size > 2 else 0.0
    raise ValueError(f"grouped_agg: unknown stat {stat!r}")


def _broadcast_lookup(
    group_keys: np.ndarray, lookup: dict, global_value: float,
) -> np.ndarray:
    """Map each row's group key through ``lookup`` (str-keyed), unseen ->
    global. Result NaN-scrubbed to ``global_value`` then to 0 as last resort.

    Group columns are low-cardinality, so the ``str(key)`` + ``dict.get`` is
    resolved once per UNIQUE key and broadcast back via the inverse index
    rather than once per row. On n=40k with ~8 groups this turns ~40k Python
    str()/get calls into ~8 (the per-row listcomp dominated the Layer 87
    grouped-agg profile: 2.9s tottime / 144 calls). Bit-identical to the
    per-row mapping (same str()+get per distinct key, deduplicated).
    """
    group_keys = np.asarray(group_keys)
    try:
        uniq, inverse = np.unique(group_keys, return_inverse=True)
        # numpy 2.0.0 briefly returned a 2-D inverse; ravel defensively so the
        # broadcast index is always 1-D and length-n.
        inverse = np.asarray(inverse).reshape(-1)
        uniq_vals = np.array(
            [lookup.get(str(_k), global_value) for _k in uniq],
            dtype=np.float64,
        )
        out = uniq_vals[inverse]
    except (TypeError, ValueError):
        # np.unique raises on unorderable mixed-type object arrays; fall back
        # to the per-row mapping (correct, just slower) so we never crash.
        out = np.array(
            [lookup.get(str(_k), global_value) for _k in group_keys],
            dtype=np.float64,
        )
    return np.nan_to_num(out, nan=global_value, posinf=global_value, neginf=global_value)


def generate_grouped_agg_features(
    X: pd.DataFrame,
    group_cols: Sequence[str],
    num_cols: Sequence[str],
    stats: Sequence[str] = STAT_NAMES,
):
    """Compute per-(group_col, num_col, stat) broadcasts plus the z-within-group
    and ratio-to-group residuals.

    Returns ``(enc_df, raw_recipes)`` where ``raw_recipes[name]`` is the
    kwargs dict for :func:`engineered_recipes.build_grouped_agg_recipe`.

    The per-group stat lookup table (str-keyed group value -> statistic) is
    stored in each recipe so :func:`apply` reads only X at replay. Unseen
    groups fall back to the fit-time global statistic.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"generate_grouped_agg_features: X must be a pandas DataFrame; " f"got {type(X).__name__}")
    group_cols = [c for c in group_cols if c in X.columns]
    stats = [s for s in stats if s in STAT_NAMES]
    encoded: dict[str, np.ndarray] = {}
    raw_recipes: dict[str, dict] = {}
    if not group_cols or not stats:
        return pd.DataFrame(index=X.index), raw_recipes

    for group_col in group_cols:
        g = X[group_col]
        g_keys = group_key_strings(g)
        cur_num_cols = [c for c in num_cols if c in X.columns and c != group_col and pd.api.types.is_numeric_dtype(X[c])]
        for num_col in cur_num_cols:
            x = np.asarray(X[num_col].to_numpy(), dtype=np.float64)
            grouped = X.groupby(group_col, observed=True, sort=False)[num_col]
            # Mean / std lookups are always needed for the residual variants
            # even if the user didn't request them as broadcast stats.
            mean_series = grouped.mean()
            std_series = grouped.std(ddof=1)
            # Keys built from the RAW group column (native dtype) -- canonicalise
            # so they match the canonical per-row keys (group_key_strings) at
            # both fit and a dtype-drifted predict (int<->float).
            lookup_mean = {canonical_group_token(k): float(v) for k, v in mean_series.items()}
            lookup_std = {canonical_group_token(k): (float(v) if np.isfinite(v) else 0.0) for k, v in std_series.items()}
            global_mean = _global_value_for_stat(x, "mean")
            global_std = _global_value_for_stat(x, "std")

            # ---- Broadcast each requested stat ----
            for stat in stats:
                agg_series = grouped.agg(_agg_func_for_stat(stat))
                lookup = {canonical_group_token(k): (float(v) if np.isfinite(v) else 0.0) for k, v in agg_series.items()}
                global_value = _global_value_for_stat(x, stat)
                broadcast = _broadcast_lookup(g_keys, lookup, global_value)
                name = engineered_name_grouped_agg(num_col, group_col, stat)
                encoded[name] = broadcast
                raw_recipes[name] = {
                    "group_col": group_col,
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

            # ---- z-within-group residual ----
            per_row_mean = _broadcast_lookup(g_keys, lookup_mean, global_mean)
            per_row_std = _broadcast_lookup(g_keys, lookup_std, global_std)
            per_row_std = np.where(per_row_std > 0.0, per_row_std, 1.0)
            zscore = np.nan_to_num(
                (x - per_row_mean) / per_row_std, nan=0.0, posinf=0.0, neginf=0.0,
            )
            z_name = engineered_name_grouped_z(num_col, group_col)
            encoded[z_name] = zscore
            raw_recipes[z_name] = {
                "group_col": group_col,
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

            # ---- ratio-to-group residual ----
            denom = np.where(np.abs(per_row_mean) > 1e-12, per_row_mean, np.nan)
            ratio = np.nan_to_num(x / denom, nan=1.0, posinf=1.0, neginf=1.0)
            r_name = engineered_name_grouped_ratio(num_col, group_col)
            encoded[r_name] = ratio
            raw_recipes[r_name] = {
                "group_col": group_col,
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


def apply_grouped_agg(X_test: pd.DataFrame, recipe: dict) -> np.ndarray:
    """Replay one grouped-agg column from the stored recipe dict, reading
    only ``X_test``. ``op`` selects the variant: ``broadcast`` (per-group
    stat), ``z_within`` (per-group z-score), ``ratio`` (x / per-group mean).
    """
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"apply_grouped_agg: X_test must be a DataFrame; got " f"{type(X_test).__name__}")
    group_col = recipe["group_col"]
    num_col = recipe["num_col"]
    op = recipe.get("op", "broadcast")
    if group_col not in X_test.columns or num_col not in X_test.columns:
        raise KeyError(f"apply_grouped_agg: missing column(s) {group_col!r}/{num_col!r} " f"from X_test")
    g_keys = group_key_strings(X_test[group_col])
    x = np.asarray(X_test[num_col].to_numpy(), dtype=np.float64)

    if op == "broadcast":
        lookup = dict(recipe["group_lookup_dict"])
        global_value = float(recipe["global_value"])
        return _broadcast_lookup(g_keys, lookup, global_value)

    lookup_mean = dict(recipe["lookup_mean"])
    global_mean = float(recipe["global_mean"])
    per_row_mean = _broadcast_lookup(g_keys, lookup_mean, global_mean)

    if op == "z_within":
        lookup_std = dict(recipe["lookup_std"])
        global_std = float(recipe.get("global_std", 1.0)) or 1.0
        per_row_std = _broadcast_lookup(g_keys, lookup_std, global_std)
        per_row_std = np.where(per_row_std > 0.0, per_row_std, 1.0)
        return np.nan_to_num(
            (x - per_row_mean) / per_row_std, nan=0.0, posinf=0.0, neginf=0.0,
        )
    if op == "ratio":
        denom = np.where(np.abs(per_row_mean) > 1e-12, per_row_mean, np.nan)
        return np.nan_to_num(x / denom, nan=1.0, posinf=1.0, neginf=1.0)
    raise ValueError(f"apply_grouped_agg: unknown op {op!r}")


def _coerce_X_for_grouped_agg(X, group_col: str, num_col: str, recipe_name: str) -> pd.DataFrame:
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


def _apply_grouped_agg_recipe(recipe, X) -> np.ndarray:
    """Adapter consumed by ``engineered_recipes.apply_recipe``: pulls the
    stored payload out of ``recipe.extra`` and replays via
    :func:`apply_grouped_agg`."""
    group_col = str(recipe.extra["group_col"])
    num_col = str(recipe.extra["num_col"])
    X_view = _coerce_X_for_grouped_agg(X, group_col, num_col, recipe.name)
    return apply_grouped_agg(
        X_view,
        {
            "group_col": group_col,
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


def grouped_agg_with_recipes(
    X: pd.DataFrame,
    *,
    group_cols: Optional[Sequence[str]] = None,
    num_cols: Optional[Sequence[str]] = None,
    stats: Sequence[str] = STAT_NAMES,
):
    """Append grouped-agg columns and emit one recipe per engineered column."""
    from .engineered_recipes import build_grouped_agg_recipe

    if not group_cols or not num_cols:
        return X.copy(), [], []
    group_cols = [c for c in group_cols if c in X.columns]
    num_cols = [c for c in num_cols if c in X.columns]
    if not group_cols or not num_cols:
        return X.copy(), [], []
    enc_df, raw_recipes = generate_grouped_agg_features(
        X, group_cols, num_cols, stats=stats,
    )
    if enc_df.empty:
        return X.copy(), [], []
    X_aug = pd.concat([X, enc_df], axis=1)
    appended = list(enc_df.columns)
    recipes = [build_grouped_agg_recipe(name=name, **raw_recipes[name]) for name in appended]
    return X_aug, appended, recipes


# ---------------------------------------------------------------------------
# CMI-uplift scorer / gate
# ---------------------------------------------------------------------------


def score_grouped_agg_by_cmi_uplift(
    raw_X: pd.DataFrame,
    eng_X: pd.DataFrame,
    y: np.ndarray,
    base_cols: Sequence[str],
    *,
    n_bins: int = 10,
    eng_to_source: Optional[dict] = None,
) -> pd.DataFrame:
    """Rank engineered aggregates by ``CMI(agg; y | base_cols)`` and gate by
    uplift over the source num_col's own marginal ``MI(num_col; y)``.

    Reuses the Layer 60 CMI primitives (``score_candidates_by_cmi`` for the
    conditional term, ``_cmi_from_binned`` for the marginal source MI).

    Parameters
    ----------
    raw_X : DataFrame
        Original (pre-engineering) frame holding ``base_cols`` and the source
        num_cols.
    eng_X : DataFrame
        Engineered aggregate columns to score.
    y : ndarray
        Target.
    base_cols : sequence of str
        Conditioning support -- typically the raw columns already present.
    eng_to_source : dict or None
        Maps each engineered column name to its source num_col name (for the
        uplift baseline). When ``None``, the uplift baseline is 0 (pure CMI
        ranking, no redundancy gate).

    Returns
    -------
    DataFrame sorted by ``cmi`` descending with columns
    ``[engineered_col, source_col, cmi, source_mi, uplift]``.
    """
    from ._mi_greedy_cmi_fe import (
        score_candidates_by_cmi,
        _quantile_bin,
        _cmi_from_binned,
    )

    if eng_X.shape[1] == 0:
        return pd.DataFrame(columns=["engineered_col", "source_col", "cmi", "source_mi", "uplift"])
    y_arr = np.asarray(y)
    if not np.issubdtype(y_arr.dtype, np.integer):
        # Continuous y must be quantile-binned, never int-truncated: astype(int64) collapses 0.7->0 and destroys the CMI gate for regression targets.
        if y_arr.dtype.kind in "fc" and int(np.unique(y_arr).size) > 32:
            try:
                y_arr = pd.qcut(y_arr, q=10, labels=False, duplicates="drop").to_numpy()
            except Exception:
                pass
        _, y_arr = np.unique(y_arr, return_inverse=True)
    y_bin = y_arr.astype(np.int64)

    base_cols = [c for c in base_cols if c in raw_X.columns]
    X_support = raw_X[base_cols] if base_cols else None
    cmi = score_candidates_by_cmi(eng_X, y_arr, X_support, nbins=n_bins)

    # Marginal MI of each source num_col (uplift baseline), computed once.
    source_mi_cache: dict[str, float] = {}
    eng_to_source = eng_to_source or {}

    def _source_mi(col: str) -> float:
        if col in source_mi_cache:
            return source_mi_cache[col]
        if col not in raw_X.columns:
            source_mi_cache[col] = 0.0
            return 0.0
        x_bin = _quantile_bin(raw_X[col].to_numpy(), nbins=n_bins)
        mi = _cmi_from_binned(x_bin, y_bin, None)
        source_mi_cache[col] = float(mi)
        return float(mi)

    rows = []
    for col in eng_X.columns:
        src = eng_to_source.get(col)
        smi = _source_mi(src) if src else 0.0
        c = float(cmi.get(col, 0.0))
        rows.append({
            "engineered_col": col,
            "source_col": src,
            "cmi": c,
            "source_mi": smi,
            "uplift": c - smi,
        })
    out = pd.DataFrame(rows)
    return out.sort_values("cmi", ascending=False, kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# End-to-end pipeline with auto-detection
# ---------------------------------------------------------------------------


def _auto_detect_group_cols(X: pd.DataFrame, max_cols: int = 4) -> list[str]:
    """Auto-detect group columns via the int-as-cat heuristic. Reuses the
    composite_auto_detect detector (cardinality 3..500); falls back to a
    self-contained scan if that import is unavailable.
    """
    try:
        from ...training.composite import (
            detect_group_column_candidates,
        )
        cands = detect_group_column_candidates(X)
        return [name for name, _info in cands[:max_cols]]
    except Exception as _e:
        logger.debug(
            "grouped_agg auto-detect: detector import failed (%s); using " "fallback cardinality scan.",
            _e,
        )
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
    """Continuous columns (float dtype, or high-cardinality numeric) excluding
    the chosen group columns.
    """
    group_set = set(group_cols)
    out: list[str] = []
    for c in X.columns:
        if c in group_set:
            continue
        # Skip already-engineered grouped columns (grpagg/grpz/grpratio/... from an earlier grouped-FE stage). They are constant
        # within group, so aggregating them again is degenerate, and the nested recipe cannot replay from raw X at transform time.
        if str(c).startswith("grp"):
            continue
        col = X[c]
        if not pd.api.types.is_numeric_dtype(col):
            continue
        if pd.api.types.is_float_dtype(col):
            out.append(str(c))
            continue
        # Integer column: treat as continuous only if high cardinality.
        if int(col.nunique(dropna=True)) > 500:
            out.append(str(c))
    return out[:max_cols]


def _filter_num_cols_by_relevance(
    X: pd.DataFrame, y, num_cols: Sequence[str], *, n_bins: int = 10,
    rel_ratio: float = 0.10, abs_floor: float = 0.01,
) -> list[str]:
    """Keep auto-detected ``num_cols`` that carry marginal information about ``y``; drop those indistinguishable from noise.

    A grouped aggregate of a source independent of y re-encodes only the group key's own marginal info (constant within group), so
    it adds nothing the group identity already provides -- but its in-sample CMI ties the genuine signal aggregate, letting the
    screen pick a noise aggregate. A column is kept when its marginal MI(x; y) clears BOTH a relative bar (``rel_ratio`` x the most
    relevant column's MI, so a few weak-but-real columns survive alongside the strongest) AND a small absolute floor (``abs_floor``,
    above the plug-in binning-bias noise level). Best-effort: on any failure return the input unchanged (never lose real columns)."""
    cols = [c for c in num_cols if c in X.columns]
    if len(cols) <= 1:
        return cols
    try:
        from ._mi_greedy_cmi_fe import _quantile_bin, _cmi_from_binned
        y_arr = np.asarray(y)
        if not np.issubdtype(y_arr.dtype, np.integer):
            y_arr = y_arr.astype(np.int64)
        _, y_bin = np.unique(y_arr, return_inverse=True)
        y_bin = y_bin.astype(np.int64)
        mis = {}
        for c in cols:
            xb = _quantile_bin(X[c].to_numpy(), nbins=n_bins)
            mis[c] = float(_cmi_from_binned(xb, y_bin, None))
        m_max = max(mis.values())
        if m_max <= 0.0:
            return cols
        thr = max(float(abs_floor), float(rel_ratio) * m_max)
        kept = [c for c in cols if mis[c] >= thr]
        return kept or cols
    except Exception as _e:
        logger.debug("grouped_agg num_col relevance filter failed (%s); using unfiltered auto set.", _e)
        return cols


def hybrid_grouped_agg_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    group_cols: Optional[Sequence[str]] = None,
    num_cols: Optional[Sequence[str]] = None,
    stats: Sequence[str] = STAT_NAMES,
    top_k: int = 10,
    n_bins: int = 10,
    min_cmi: float = 1e-4,
    min_uplift: float = 0.0,
):
    """End-to-end grouped-agg FE pipeline.

    1. Auto-detect ``group_cols`` (int-as-cat heuristic, cardinality 3..500)
       and ``num_cols`` (continuous) when not supplied.
    2. Materialise every (group, num, stat) broadcast + z / ratio residual.
    3. Score by ``CMI(agg; y | base_cols)`` and gate by uplift over the source
       num_col marginal MI; keep the top ``top_k`` survivors.
    4. Append survivors to X; return ``(X_aug, appended, recipes, scores)``.

    ``y`` is consumed ONLY by the CMI scorer / gate; the recipes carry no y
    reference, so transform-time replay is leakage-free.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"hybrid_grouped_agg_fe: X must be a pandas DataFrame; got " f"{type(X).__name__}")
    if group_cols is None or len(group_cols) == 0:
        group_cols = _auto_detect_group_cols(X)
    else:
        group_cols = [c for c in group_cols if c in X.columns]
    if not group_cols:
        return X.copy(), [], [], pd.DataFrame()
    _num_cols_auto = num_cols is None or len(num_cols) == 0
    if _num_cols_auto:
        num_cols = _auto_detect_num_cols(X, group_cols)
        # y-aware relevance filter (auto-detect only). A grouped aggregate of a source column that is INDEPENDENT of y can only
        # re-encode the group key's own marginal info about y (the aggregate is constant within each group), so it adds nothing the
        # group identity does not already carry -- yet its in-sample CMI ties the genuine signal aggregate, letting the downstream
        # screen pick a pure-noise aggregate by chance. Restrict the auto-detected sources to those with marginal MI about y above a
        # noise floor so only columns carrying real signal (e.g. a per-group-mean-driven x) get aggregated. Manual num_cols (caller
        # supplied) are trusted as-is and never filtered. Best-effort: any failure falls back to the unfiltered auto set.
        num_cols = _filter_num_cols_by_relevance(X, y, num_cols, n_bins=n_bins)
    else:
        num_cols = [c for c in num_cols if c in X.columns]
    if not num_cols:
        return X.copy(), [], [], pd.DataFrame()

    enc_df, raw_recipes = generate_grouped_agg_features(
        X, group_cols, num_cols, stats=stats,
    )
    if enc_df.empty:
        return X.copy(), [], [], pd.DataFrame()

    base_cols = list(num_cols)  # condition on the raw source cols
    eng_to_source = {name: raw_recipes[name]["num_col"] for name in enc_df.columns}
    scores = score_grouped_agg_by_cmi_uplift(
        X, enc_df, y, base_cols, n_bins=n_bins, eng_to_source=eng_to_source,
    )
    # Gate: positive CMI above floor AND uplift over source marginal MI.
    keep = scores[(scores["cmi"] >= float(min_cmi)) & (scores["uplift"] >= float(min_uplift))]
    winners = list(keep["engineered_col"].head(int(top_k)))
    if not winners:
        return X.copy(), [], [], scores

    from .engineered_recipes import build_grouped_agg_recipe

    X_aug = pd.concat([X, enc_df[winners]], axis=1)
    recipes = [build_grouped_agg_recipe(name=name, **raw_recipes[name]) for name in winners]
    return X_aug, winners, recipes, scores
