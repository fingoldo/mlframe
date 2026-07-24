"""Per-group histogram + quantile FE with target-aware edges (Layer 88, 2026-06-01).

NVIDIA cuDF Kaggle-Grandmaster blog technique #2 (companion to the Layer 87
grouped multi-stat aggregator). Where Layer 87 broadcasts a per-group scalar
statistic (mean / std / ...), Layer 88 captures the *distributional* position
of a row WITHIN its group:

* ``percentile-rank-within-group`` -- the empirical CDF position of a row's
  value among the values of its own group, i.e. ``P(X <= x | group)``. A row
  sitting at the group's median maps to ~0.5 regardless of the group's
  absolute location / scale. This recovers signals where ``y`` depends on
  whether ``x`` is low / high *relative to its group's distribution*, not on
  ``x`` itself nor on any single group moment (Layer 87 mean / std miss it
  whenever the discriminating structure is bimodal-within-group).
* ``group quantile spread`` -- per-group IQR (``q75 - q25``) and
  ``p90 - p10`` broadcast back to rows. A dispersion feature: the WIDTH of a
  group's distribution, orthogonal to the group's centre.

Target-aware group bins (the IT enhancement)
---------------------------------------------

Instead of fixed quantile edges, :func:`generate_target_aware_group_bins`
finds, *per group*, the bin edges that MAXIMISE ``I(bin; y)`` via the
Fayyad-Irani MDLP supervised binner (reused from ``_adaptive_nbins``). To stay
leak-safe the supervised edges are fit on K-fold OOF splits (the same
round-robin fold assignment Layer 33 uses for K-fold target encoding): a row's
bin index is assigned from edges fit on the OTHER folds, so the bin a row lands
in never saw that row's own ``y``. The fitted per-group edges (refit on all
folds) are stored in the recipe for leak-free test-time replay.

Leakage safety (CRITICAL)
-------------------------

* ``grouped_quantile`` recipes store ONLY the per-group sorted-value arrays (or
  fixed quantile breakpoints) computed on TRAIN; replay reads only X. Unseen
  groups fall back to the global (pooled) quantiles fit at train time.
* ``target_aware_group_bin`` recipes store the per-group MDLP edges refit on
  ALL train rows after the OOF scoring pass; the OOF assignment is used ONLY to
  compute the leak-safe MI uplift score, never persisted. Replay maps a row's
  value through ``searchsorted`` on its group's stored edges -- a pure function
  of X. No ``y`` reference is captured in either recipe kind.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._internals import group_key_strings

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_QUANTILES",
    "engineered_name_grouped_pctrank",
    "engineered_name_grouped_iqr",
    "engineered_name_grouped_p90p10",
    "engineered_name_target_aware_bin",
    "generate_grouped_quantile_features",
    "generate_target_aware_group_bins",
    "apply_grouped_quantile",
    "apply_target_aware_group_bin",
    "score_grouped_quantile_by_mi_uplift",
    "hybrid_grouped_quantile_fe",
]

DEFAULT_QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)

# Minimum per-group sample size to fit a trustworthy within-group quantile /
# supervised bin; smaller groups fall back to the pooled global edges.
_MIN_GROUP_SIZE = 8


def engineered_name_grouped_pctrank(num_col: str, group_col: str) -> str:
    """Column name for the percentile-rank-within-group feature of ``num_col`` grouped by ``group_col``."""
    return f"grppct({num_col}|{group_col})"


def engineered_name_grouped_iqr(num_col: str, group_col: str) -> str:
    """Column name for the per-group IQR (q75-q25) spread broadcast of ``num_col`` grouped by ``group_col``."""
    return f"grpiqr({num_col}|{group_col})"


def engineered_name_grouped_p90p10(num_col: str, group_col: str) -> str:
    """Column name for the per-group p90-p10 spread broadcast of ``num_col`` grouped by ``group_col``."""
    return f"grpp90p10({num_col}|{group_col})"


def engineered_name_target_aware_bin(num_col: str, group_col: str) -> str:
    """Column name for the target-aware (MDLP-supervised) per-group bin index of ``num_col`` grouped by ``group_col``."""
    return f"grptabin({num_col}|{group_col})"


# ---------------------------------------------------------------------------
# Percentile-rank-within-group + spread
# ---------------------------------------------------------------------------


def _broadcast_lookup(g_keys: np.ndarray, lookup: dict, glob: float) -> np.ndarray:
    """Map each row's group key through ``lookup`` (str-keyed), unseen -> glob.

    Group columns are low-cardinality, so the ``str(key)`` + ``dict.get`` is
    resolved once per UNIQUE key (np.unique return_inverse) and broadcast back
    via the inverse index, not once per row -- the per-row listcomp form was a
    Layer-88 grouped-quantile hotspot (~0.65s x 2 sites / 32 calls each). Bit-
    identical to the per-row mapping (same str()+get per distinct key). Ravels
    the inverse (numpy 2.0.0 briefly returned 2-D) and falls back to the per-row
    path on the TypeError np.unique raises for unorderable mixed-type objects.
    """
    g_keys = np.asarray(g_keys)
    try:
        uniq, inverse = np.unique(g_keys, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        uniq_vals = np.array([lookup.get(str(_k), glob) for _k in uniq], dtype=np.float64)
        out = uniq_vals[inverse]
    except (TypeError, ValueError):
        out = np.array([lookup.get(str(_k), glob) for _k in g_keys], dtype=np.float64)
    return np.nan_to_num(out, nan=glob, posinf=glob, neginf=glob)


def _pct_rank_in_sorted(sorted_vals: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Empirical CDF position of each ``x`` in ``sorted_vals`` -- the fraction
    of group values ``<= x`` via the midpoint of the left / right insertion
    ranks (ties get their average rank, matching ``rank(pct=True)`` semantics).
    Returns values in ``[0, 1]``; an empty reference yields 0.5 (neutral).
    """
    m = sorted_vals.size
    if m == 0:
        return np.full(x.shape, 0.5, dtype=np.float64)
    lo = np.searchsorted(sorted_vals, x, side="left")
    hi = np.searchsorted(sorted_vals, x, side="right")
    # Average rank for ties; normalise by m so the result is a CDF position.
    return (lo + hi) / (2.0 * m)


def generate_grouped_quantile_features(
    X: pd.DataFrame,
    group_cols: Sequence[str],
    num_cols: Sequence[str],
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
):
    """Compute per-(group_col, num_col) percentile-rank-within-group plus the
    group IQR / p90-p10 spread broadcasts.

    Returns ``(enc_df, raw_recipes)`` where ``raw_recipes[name]`` is the kwargs
    dict for :func:`engineered_recipes.build_grouped_quantile_recipe`.

    Each recipe stores the per-group SORTED value array (for the percentile
    variant) or the per-group spread scalar (for IQR / p90-p10), plus the
    pooled-global fallback used for groups unseen at replay. Replay reads only
    X -- no ``y`` reference is captured, so transform() is leakage-free.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"generate_grouped_quantile_features: X must be a pandas DataFrame; " f"got {type(X).__name__}")
    group_cols = [c for c in group_cols if c in X.columns]
    quantiles = tuple(float(q) for q in quantiles if 0.0 <= float(q) <= 1.0)
    encoded: dict[str, np.ndarray] = {}
    raw_recipes: dict[str, dict] = {}
    if not group_cols:
        return pd.DataFrame(index=X.index), raw_recipes

    for group_col in group_cols:
        g_keys = group_key_strings(X[group_col])
        cur_num_cols = [c for c in num_cols if c in X.columns and c != group_col and pd.api.types.is_numeric_dtype(X[c])]
        # Row indices per group key depend only on group_col; hoist once (mirrors generate_target_aware_group_bins's
        # group_rows) instead of rebuilding the groupby-of-arange for every num_col.
        n = len(X)
        group_rows: dict[str, np.ndarray] = {str(gv): idx.to_numpy() for gv, idx in pd.Series(np.arange(n)).groupby(g_keys, sort=False)}
        for num_col in cur_num_cols:
            x = np.asarray(X[num_col].to_numpy(), dtype=np.float64)
            finite_all = x[np.isfinite(x)]
            global_sorted = np.sort(finite_all) if finite_all.size else np.array([], dtype=np.float64)
            global_iqr = float(np.quantile(finite_all, 0.75) - np.quantile(finite_all, 0.25)) if finite_all.size else 0.0
            global_p90p10 = float(np.quantile(finite_all, 0.90) - np.quantile(finite_all, 0.10)) if finite_all.size else 0.0

            # Per-group sorted-value arrays (str-keyed) for the CDF position.
            group_sorted: dict[str, list] = {}
            iqr_lookup: dict[str, float] = {}
            p90p10_lookup: dict[str, float] = {}
            pct_out = np.empty(x.shape, dtype=np.float64)

            for gv, rows in group_rows.items():
                vals = x[rows]
                fin = vals[np.isfinite(vals)]
                if fin.size >= _MIN_GROUP_SIZE:
                    sv = np.sort(fin)
                    q25, q75 = np.quantile(fin, [0.25, 0.75])
                    p10, p90 = np.quantile(fin, [0.10, 0.90])
                    group_iqr = float(q75 - q25)
                    group_p90p10 = float(p90 - p10)
                else:
                    sv = global_sorted
                    group_iqr = global_iqr
                    group_p90p10 = global_p90p10
                group_sorted[str(gv)] = sv.tolist()
                iqr_lookup[str(gv)] = group_iqr
                p90p10_lookup[str(gv)] = group_p90p10
                # Fit ranks each train value within a sorted set that INCLUDES itself, whereas replay ranks a test
                # value against the stored train-only sorted array: a small systematic ~1/m per-group distributional
                # offset (not a y-leak). Acceptable and unchanged by design; documented so a future reader does not
                # mistake it for a fit/serve skew (mrmr_critique FE-F6).
                pct_out[rows] = _pct_rank_in_sorted(sv, vals)

            pct_out = np.nan_to_num(pct_out, nan=0.5, posinf=1.0, neginf=0.0)

            pct_name = engineered_name_grouped_pctrank(num_col, group_col)
            encoded[pct_name] = pct_out
            raw_recipes[pct_name] = {
                "group_col": group_col,
                "num_col": num_col,
                "op": "pct_rank",
                "group_sorted": group_sorted,
                "global_sorted": global_sorted.tolist(),
                "iqr_lookup": iqr_lookup,
                "p90p10_lookup": p90p10_lookup,
                "global_iqr": global_iqr,
                "global_p90p10": global_p90p10,
                "quantiles": list(quantiles),
            }

            # IQR spread broadcast.
            iqr_out = _broadcast_lookup(g_keys, iqr_lookup, global_iqr)
            iqr_name = engineered_name_grouped_iqr(num_col, group_col)
            encoded[iqr_name] = iqr_out
            raw_recipes[iqr_name] = {
                "group_col": group_col,
                "num_col": num_col,
                "op": "iqr",
                "group_sorted": {},
                "global_sorted": [],
                "iqr_lookup": iqr_lookup,
                "p90p10_lookup": p90p10_lookup,
                "global_iqr": global_iqr,
                "global_p90p10": global_p90p10,
                "quantiles": list(quantiles),
            }

            # p90 - p10 spread broadcast.
            p_out = _broadcast_lookup(g_keys, p90p10_lookup, global_p90p10)
            p_name = engineered_name_grouped_p90p10(num_col, group_col)
            encoded[p_name] = p_out
            raw_recipes[p_name] = {
                "group_col": group_col,
                "num_col": num_col,
                "op": "p90p10",
                "group_sorted": {},
                "global_sorted": [],
                "iqr_lookup": iqr_lookup,
                "p90p10_lookup": p90p10_lookup,
                "global_iqr": global_iqr,
                "global_p90p10": global_p90p10,
                "quantiles": list(quantiles),
            }

    enc_df = pd.DataFrame(encoded, index=X.index)
    return enc_df, raw_recipes


def apply_grouped_quantile(X_test: pd.DataFrame, recipe: dict) -> np.ndarray:
    """Replay one grouped-quantile column from the stored recipe, reading only
    ``X_test``. ``op`` selects the variant: ``pct_rank`` (CDF position of x in
    its group's stored sorted values), ``iqr`` / ``p90p10`` (per-group spread
    broadcast). Unseen groups fall back to the pooled global edges.
    """
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"apply_grouped_quantile: X_test must be a DataFrame; got " f"{type(X_test).__name__}")
    group_col = recipe["group_col"]
    num_col = recipe["num_col"]
    op = recipe.get("op", "pct_rank")
    if group_col not in X_test.columns or num_col not in X_test.columns:
        raise KeyError(f"apply_grouped_quantile: missing column(s) {group_col!r}/" f"{num_col!r} from X_test")
    g_keys = group_key_strings(X_test[group_col])
    x = np.asarray(X_test[num_col].to_numpy(), dtype=np.float64)

    if op == "iqr":
        lookup = dict(recipe["iqr_lookup"])
        glob = float(recipe["global_iqr"])
        return _broadcast_lookup(g_keys, lookup, glob)
    if op == "p90p10":
        lookup = dict(recipe["p90p10_lookup"])
        glob = float(recipe["global_p90p10"])
        return _broadcast_lookup(g_keys, lookup, glob)
    if op == "pct_rank":
        group_sorted = recipe["group_sorted"]
        global_sorted = np.asarray(recipe["global_sorted"], dtype=np.float64)
        # Cache per-group sorted ndarrays to avoid re-converting per row.
        cache: dict[str, np.ndarray] = {}
        out = np.empty(x.shape, dtype=np.float64)
        for gv, idx in pd.Series(np.arange(len(x))).groupby(g_keys, sort=False):
            rows = idx.to_numpy()
            key = str(gv)
            sv = cache.get(key)
            if sv is None:
                if key in group_sorted:
                    sv = np.asarray(group_sorted[key], dtype=np.float64)
                else:
                    sv = global_sorted
                cache[key] = sv
            out[rows] = _pct_rank_in_sorted(sv, x[rows])
        return np.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0)
    raise ValueError(f"apply_grouped_quantile: unknown op {op!r}")


# ---------------------------------------------------------------------------
# Target-aware per-group supervised bins (IT enhancement, OOF leak-safe)
# ---------------------------------------------------------------------------


def _fit_group_edges(vals: np.ndarray, yv: np.ndarray, n_bins: int) -> np.ndarray:
    """Supervised MDLP edges for one group's ``(vals, yv)``; falls back to
    fixed quantile edges when MDLP returns no split (degenerate / weak signal).
    Returns INNER edges (no -inf / +inf sentinels).
    """
    from ._adaptive_nbins import edges_fayyad_irani

    fin = np.isfinite(vals)
    vals = vals[fin]
    yv = yv[fin]
    if vals.size < _MIN_GROUP_SIZE or np.unique(vals).size < 2:
        return np.array([], dtype=np.float64)
    try:
        edges = edges_fayyad_irani(vals, yv.astype(np.int64))
    except Exception:
        edges = np.array([], dtype=np.float64)
    if edges.size == 0:
        # Fallback: fixed quantile edges capped at n_bins.
        nb = max(2, int(n_bins))
        qs = np.linspace(0.0, 1.0, nb + 1)[1:-1]
        edges = np.unique(np.quantile(vals, qs))
    return np.asarray(edges, dtype=np.float64)


def generate_target_aware_group_bins(
    X: pd.DataFrame,
    y: np.ndarray,
    group_cols: Sequence[str],
    num_cols: Sequence[str],
    n_bins: int = 5,
    n_folds: int = 5,
    random_state: int = 0,
):
    """Per-group supervised bin index that maximises ``I(bin; y)`` within each
    group, assigned OOF (K-fold) to be leak-safe.

    For each (group, num) pair:

    1. Round-robin K-fold assignment over a shuffled index (Layer 33 pattern).
    2. For each fold ``f``: fit per-group MDLP edges on rows in folds ``!= f``;
       assign the bin index for fold-``f`` rows via ``searchsorted`` on their
       group's edges. The bin a row lands in never saw its own ``y``.
    3. Refit per-group edges on ALL rows for the persisted recipe (test-time
       replay), storing one inner-edge array per group + a pooled global
       fallback for unseen groups.

    Returns ``(enc_df, raw_recipes)``. The OOF bin index is the engineered
    column (leak-safe for in-fold MI scoring); the recipe carries the all-rows
    refit edges so test replay is a pure function of X.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"generate_target_aware_group_bins: X must be a pandas DataFrame; " f"got {type(X).__name__}")
    group_cols = [c for c in group_cols if c in X.columns]
    y_arr = np.asarray(y)
    if y_arr.dtype.kind in "fc":
        if int(np.unique(y_arr).size) <= 32:
            y_arr = y_arr.astype(np.int64)
        else:
            try:
                y_arr = pd.qcut(y_arr, q=10, labels=False, duplicates="drop").to_numpy()
            except Exception:
                y_arr = y_arr.astype(np.int64)
    y_arr = y_arr.astype(np.int64)

    encoded: dict[str, np.ndarray] = {}
    raw_recipes: dict[str, dict] = {}
    if not group_cols:
        return pd.DataFrame(index=X.index), raw_recipes

    n = len(X)
    n_folds = max(2, int(n_folds))
    rng = np.random.default_rng(int(random_state))
    perm = rng.permutation(n)
    fold_ids = np.empty(n, dtype=np.int64)
    fold_ids[perm] = np.arange(n) % n_folds
    # train_mask/test_mask depend only on (f, fold_ids), not on group_col/num_col; precompute once
    # instead of recomputing inside the group_col x num_col x fold triple loop below.
    _fold_masks = [(fold_ids != f) for f in range(n_folds)]
    _fold_test_masks = [~m for m in _fold_masks]

    for group_col in group_cols:
        g_keys = group_key_strings(X[group_col])
        cur_num_cols = [c for c in num_cols if c in X.columns and c != group_col and pd.api.types.is_numeric_dtype(X[c])]
        # Precompute the row indices per group key once.
        group_rows: dict[str, np.ndarray] = {str(gv): idx.to_numpy() for gv, idx in pd.Series(np.arange(n)).groupby(g_keys, sort=False)}
        for num_col in cur_num_cols:
            x = np.asarray(X[num_col].to_numpy(), dtype=np.float64)
            finite_all = x[np.isfinite(x)]
            global_edges = _fit_group_edges(finite_all, y_arr[np.isfinite(x)], n_bins) if finite_all.size else np.array([], dtype=np.float64)

            # ---- OOF bin assignment for leak-safe MI scoring ----
            # bench-attempt-FUTURE (2026-06-23): capping the n_folds x n_groups MDLP refits below
            # (e.g. refit only groups above a higher min-count, route the rest to global_edges)
            # changes OOF bin edges => NOT identity-safe and NOT selection-equivalent. _benchmarks/
            # bench_grouped_quantile_fe.py measured 100% of qualifying groups (n=100k, n_groups
            # 50/100/200) having per-group MDLP edges that DIFFER from the pooled global edges, so
            # any group rerouted to the global fallback gets a different searchsorted bin index for
            # every one of its rows => different oof_bin values => different MI(bin;y) => possibly
            # different downstream feature selection. Per-group divergence from global IS the whole
            # point of target-aware bins (within-group signal global edges miss), so the cap cannot
            # be made identity-preserving. Uncapped cost is bounded (0.36-0.48s @ n=100k here); revisit
            # only for pathologically high group counts, and then with a SIZE/COUNT gate, never a blanket cap.
            oof_bin = np.zeros(n, dtype=np.float64)
            for f in range(n_folds):
                train_mask = _fold_masks[f]
                test_mask = _fold_test_masks[f]
                if not train_mask.any() or not test_mask.any():
                    continue
                # Small-group fallback edges must be fit on this fold's TRAIN rows only. Using the all-rows
                # ``global_edges`` (fit on every row INCLUDING the scored fold) let the fold's own y shape the cut
                # points that then bin it -- a y-leak into the OOF MI score that can float a pure-noise tiny-group bin
                # above the gate. The persisted recipe below still uses all-rows edges (serving is out-of-sample).
                _tr_finite = train_mask & np.isfinite(x)
                fold_global_edges = _fit_group_edges(x[_tr_finite], y_arr[_tr_finite], n_bins) if _tr_finite.any() else global_edges
                for rows in group_rows.values():
                    tr = rows[train_mask[rows]]
                    te = rows[test_mask[rows]]
                    if te.size == 0:
                        continue
                    if tr.size >= _MIN_GROUP_SIZE:
                        edges = _fit_group_edges(x[tr], y_arr[tr], n_bins)
                    else:
                        edges = fold_global_edges
                    oof_bin[te] = np.searchsorted(edges, x[te], side="right")

            # ---- All-rows refit edges for the persisted recipe ----
            group_edges: dict[str, list] = {}
            for key, rows in group_rows.items():
                if rows.size >= _MIN_GROUP_SIZE:
                    edges = _fit_group_edges(x[rows], y_arr[rows], n_bins)
                else:
                    edges = global_edges
                group_edges[key] = np.asarray(edges, dtype=np.float64).tolist()

            name = engineered_name_target_aware_bin(num_col, group_col)
            encoded[name] = np.nan_to_num(oof_bin, nan=0.0)
            raw_recipes[name] = {
                "group_col": group_col,
                "num_col": num_col,
                "op": "target_aware_bin",
                "group_edges": group_edges,
                "global_edges": np.asarray(global_edges, dtype=np.float64).tolist(),
                "n_bins": int(n_bins),
            }

    enc_df = pd.DataFrame(encoded, index=X.index)
    return enc_df, raw_recipes


def apply_target_aware_group_bin(X_test: pd.DataFrame, recipe: dict) -> np.ndarray:
    """Replay a target-aware group bin index from the stored per-group edges,
    reading only ``X_test``. Each row maps its value through ``searchsorted`` on
    its group's stored edges; unseen groups use the pooled global edges.
    """
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"apply_target_aware_group_bin: X_test must be a DataFrame; got " f"{type(X_test).__name__}")
    group_col = recipe["group_col"]
    num_col = recipe["num_col"]
    if group_col not in X_test.columns or num_col not in X_test.columns:
        raise KeyError(f"apply_target_aware_group_bin: missing column(s) {group_col!r}/" f"{num_col!r} from X_test")
    g_keys = group_key_strings(X_test[group_col])
    x = np.asarray(X_test[num_col].to_numpy(), dtype=np.float64)
    group_edges = recipe["group_edges"]
    global_edges = np.asarray(recipe["global_edges"], dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.float64)
    cache: dict[str, np.ndarray] = {}
    for gv, idx in pd.Series(np.arange(len(x))).groupby(g_keys, sort=False):
        rows = idx.to_numpy()
        key = str(gv)
        edges = cache.get(key)
        if edges is None:
            if key in group_edges:
                edges = np.asarray(group_edges[key], dtype=np.float64)
            else:
                edges = global_edges
            cache[key] = edges
        out[rows] = np.searchsorted(edges, x[rows], side="right")
    return np.nan_to_num(out, nan=0.0)


# ---------------------------------------------------------------------------
# Recipe adapters consumed by engineered_recipes.apply_recipe
# ---------------------------------------------------------------------------


def _coerce_X(X, group_col: str, num_col: str, recipe_name: str) -> pd.DataFrame:
    """Extract ``[group_col, num_col]`` from ``X`` as a pandas frame regardless of the incoming carrier (pandas / polars / structured ndarray), for recipe replay."""
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


def _apply_grouped_quantile_recipe(recipe, X) -> np.ndarray:
    """Adapter: pulls the stored payload from ``recipe.extra`` and replays via
    :func:`apply_grouped_quantile`."""
    group_col = str(recipe.extra["group_col"])
    num_col = str(recipe.extra["num_col"])
    X_view = _coerce_X(X, group_col, num_col, recipe.name)
    return apply_grouped_quantile(
        X_view,
        {
            "group_col": group_col,
            "num_col": num_col,
            "op": str(recipe.extra.get("op", "pct_rank")),
            "group_sorted": dict(recipe.extra.get("group_sorted", {})),
            "global_sorted": list(recipe.extra.get("global_sorted", [])),
            "iqr_lookup": dict(recipe.extra.get("iqr_lookup", {})),
            "p90p10_lookup": dict(recipe.extra.get("p90p10_lookup", {})),
            "global_iqr": float(recipe.extra.get("global_iqr", 0.0)),
            "global_p90p10": float(recipe.extra.get("global_p90p10", 0.0)),
        },
    )


def _apply_target_aware_group_bin_recipe(recipe, X) -> np.ndarray:
    """Adapter: pulls the stored payload from ``recipe.extra`` and replays via
    :func:`apply_target_aware_group_bin`."""
    group_col = str(recipe.extra["group_col"])
    num_col = str(recipe.extra["num_col"])
    X_view = _coerce_X(X, group_col, num_col, recipe.name)
    return apply_target_aware_group_bin(
        X_view,
        {
            "group_col": group_col,
            "num_col": num_col,
            "group_edges": dict(recipe.extra.get("group_edges", {})),
            "global_edges": list(recipe.extra.get("global_edges", [])),
            "n_bins": int(recipe.extra.get("n_bins", 5)),
        },
    )


# ---------------------------------------------------------------------------
# MI-uplift scorer / gate
# ---------------------------------------------------------------------------


def score_grouped_quantile_by_mi_uplift(
    raw_X: pd.DataFrame,
    eng_X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_bins: int = 10,
    eng_to_source: Optional[dict] = None,
) -> pd.DataFrame:
    """Rank engineered distributional columns by ``MI(col; y)`` and gate by
    uplift over the source num_col's own marginal ``MI(num_col; y)``.

    Reuses the Layer 19 plug-in MI estimator (``_plug_in_mi``) with
    quantile-binning of continuous engineered columns. Returns a frame sorted
    by ``mi`` descending with ``[engineered_col, source_col, mi, source_mi,
    uplift]``.
    """
    from ._adaptive_nbins import _plug_in_mi

    if eng_X.shape[1] == 0:
        return pd.DataFrame(columns=["engineered_col", "source_col", "mi", "source_mi", "uplift"])
    y_arr = np.asarray(y)
    if not np.issubdtype(y_arr.dtype, np.integer):
        if y_arr.dtype.kind in "fc" and int(np.unique(y_arr).size) > 32:
            try:
                y_arr = pd.qcut(y_arr, q=10, labels=False, duplicates="drop").to_numpy()
            except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                logger.debug("suppressed in _grouped_quantile_fe.py:572: %s", e)
                pass
        _, y_arr = np.unique(y_arr, return_inverse=True)
    y_bin = y_arr.astype(np.int64)

    def _bin(arr: np.ndarray) -> np.ndarray:
        """Quantile-bin (or pass through as-is if already low-cardinality integer-valued) a column into ``n_bins`` int codes for the plug-in MI estimator."""
        a = np.asarray(arr, dtype=np.float64)
        # Compute the finite mask + finite subset ONCE (the original recomputed
        # np.isfinite(a) 4x and a[np.isfinite(a)] 3x). All-finite columns skip
        # the gather entirely. Bit-identical; ~1.18x at the L88 call volume.
        fin = np.isfinite(a)
        all_finite = fin.all()
        a_fin = a if all_finite else a[fin]
        # Integer-valued bin index columns (target_aware_bin) are used as-is.
        uniq = np.unique(a_fin)
        if uniq.size <= n_bins and a_fin.size and np.all(a_fin == np.round(a_fin)):
            if all_finite:
                out = a
            else:
                out = a.copy()
                out[~fin] = 0.0
            _, codes = np.unique(out, return_inverse=True)
            return codes.astype(np.int64)
        q = np.quantile(a_fin, np.linspace(0, 1, n_bins + 1)) if a_fin.size else np.array([0.0, 1.0])
        q = np.unique(q)
        if q.size < 2:
            return np.zeros(a.shape, dtype=np.int64)
        codes = np.searchsorted(q[1:-1], a, side="right")
        if not all_finite:
            codes[~fin] = 0
        return codes.astype(np.int64)

    eng_to_source = eng_to_source or {}
    source_mi_cache: dict[str, float] = {}

    def _source_mi(col: Optional[str]) -> float:
        """Cached ``MI(source_col; y)`` lookup, used as the baseline an engineered column must beat (the uplift gate)."""
        if not col or col not in raw_X.columns:
            return 0.0
        if col in source_mi_cache:
            return source_mi_cache[col]
        mi = _plug_in_mi(_bin(raw_X[col].to_numpy()), y_bin)
        source_mi_cache[col] = float(mi)
        return float(mi)

    rows = []
    for col in eng_X.columns:
        mi = float(_plug_in_mi(_bin(eng_X[col].to_numpy()), y_bin))
        src = eng_to_source.get(col)
        smi = _source_mi(src)
        rows.append({
            "engineered_col": col,
            "source_col": src,
            "mi": mi,
            "source_mi": smi,
            "uplift": mi - smi,
        })
    out = pd.DataFrame(rows)
    return out.sort_values("mi", ascending=False, kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Auto-detection helpers (shared shape with Layer 87)
# ---------------------------------------------------------------------------


def _auto_detect_group_cols(X: pd.DataFrame, max_cols: int = 4) -> list[str]:
    """Pick up to ``max_cols`` candidate grouping columns via the shared composite-target group-column detector, falling back to a low/mid-cardinality non-float scan when that detector is unavailable."""
    try:
        from ...training.composite import detect_group_column_candidates
        cands = detect_group_column_candidates(X)
        return [name for name, _info in cands[:max_cols]]
    except Exception as _e:
        logger.debug(
            "grouped_quantile auto-detect: detector import failed (%s); using " "fallback cardinality scan.",
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
    """Pick up to ``max_cols`` numeric source columns for grouped-quantile FE: floats always qualify, high-cardinality
    integers qualify, group columns and already-``grp``-prefixed engineered columns are excluded (see inline comment
    on why nesting on an engineered column would break replay)."""
    group_set = set(group_cols)
    out: list[str] = []
    for c in X.columns:
        if c in group_set:
            continue
        # Skip already-engineered grouped columns (grpagg/grpz/grpratio/grpiqr/grpp90p10/... appended by an EARLIER grouped-FE
        # stage). A per-group quantile of one of these builds a nested recipe whose transform-replay needs the intermediate
        # engineered column materialised first -- but transform() replays from raw X only, so it raises KeyError on the missing
        # source. The grouped aggregates are also constant within group, so a quantile of them is degenerate. Keep the source scope
        # to raw columns so every grouped-quantile recipe is 1-deep and replayable.
        if str(c).startswith("grp"):
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


# ---------------------------------------------------------------------------
# End-to-end pipeline with MI uplift gate
# ---------------------------------------------------------------------------


def hybrid_grouped_quantile_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    group_cols: Optional[Sequence[str]] = None,
    num_cols: Optional[Sequence[str]] = None,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    target_aware: bool = False,
    n_bins: int = 5,
    top_k: int = 8,
    n_bins_mi: int = 10,
    min_mi: float = 1e-4,
    min_uplift: float = 0.0,
    n_folds: int = 5,
    random_state: int = 0,
):
    """End-to-end per-group quantile / histogram FE pipeline.

    1. Auto-detect ``group_cols`` / ``num_cols`` when not supplied.
    2. Materialise per-group percentile-rank + IQR / p90-p10 spread; when
       ``target_aware`` also the OOF supervised per-group bin index.
    3. Score by ``MI(col; y)`` and gate by uplift over the source num_col
       marginal MI; keep the top ``top_k`` survivors.
    4. Append survivors to X; return ``(X_aug, appended, recipes, scores)``.

    ``y`` is consumed only by the MI gate and (when ``target_aware``) by the
    OOF bin fit; the persisted recipes carry no ``y`` reference, so transform()
    replay is leakage-free.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"hybrid_grouped_quantile_fe: X must be a pandas DataFrame; got " f"{type(X).__name__}")
    if group_cols is None or len(group_cols) == 0:
        group_cols = _auto_detect_group_cols(X)
    else:
        group_cols = [c for c in group_cols if c in X.columns]
    if not group_cols:
        return X, [], [], pd.DataFrame()
    if num_cols is None or len(num_cols) == 0:
        num_cols = _auto_detect_num_cols(X, group_cols)
    else:
        num_cols = [c for c in num_cols if c in X.columns]
    if not num_cols:
        return X, [], [], pd.DataFrame()

    enc_df, raw_recipes = generate_grouped_quantile_features(
        X, group_cols, num_cols, quantiles=quantiles,
    )
    if target_aware:
        tab_df, tab_recipes = generate_target_aware_group_bins(
            X, y, group_cols, num_cols, n_bins=n_bins,
            n_folds=n_folds, random_state=random_state,
        )
        if not tab_df.empty:
            enc_df = pd.concat([enc_df, tab_df], axis=1)
            raw_recipes.update(tab_recipes)
    if enc_df.empty:
        return X, [], [], pd.DataFrame()

    eng_to_source = {name: raw_recipes[name]["num_col"] for name in enc_df.columns}
    scores = score_grouped_quantile_by_mi_uplift(
        X, enc_df, y, n_bins=n_bins_mi, eng_to_source=eng_to_source,
    )
    keep = scores[(scores["mi"] >= float(min_mi)) & (scores["uplift"] >= float(min_uplift))]
    winners = list(keep["engineered_col"].head(int(top_k)))
    if not winners:
        return X, [], [], scores

    from .engineered_recipes import (
        build_grouped_quantile_recipe,
        build_target_aware_group_bin_recipe,
    )

    X_aug = pd.concat([X, enc_df[winners]], axis=1)
    recipes = []
    for name in winners:
        payload = raw_recipes[name]
        if payload.get("op") == "target_aware_bin":
            recipes.append(build_target_aware_group_bin_recipe(name=name, **payload))
        else:
            recipes.append(build_grouped_quantile_recipe(name=name, **payload))
    return X_aug, winners, recipes, scores
