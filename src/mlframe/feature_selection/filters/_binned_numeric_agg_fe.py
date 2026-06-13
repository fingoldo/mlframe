"""Grouped aggregation over CELLS of quantile-binned NUMERIC columns (2026-06-13).

The existing ``_grouped_agg_fe`` groups by low-cardinality CATEGORICAL columns. This sibling forms the group
key by QUANTILE-BINNING a numeric column (unsupervised, equal-frequency cells -> uniform per-cell sample size,
so higher moments are equally reliable) and aggregates ANOTHER numeric column's per-cell statistics
(mean / std / skew / kurt). It captures the regime where the WITHIN-CELL SPREAD / SHAPE of a feature carries
signal the cell mean cannot -- e.g. a heteroscedastic target whose variance, not mean, depends on the cell
(measured +0.9 OOS R2 on a sigma(cell) target where the cell mean is ~constant; bench_multistat_cell_encoding).

Design decisions (all measurement-backed, see ``_benchmarks/bench_cell_binning_for_moments``):
* UNSUPERVISED quantile binning, NOT MRMR's supervised MI binning -- supervised cell edges built from y would
  leak y into a feature aggregated to predict y, and MDLP gives uneven cells (tiny cells -> garbage moments).
* bin count = ``min(nbins_base, moment_stability_cap)`` where the cap = ``floor((n / n_min(highest_moment))^(1/k))``
  ties resolution to the highest requested moment (mean ~5, std ~12, skew ~30, kurt ~100 rows/cell). Freedman-
  Diaconis is REJECTED: it over-bins at large n and degrades (it optimises 1-D density, not per-cell occupancy).
* HIGH-MOMENT AUTO-DROP: when the cap forces fewer bins than ``nbins_base`` (small n / high moment), the
  high-order moments whose n_min cannot be met are dropped rather than coarsening every column -- an unreliable
  kurt at 3 bins is worse than no kurt.
* edges stored per group column for leak-safe transform replay (searchsorted), exactly like ``include_numeric``.
* vectorised via ``np.bincount`` raw-moment accumulation; returns numpy arrays (not lists -- see bench).
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_STATS = ("mean", "std", "skew", "kurt")
# Minimum rows-per-cell for a stable estimate of each moment order (rule-of-thumb, used by the moment cap).
_N_MIN = {"mean": 5, "std": 12, "skew": 30, "kurt": 100}

__all__ = [
    "SUPPORTED_STATS",
    "engineered_name_binned_agg",
    "quantile_edges",
    "resolve_nbins_and_stats",
    "per_cell_stats_bincount",
    "fit_binned_numeric_agg",
    "apply_binned_numeric_agg",
]


def engineered_name_binned_agg(num_col: str, group_col: str, stat: str) -> str:
    return f"binagg_{stat}({num_col}|qbin({group_col}))"


def quantile_edges(x: np.ndarray, nbins: int) -> np.ndarray:
    """Inner quantile cut points (unique-deduped); code = searchsorted(edges, v, side='right')."""
    qs = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    return np.unique(np.quantile(np.asarray(x, dtype=np.float64), qs))


def resolve_nbins_and_stats(n: int, stats: Sequence[str], nbins_base: int, k: int = 1) -> tuple:
    """Return (nbins, kept_stats): nbins = min(nbins_base, moment cap for the highest KEPT moment), dropping
    high-order moments whose per-cell sample floor cannot be met at any nbins >= 2 (HIGH-MOMENT AUTO-DROP)."""
    order = [s for s in SUPPORTED_STATS if s in stats]  # canonical order, low->high moment
    kept = list(order)
    # Drop the highest moments first while even nbins=2 would violate their n_min in a k-way cross.
    while kept:
        highest = kept[-1]
        cap = int(np.floor((n / _N_MIN[highest]) ** (1.0 / k)))
        if cap >= 2:
            nbins = max(2, min(int(nbins_base), cap))
            return nbins, kept
        kept.pop()  # even 2 bins can't satisfy this moment's floor -> drop it
    return 2, ["mean"]  # degenerate fallback


def per_cell_stats_bincount(codes: np.ndarray, v: np.ndarray, n_cells: int, stats: Sequence[str]) -> dict:
    """Vectorised per-cell statistics of ``v`` via raw-moment ``np.bincount`` (O(n), no Python per-row loop).
    Returns ``{stat: np.ndarray(n_cells)}``. Empty cells get NaN (caller substitutes the global value)."""
    cnt = np.bincount(codes, minlength=n_cells).astype(np.float64)
    safe = np.maximum(cnt, 1.0)
    s1 = np.bincount(codes, weights=v, minlength=n_cells)
    mean = s1 / safe
    out: dict = {}
    need_hi = any(s in ("std", "skew", "kurt") for s in stats)
    if need_hi:
        s2 = np.bincount(codes, weights=v * v, minlength=n_cells)
        m2 = np.maximum(s2 / safe - mean * mean, 0.0)
        std = np.sqrt(m2)
    for stat in stats:
        if stat == "mean":
            raw = mean
        elif stat == "std":
            raw = std
        elif stat == "skew":
            s3 = np.bincount(codes, weights=v ** 3, minlength=n_cells)
            m3 = s3 / safe - 3.0 * mean * (s2 / safe) + 2.0 * mean ** 3
            raw = np.where(std > 1e-9, m3 / (std ** 3 + 1e-12), 0.0)
        elif stat == "kurt":
            s3 = np.bincount(codes, weights=v ** 3, minlength=n_cells)
            s4 = np.bincount(codes, weights=v ** 4, minlength=n_cells)
            m4 = s4 / safe - 4.0 * mean * (s3 / safe) + 6.0 * mean ** 2 * (s2 / safe) - 3.0 * mean ** 4
            raw = np.where(m2 > 1e-12, m4 / (m2 * m2 + 1e-12) - 3.0, 0.0)
        else:
            raise ValueError(f"binned_numeric_agg stat {stat!r} not in {SUPPORTED_STATS}")
        out[stat] = np.where(cnt > 0, raw, np.nan)
    return out


def _global_stat(v: np.ndarray, stat: str) -> float:
    vf = v[np.isfinite(v)]
    if vf.size == 0:
        return 0.0
    if stat == "mean":
        return float(np.mean(vf))
    if stat == "std":
        return float(np.std(vf))
    from scipy.stats import kurtosis, skew
    sd = float(np.std(vf))
    if sd <= 1e-12:
        return 0.0
    if stat == "skew":
        return float(skew(vf)) if vf.size > 2 else 0.0
    if stat == "kurt":
        return float(kurtosis(vf)) if vf.size > 3 else 0.0
    return 0.0


def fit_binned_numeric_agg(
    X: pd.DataFrame, y: np.ndarray, *,
    group_num_cols: Sequence[str], agg_num_cols: Sequence[str],
    stats: Sequence[str] = SUPPORTED_STATS, nbins_base: int = 10,
    n_folds: int = 5, random_state: int = 0,
) -> tuple:
    """OOF fit of per-(quantile-cell) statistics of ``agg_num_cols`` grouped by quantile-binned ``group_num_cols``.

    Returns ``(feat_df, recipes)``: ``feat_df`` has one OOF column per (group, agg, kept_stat); ``recipes`` maps
    output-name -> dict carrying the group column's quantile ``edges`` + per-cell ``lookup`` (numpy array, indexed
    by bin code) + ``global`` fallback, so transform replays leak-free via ``apply_binned_numeric_agg``.
    """
    n = len(X)
    y_arr = np.asarray(y, dtype=np.float64).ravel()  # noqa: F841 (kept for parity / future y-aware gating)
    rng = np.random.default_rng(int(random_state))
    fold_ids = np.empty(n, dtype=np.int64)
    fold_ids[rng.permutation(n)] = np.arange(n) % int(n_folds)

    feat_cols: dict[str, np.ndarray] = {}
    recipes: dict[str, dict] = {}
    for gcol in group_num_cols:
        gvals = np.asarray(X[gcol].to_numpy(), dtype=np.float64)
        if not np.isfinite(gvals).all():
            continue  # v1 skips NaN-bearing group columns (quantile-edge replay has no NaN bin)
        nbins, kept_stats = resolve_nbins_and_stats(n, stats, nbins_base, k=1)
        edges = quantile_edges(gvals, nbins)
        if edges.size == 0:
            continue
        codes = np.searchsorted(edges, gvals, side="right")
        n_cells = int(codes.max()) + 1
        for acol in agg_num_cols:
            if acol == gcol:
                continue
            av = np.asarray(X[acol].to_numpy(), dtype=np.float64)
            finite = np.isfinite(av)
            globals_ = {s: _global_stat(av[finite], s) for s in kept_stats}
            oof = {s: np.full(n, globals_[s], dtype=np.float64) for s in kept_stats}
            for f in range(int(n_folds)):
                tr = (fold_ids != f) & finite
                if not tr.any():
                    continue
                per = per_cell_stats_bincount(codes[tr], av[tr], n_cells, kept_stats)
                test = np.where(fold_ids == f)[0]
                ct = codes[test]
                for s in kept_stats:
                    vals = per[s][ct]
                    oof[s][test] = np.where(np.isfinite(vals), vals, globals_[s])
            full = per_cell_stats_bincount(codes[finite], av[finite], n_cells, kept_stats)
            for s in kept_stats:
                name = engineered_name_binned_agg(acol, gcol, s)
                feat_cols[name] = oof[s]
                lut = np.where(np.isfinite(full[s]), full[s], globals_[s]).astype(np.float64)
                recipes[name] = {
                    "group_col": gcol, "agg_col": acol, "stat": s,
                    "edges": edges, "lookup": lut, "global": float(globals_[s]),
                }
    feat_df = pd.DataFrame(feat_cols, index=X.index)
    return feat_df, recipes


def apply_binned_numeric_agg(X: pd.DataFrame, recipe: dict) -> np.ndarray:
    """Leak-free replay: bin the raw group column through the stored quantile edges and gather the per-cell
    statistic; unseen / out-of-range / non-finite group values fall back to the global statistic."""
    gv = np.asarray(X[recipe["group_col"]].to_numpy(), dtype=np.float64)
    edges = np.asarray(recipe["edges"], dtype=np.float64)
    lut = np.asarray(recipe["lookup"], dtype=np.float64)
    g = float(recipe["global"])
    codes = np.searchsorted(edges, gv, side="right")
    codes = np.clip(codes, 0, lut.size - 1)
    out = lut[codes]
    out[~np.isfinite(gv)] = g
    return out


def build_binned_numeric_agg_recipe(name: str, info: dict):
    """Wrap a per-output fit ``info`` dict into a leak-safe ``EngineeredRecipe(kind='binned_numeric_agg')``.
    ``src_names = (group_col, agg_col)`` so the fit-end recipe router resolves both parents; the quantile
    ``edges`` + per-cell ``lookup`` + ``global`` fallback ride in ``extra`` for ``apply_binned_numeric_agg``."""
    from .engineered_recipes import EngineeredRecipe
    return EngineeredRecipe(
        name=name, kind="binned_numeric_agg",
        src_names=(info["group_col"], info["agg_col"]),
        extra={
            "group_col": info["group_col"], "agg_col": info["agg_col"], "stat": info["stat"],
            "edges": np.asarray(info["edges"], dtype=np.float64),
            "lookup": np.asarray(info["lookup"], dtype=np.float64),
            "global": float(info["global"]),
        },
    )


def _auto_detect_numeric_cols(X: pd.DataFrame, max_card_group: int = 10**9) -> list:
    """RAW numeric columns eligible as group / aggregate sources (finite, non-constant)."""
    out = []
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            continue
        v = X[c].to_numpy()
        if not np.isfinite(np.asarray(v, dtype=np.float64)).all():
            continue
        if np.unique(v).size < 2:
            continue
        out.append(c)
    return out


def _cheap_mi_with_y(col: np.ndarray, y_codes: np.ndarray, nbins: int = 10) -> float:
    """Cheap MI(qbin(col); y_codes) via a bincount joint histogram -- the relevance proxy for GROUP pre-selection.
    A group column only helps if its quantile cells separate y; this scores exactly that in O(n)."""
    cv = np.asarray(col, dtype=np.float64)
    if not np.isfinite(cv).all() or np.unique(cv).size < 2:
        return 0.0
    edges = quantile_edges(cv, nbins)
    if edges.size == 0:
        return 0.0
    xc = np.searchsorted(edges, cv, side="right")
    return float(compute_mi_from_codes(xc, y_codes))


def compute_mi_from_codes(a: np.ndarray, b: np.ndarray) -> float:
    """Plug-in MI of two integer-code arrays via a 2-D bincount joint histogram (nats)."""
    na, nb = int(a.max()) + 1, int(b.max()) + 1
    n = a.size
    joint = np.bincount(a * nb + b, minlength=na * nb).astype(np.float64).reshape(na, nb)
    pj = joint / n
    pa = pj.sum(axis=1, keepdims=True)
    pb = pj.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = pj * np.log(pj / (pa * pb))
    return float(np.nansum(np.where(pj > 0, term, 0.0)))


def binned_numeric_agg_with_recipes(
    X: pd.DataFrame, y: np.ndarray, *,
    group_num_cols: Sequence[str] = None, agg_num_cols: Sequence[str] = None,
    stats: Sequence[str] = SUPPORTED_STATS, nbins_base: int = 10,
    n_folds: int = 5, random_state: int = 0, max_pairs: int = 64,
    max_group_cols: int = 16, max_agg_cols: int = 16,
    mi_gate: bool = True, reject_sink=None,
) -> tuple:
    """End-to-end: relevance-select numeric group/aggregate columns, OOF-fit per-cell stats, append ``binagg_*``
    columns and build replay recipes. Returns ``(X_aug, appended, recipes)`` mirroring ``kfold_target_encode_with_recipes``.

    SCALABLE + CHEAP-PROBE selection (replaces the prior arbitrary top-K-by-variance, which did not scale past a
    handful of columns -- at 10k features the G*A pair space is ~1e8):
    * GROUP columns ranked by ``MI(qbin(g); y)`` -- a group key only helps if its cells separate y (O(p) cheap MIs);
    * AGG columns ranked by variance (a near-constant column has no per-cell shape to aggregate);
    * top ``max_group_cols`` x top ``max_agg_cols`` bounds the computed pair set; ``max_pairs`` caps it further;
    * PROBE-GATE (``mi_gate``, default ON): the emitted columns pass the shipped ``local_mi_gate`` MI floor, so a
      target with NO cell-conditional structure yields ZERO columns -- the family probes cheaply and exits without
      perturbing selection, which is what makes an ON-by-default flip safe. On a positive response it keeps the
      MI-relevant survivors (escalation is then just the standard MRMR screen over them)."""
    cols = _auto_detect_numeric_cols(X)
    gcands = [c for c in (group_num_cols or cols) if c in X.columns]
    acands = [c for c in (agg_num_cols or cols) if c in X.columns]
    if not gcands or not acands:
        return X.copy(), [], []

    # GROUP pre-selection by cheap MI(qbin(g); y). y discretised once (deciles) for the relevance proxy.
    y_arr = np.asarray(y.to_numpy() if hasattr(y, "to_numpy") else y, dtype=np.float64).ravel()
    if np.unique(y_arr).size <= 20:  # already categorical / few classes
        _, y_codes = np.unique(y_arr, return_inverse=True)
    else:
        y_codes = np.searchsorted(quantile_edges(y_arr, 10), y_arr, side="right")
    g_mi = {g: _cheap_mi_with_y(X[g].to_numpy(), y_codes) for g in gcands}
    gsel = sorted([g for g in gcands if g_mi[g] > 0.0], key=lambda g: g_mi[g], reverse=True)[: max(1, int(max_group_cols))]
    # AGG pre-selection by variance (unsupervised -- the aggregated column needs spread to have per-cell shape).
    a_var = {a: float(np.var(np.asarray(X[a].to_numpy(), dtype=np.float64))) for a in acands}
    asel = sorted(acands, key=lambda a: a_var.get(a, 0.0), reverse=True)[: max(1, int(max_agg_cols))]
    if not gsel or not asel:
        return X.copy(), [], []

    feat_df, raw = fit_binned_numeric_agg(
        X, y, group_num_cols=gsel, agg_num_cols=asel,
        stats=stats, nbins_base=nbins_base, n_folds=n_folds, random_state=random_state,
    )
    if feat_df.shape[1] == 0:
        return X.copy(), [], []
    # max_pairs cap on the (group, agg) pairs actually emitted, by descending group MI then agg variance.
    pair_rank = {(g, a): (g_mi.get(g, 0.0), a_var.get(a, 0.0)) for g in gsel for a in asel}
    names_by_pair = {}
    for nm in feat_df.columns:
        names_by_pair.setdefault((raw[nm]["group_col"], raw[nm]["agg_col"]), []).append(nm)
    top_pairs = sorted(names_by_pair, key=lambda p: pair_rank.get(p, (0.0, 0.0)), reverse=True)[: max(1, int(max_pairs))]
    keep = [nm for p in top_pairs for nm in names_by_pair[p]]
    feat_df = feat_df[keep]

    # PROBE-GATE: keep only columns clearing the local MI floor vs the raw baseline (cheap exit when no signal).
    if mi_gate and feat_df.shape[1] > 0:
        from ._unified_fe_gate import local_mi_gate
        survivors = set(local_mi_gate(feat_df, y, raw_X=X, reject_sink=reject_sink))
        feat_df = feat_df[[c for c in feat_df.columns if c in survivors]]
        if feat_df.shape[1] == 0:
            return X.copy(), [], []

    X_aug = pd.concat([X, feat_df], axis=1)
    recipes = [build_binned_numeric_agg_recipe(n, raw[n]) for n in feat_df.columns]
    return X_aug, list(feat_df.columns), recipes
