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
    pairs: "Optional[set]" = None,
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
            if pairs is not None and (gcol, acol) not in pairs:
                continue  # PRE-CAP: only compute OOF for the kept top-max_pairs (bit-identical output)
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
    mi_gate: bool = True, redundancy_gate: bool = True,
    min_cmi_gain: float = 0.005, reject_sink=None,
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

    # PRE-CAP (2026-06-17 perf): the OOF fit below previously computed all gsel x asel pairs and only
    # then capped to top-``max_pairs`` by ``pair_rank`` (group MI, then agg variance) -- both already
    # known here, BEFORE any OOF work. Rank + cap the (group, agg) pairs up front and compute OOF for
    # only those, so per_cell_stats_bincount runs ``max_pairs`` times instead of |gsel|*|asel| (e.g.
    # 64 vs 256). Output is BIT-IDENTICAL: the same top pairs are emitted with the same OOF values; the
    # post-fit cap below stays as a no-op safety net. (NaN-bearing group cols are skipped inside the fit
    # exactly as before.)
    _ranked_pairs = sorted(
        ((g, a) for g in gsel for a in asel if g != a),
        key=lambda p: (g_mi.get(p[0], 0.0), a_var.get(p[1], 0.0)), reverse=True,
    )
    _precap_pairs = set(_ranked_pairs[: max(1, int(max_pairs))])
    feat_df, raw = fit_binned_numeric_agg(
        X, y, group_num_cols=gsel, agg_num_cols=asel,
        stats=stats, nbins_base=nbins_base, n_folds=n_folds, random_state=random_state,
        pairs=_precap_pairs,
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

    # REDUNDANCY GATE: a ``binagg_*`` column ``stat(a | qbin(g))`` is a deterministic function of its source
    # columns ``(g, a)``. The Tier-1 MI floor above keeps it whenever MI(col; y) clears the raw noise floor,
    # but that fires even when the column carries NO information about y beyond ``g``/``a`` themselves -- e.g. on a
    # linearly-separable target where the raw source already explains y, the binned aggregate is a redundant
    # re-encoding of raw signal. Keep a column only when ``CMI(col; y | g, a) >= min_cmi_gain``, conditioning on
    # its OWN sources (cheap: at most two raw columns). Collapses spurious appends to zero on data with no
    # residual per-cell structure while preserving genuine cell-conditional features (where the per-cell shape
    # adds information the raw marginals cannot).
    if redundancy_gate and feat_df.shape[1] > 0:
        from ._mi_greedy_cmi_fe import _cmi_from_binned, _quantile_bin, _renumber_joint
        from ._unified_fe_gate import _coerce_y_classes

        y_cls = _coerce_y_classes(y)
        _src_bin_cache: dict[str, np.ndarray] = {}
        # Permutation-null floor. The plug-in CMI estimator is positively biased at finite n: a binagg column that is
        # genuinely redundant with its sources still scores a small POSITIVE CMI that a fixed threshold cannot separate
        # from real signal (the bias grows as n shrinks / the conditioning support fragments, and the OOF per-fold
        # aggregate adds further sampling noise). Calibrate per-candidate: score the SAME candidate against shuffled-y
        # under the same conditioning; the max over a handful of permutations is the candidate's own noise ceiling.
        # Keep it only when its observed CMI clears BOTH the absolute floor AND that ceiling -- genuine cell-conditional
        # signal sits far above the null, redundant re-encodings sit at it.
        _n_perm = 15
        _rng = np.random.default_rng(int(random_state))

        def _src_bins(col: str) -> np.ndarray:
            b = _src_bin_cache.get(col)
            if b is None:
                b = _quantile_bin(X[col].to_numpy(dtype=np.float64), nbins=nbins_base)
                _src_bin_cache[col] = b
            return b

        kept_cols = []
        for nm in feat_df.columns:
            srcs = [c for c in (raw[nm].get("group_col"), raw[nm].get("agg_col")) if c in X.columns]
            z_joint = _renumber_joint(*[_src_bins(c) for c in srcs])[0] if srcs else None
            cand_bin = _quantile_bin(feat_df[nm].to_numpy(dtype=np.float64), nbins=nbins_base)
            cmi = _cmi_from_binned(cand_bin, y_cls, z_joint)
            null_ceiling = 0.0
            if np.isfinite(cmi) and cmi >= float(min_cmi_gain):
                for _ in range(_n_perm):
                    yp = y_cls[_rng.permutation(y_cls.shape[0])]
                    c0 = _cmi_from_binned(cand_bin, yp, z_joint)
                    if np.isfinite(c0) and c0 > null_ceiling:
                        null_ceiling = c0
            keep = np.isfinite(cmi) and cmi >= float(min_cmi_gain) and cmi > null_ceiling
            if keep:
                kept_cols.append(nm)
            elif reject_sink is not None:
                try:
                    reject_sink(
                        gate="binagg_source_redundancy", candidate=str(nm),
                        operands=tuple(srcs), operator="binned_numeric_agg_redundancy_gate",
                        observed=float(cmi) if np.isfinite(cmi) else 0.0,
                        threshold=max(float(min_cmi_gain), float(null_ceiling)),
                        reason="binagg CMI about y given its sources does not clear the permutation-null ceiling",
                    )
                except Exception:
                    pass
        feat_df = feat_df[kept_cols]
        if feat_df.shape[1] == 0:
            return X.copy(), [], []

    X_aug = pd.concat([X, feat_df], axis=1)
    recipes = [build_binned_numeric_agg_recipe(n, raw[n]) for n in feat_df.columns]
    return X_aug, list(feat_df.columns), recipes
