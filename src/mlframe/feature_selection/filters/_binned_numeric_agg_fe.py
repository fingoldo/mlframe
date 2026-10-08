"""Grouped aggregation over CELLS of quantile-binned NUMERIC columns.

The existing ``_grouped_agg_fe`` groups by low-cardinality CATEGORICAL columns. This sibling forms the group
key by QUANTILE-BINNING a numeric column (unsupervised, equal-frequency cells -> uniform per-cell sample size,
so higher moments are equally reliable) and aggregates ANOTHER numeric column's per-cell statistics
(mean / std / skew / kurt). It captures the regime where the WITHIN-CELL SPREAD / SHAPE of a feature carries
signal the cell mean cannot - e.g. a heteroscedastic target whose variance, not mean, depends on the cell
(measured +0.9 OOS R2 on a sigma(cell) target where the cell mean is ~constant; bench_multistat_cell_encoding).

Design decisions (all measurement-backed, see ``_benchmarks/bench_cell_binning_for_moments``):
* UNSUPERVISED quantile binning, NOT MRMR's supervised MI binning - supervised cell edges built from y would
  leak y into a feature aggregated to predict y, and MDLP gives uneven cells (tiny cells -> garbage moments).
* bin count = ``min(nbins_base, moment_stability_cap)`` where the cap = ``floor((n / n_min(highest_moment))^(1/k))``
  ties resolution to the highest requested moment (mean ~5, std ~12, skew ~30, kurt ~100 rows/cell). Freedman-
  Diaconis is REJECTED: it over-bins at large n and degrades (it optimises 1-D density, not per-cell occupancy).
* HIGH-MOMENT AUTO-DROP: when the cap forces fewer bins than ``nbins_base`` (small n / high moment), the
  high-order moments whose n_min cannot be met are dropped rather than coarsening every column - an unreliable
  kurt at 3 bins is worse than no kurt.
* edges stored per group column for leak-safe transform replay (searchsorted), exactly like ``include_numeric``.
* vectorised via ``np.bincount`` raw-moment accumulation; returns numpy arrays (not lists - see bench).
"""
from __future__ import annotations

import logging

from collections.abc import Sequence
from typing import Optional, Any

import numpy as np
import pandas as pd
from numba import njit

from ._fast_host_ops import searchsorted_right  # multi-core bin search for the large group columns
from ._binned_agg_cheap_mi import (  # noqa: F401  -- carved sibling, re-exported
    _DEVICE_QUANTILE_MIN_N,
    _cheap_mi_batch_njit,
    _cheap_mi_edge_dedup_njit,
    _cheap_mi_group_selection,
    _cheap_mi_with_y,
    compute_mi_from_codes,
    quantile_edges,
)
from ._binned_numeric_agg_cands import DeviceOofCandidates, HostCandidates, device_born_candidates

logger = logging.getLogger(__name__)


@njit(cache=True)
def _per_cell_count_sum_njit(codes, v, n_cells):
    """Per-cell ``(cnt, s1)`` only -- pass 1 of the two-pass stable scheme, with the dead higher moments gone.

    Accumulates in the same row order as :func:`_per_cell_raw_moments_njit`, so ``cnt`` and ``s1`` are
    bit-identical to that kernel's first two outputs.
    """
    n_cells = int(n_cells)
    cnt = np.zeros(n_cells, dtype=np.float64)
    s1 = np.zeros(n_cells, dtype=np.float64)
    for i in range(codes.shape[0]):
        c = codes[i]
        cnt[c] += 1.0
        s1[c] += v[i]
    return cnt, s1


@njit(cache=True)
def _per_cell_raw_moments_njit(codes, v, n_cells):
    """One-pass per-cell raw moment accumulator: returns ``(cnt, s1, s2, s3, s4)`` each ``(n_cells,)``.

    Replaces the per-cell-stats path's FOUR separate ``np.bincount(weights=v**k)`` passes (each a full
    array power + histogram) with a SINGLE O(n) walk. ``s_k = sum(v**k)`` per cell; the powers are built
    by repeated multiplication (``x2=x*x``; ``s3+=x2*x``; ``s4+=x2*x2``) - matches ``np.bincount(v*v)``
    for s2 exactly; s3/s4 differ from numpy ``v**3``/``v**4`` only at the last ULP, far below the
    quantile-bin resolution the moments feed, so the engineered codes are unchanged."""
    n_cells = int(n_cells)
    cnt = np.zeros(n_cells, dtype=np.float64)
    s1 = np.zeros(n_cells, dtype=np.float64)
    s2 = np.zeros(n_cells, dtype=np.float64)
    s3 = np.zeros(n_cells, dtype=np.float64)
    s4 = np.zeros(n_cells, dtype=np.float64)
    for i in range(codes.shape[0]):
        c = codes[i]
        x = v[i]
        x2 = x * x
        cnt[c] += 1.0
        s1[c] += x
        s2[c] += x2
        s3[c] += x2 * x
        s4[c] += x2 * x2
    return cnt, s1, s2, s3, s4

@njit(cache=True)
def _per_cell_centered_moments_njit(codes, v, cell_mean, n_cells):
    """One-pass per-cell CENTERED moment accumulator: given each cell's already-computed mean (indexed
    by cell id), returns ``(cm2, cm3, cm4)`` each ``(n_cells,)`` -- ``sum((x-mean_c)^k)`` per cell for
    k=2,3,4. Numerically stable (no raw-power cancellation) -- the whole-array analog of
    :func:`_centered_moments_njit`, whose own docstring documents the measured catastrophic failure of
    the raw-moment binomial-expansion form this replaces in :func:`_derive_cell_stats`."""
    n_cells = int(n_cells)
    cm2 = np.zeros(n_cells, dtype=np.float64)
    cm3 = np.zeros(n_cells, dtype=np.float64)
    cm4 = np.zeros(n_cells, dtype=np.float64)
    for i in range(codes.shape[0]):
        c = codes[i]
        d = v[i] - cell_mean[c]
        d2 = d * d
        cm2[c] += d2
        cm3[c] += d2 * d
        cm4[c] += d2 * d2
    return cm2, cm3, cm4


def _per_cell_moments_stable(codes: np.ndarray, v: np.ndarray, n_cells: int) -> tuple:
    """Per-cell ``(cnt, mean, cm2, cm3, cm4)`` via the numerically-stable two-pass scheme: pass 1
    (:func:`_per_cell_raw_moments_njit`, cnt/s1 only used) gets each cell's mean -- a plain additive sum,
    safe from cancellation; pass 2 (:func:`_per_cell_centered_moments_njit`) accumulates CENTERED powers
    directly. Feeds :func:`_derive_cell_stats`."""
    codes_i = np.ascontiguousarray(codes, dtype=np.int64)
    v_f = np.ascontiguousarray(v, dtype=np.float64)
    # The pruned twin: this pass needs only `cnt` and `s1`, and the full kernel also accumulated s2, s3 and s4
    # -- 2.5x the arithmetic and three unused `np.zeros(n_cells)` per call, on a kernel `fit_binned_numeric_agg`
    # runs once per fold per (group_col, agg_col) pair. Bit-identical by construction: same row order, same adds.
    # The GPU twin already bincounts only cnt and s1, so this was a host-only gap.
    cnt, s1 = _per_cell_count_sum_njit(codes_i, v_f, int(n_cells))
    mean = s1 / np.maximum(cnt, 1.0)
    cm2, cm3, cm4 = _per_cell_centered_moments_njit(codes_i, v_f, mean, int(n_cells))
    return cnt, mean, cm2, cm3, cm4


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
    """Canonical engineered-feature name for a binned-numeric aggregate, e.g. ``binagg_mean(price|qbin(area))``."""
    return f"binagg_{stat}({num_col}|qbin({group_col}))"


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


def _derive_cell_stats(cnt: np.ndarray, mean: np.ndarray, cm2: np.ndarray, cm3: np.ndarray, cm4: np.ndarray, stats: Sequence[str]) -> dict:
    """Derive per-cell statistics from ``(cnt, mean, cm2, cm3, cm4)`` -- CENTERED moment sums (see
    :func:`_per_cell_moments_stable` / :func:`_per_cell_centered_moments_njit`), not the raw-power form this
    replaced: the raw-power binomial-expansion derivation (``s3/n - 3*mean*s2/n + 2*mean**3``)
    is catastrophically unstable on large-offset/small-scale columns -- the exact bug class already fixed for
    the whole-column global stats (:func:`_global_stats_all`) and target-encoding's per-category moments
    (``_target_encoding_fe.py``), confirmed live here too via a 1e13-scale skew/kurt error on synthetic data
    with a large per-cell offset. No ``+1e-12`` epsilon pad on the skew/kurt denominators either (that pad
    corrupts an already-small-but-correctly-computed denominator by ~30-100% once cancellation itself is
    fixed -- see the target-encoding fix's own docstring for the same finding); the ``std > 1e-9`` /
    ``m2 > 1e-12`` guards already bound the denominator away from true zero.

    Empty cells get NaN (caller substitutes the global value)."""
    safe = np.maximum(cnt, 1.0)
    out: dict = {}
    need_hi = any(s in ("std", "skew", "kurt") for s in stats)
    if need_hi:
        m2 = cm2 / safe
        std = np.sqrt(np.maximum(m2, 0.0))
    for stat in stats:
        if stat == "mean":
            raw = mean
        elif stat == "std":
            raw = std
        elif stat == "skew":
            m3 = cm3 / safe
            # np.where evaluates BOTH branches elementwise before selecting, so m3/std**3 still runs
            # (and warns) on the std<=1e-9 cells even though their result is discarded -- suppress the
            # resulting divide/invalid RuntimeWarning locally rather than at every one of this
            # function's callers; the discarded values are never read.
            with np.errstate(divide="ignore", invalid="ignore"):
                raw = np.where(std > 1e-9, m3 / std**3, 0.0)
        elif stat == "kurt":
            m4 = cm4 / safe
            with np.errstate(divide="ignore", invalid="ignore"):
                raw = np.where(m2 > 1e-12, m4 / (m2 * m2) - 3.0, 0.0)
        else:
            raise ValueError(f"binned_numeric_agg stat {stat!r} not in {SUPPORTED_STATS}")
        out[stat] = np.where(cnt > 0, raw, np.nan)
    return out


def per_cell_stats_bincount(codes: np.ndarray, v: np.ndarray, n_cells: int, stats: Sequence[str]) -> dict:
    """Vectorised per-cell statistics of ``v`` via one-pass njit centered-moment accumulation (O(n), no
    Python per-row loop, no ``np.bincount``). Returns ``{stat: np.ndarray(n_cells)}``. Empty cells get NaN
    (caller substitutes the global value). Two njit passes (mean, then centered powers) -- numerically
    stable, see :func:`_derive_cell_stats`."""
    cnt, mean, cm2, cm3, cm4 = _per_cell_moments_stable(codes, v, n_cells)
    return _derive_cell_stats(cnt, mean, cm2, cm3, cm4, stats)


def _global_stat(v: np.ndarray, stat: str) -> float:
    """Fallback statistic computed over the whole (finite-valued) column, used to fill empty bin cells at fit/apply time."""
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


@njit(cache=True)
def _centered_moments_njit(vf: np.ndarray) -> tuple:
    """Two-pass (mean, then centred sums) computation of ``(mean, sum((x-mean)^2), sum((x-mean)^3),
    sum((x-mean)^4))`` over the WHOLE array in one njit call.

    bench-attempt-rejected (2026-08-04): a single-pass RAW-moment form (``sum(x)``/``sum(x**2)``/
    ``sum(x**3)``/``sum(x**4)``, deriving centred moments via the textbook binomial expansion - the same
    approach :func:`_derive_cell_stats` uses for per-cell stats) was measured CATASTROPHICALLY WRONG on
    columns with a large mean relative to their spread (e.g. offset~1e4, std~1e-3): skew/kurt errors up to
    15 orders of magnitude on synthetic data, from the classic large-nearly-equal-numbers cancellation in
    ``sum(x**k) - k*mean*sum(x**(k-1)) + ...``. This two-pass form (mean first, then accumulate CENTRED
    powers ``(x-mean)**k`` directly - no expansion, no cancellation) is what scipy's own skew/kurtosis
    implementation does internally, and is what this function replicates. Still only 2 full traversals
    instead of 4 (one per separately-called stat), the real win this fusion targets."""
    n = vf.shape[0]
    s = 0.0
    for i in range(n):
        s += vf[i]
    mean = s / n
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0
    for i in range(n):
        d = vf[i] - mean
        d2 = d * d
        s2 += d2
        s3 += d2 * d
        s4 += d2 * d2
    return mean, s2, s3, s4


def _global_stats_all(v: np.ndarray, needed: Sequence[str]) -> dict:
    """Fused replacement for calling :func:`_global_stat` separately per stat.

    ``{s: _global_stat(v, s) for s in needed}`` pays one INDEPENDENT full-array pass per stat -
    ``np.mean``, ``np.std`` (recomputes mean internally), ``scipy.stats.skew``/``kurtosis`` (each
    recomputes mean + variance internally) - up to 4 full traversals of the same column for the
    ``SUPPORTED_STATS`` default. This instead calls :func:`_centered_moments_njit` (mean pass + one fused
    centred-power pass) - 2 traversals total, numerically stable (see that function's docstring for the
    catastrophic-cancellation failure of the naive raw-moment-expansion alternative). Verified against the
    original per-stat calls across 30 synthetic scenarios spanning extreme scale/offset combinations, incl.
    NaN-mixed and constant columns."""
    vf = v[np.isfinite(v)]
    n = vf.shape[0]
    if n == 0:
        return {s: 0.0 for s in needed}
    if not any(s in ("std", "skew", "kurt") for s in needed):
        # mean-only request: skip the centred-moment kernel entirely (no second pass needed).
        return {"mean": float(np.mean(vf))}
    vmin = float(vf.min())
    vmax = float(vf.max())
    if vmin == vmax:
        # Constant-column fast path: sequential float summation of n copies of the SAME value is not
        # always bit-exact to n*value (accumulator rounding drift for large n / certain values), which can
        # leave `mean` a few ULP off `value` and inflate std from a true 0 to a tiny-but->1e-12 epsilon on
        # a huge-offset column, blowing up skew/kurt as (near-zero)/(near-zero)^k. Skip the kernel and
        # numpy's own mean/std entirely; this matches BOTH exactly by construction (variance of a constant
        # set is exactly 0, no accumulation involved).
        out_const: dict = {}
        if "mean" in needed:
            out_const["mean"] = vmin
        if "std" in needed:
            out_const["std"] = 0.0
        if "skew" in needed:
            out_const["skew"] = 0.0
        if "kurt" in needed:
            out_const["kurt"] = 0.0
        return out_const
    mean, s2, s3, s4 = _centered_moments_njit(np.ascontiguousarray(vf, dtype=np.float64))
    out: dict = {}
    if "mean" in needed:
        out["mean"] = mean
    var = s2 / n
    std = float(var**0.5)
    if "std" in needed:
        out["std"] = std
    if std <= 1e-12:
        if "skew" in needed:
            out["skew"] = 0.0
        if "kurt" in needed:
            out["kurt"] = 0.0
        return out
    if "skew" in needed:
        m3 = s3 / n
        out["skew"] = (m3 / std**3) if n > 2 else 0.0
    if "kurt" in needed:
        m4 = s4 / n
        out["kurt"] = (m4 / (var * var) - 3.0) if n > 3 else 0.0
    return out


def fit_binned_numeric_agg(
    X: pd.DataFrame, y: np.ndarray, *,
    group_num_cols: Sequence[str], agg_num_cols: Sequence[str],
    stats: Sequence[str] = SUPPORTED_STATS, nbins_base: int = 10,
    n_folds: int = 5, random_state: int = 0,
    pairs: "Optional[set]" = None,
    recipe_only: bool = False,
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
    # Per-agg-column caches, shared ACROSS group columns: the agg values / finite mask / GLOBAL stats depend
    # only on ``acol`` yet were recomputed for every (gcol, acol) pair - and ``_global_stat``'s scipy
    # skew/kurtosis do several full-n passes each, so at 16 group columns the same column's globals ran up to
    # 16x (the dominant host cost of the recipe fit at 1M rows). Bit-identical values, just computed once.
    _av_cache: dict[str, tuple] = {}
    _globals_cache: dict[tuple, dict] = {}
    # Fold membership depends ONLY on ``f`` (not on gcol/acol), yet the per-(gcol, acol) OOF loop below
    # recomputed ``np.where(fold_ids == f)`` every pair*fold. Precompute it ONCE per call (bit-identical -
    # same indices, just hoisted out of the pair loops).
    _fold_test = None if recipe_only else [np.where(fold_ids == f)[0] for f in range(int(n_folds))]
    from ._fe_deadline import fe_deadline_passed

    for gcol in group_num_cols:
        # Optional-enrichment wall-clock budget: stop the (group_col, agg_col) sweep once MRMR.fit's
        # deadline passes; return whatever columns/recipes were engineered so far. No-op without a budget
        # (mirrors the orth-univariate/pair-cross/extra-basis generators' internal deadline check).
        if fe_deadline_passed():
            break
        gvals = np.asarray(X[gcol].to_numpy(), dtype=np.float64)
        if not np.isfinite(gvals).all():
            continue  # v1 skips NaN-bearing group columns (quantile-edge replay has no NaN bin)
        nbins, kept_stats = resolve_nbins_and_stats(n, stats, nbins_base, k=1)
        edges = quantile_edges(gvals, nbins)
        if edges.size == 0:
            continue
        codes = searchsorted_right(edges, gvals)
        n_cells = int(codes.max()) + 1
        # ``codes[test]`` depends on (gcol, f) but NOT acol - hoist it out of the acol loop.
        _ct_by_fold = None if recipe_only else [codes[_ft] for _ft in _fold_test]  # type: ignore[union-attr]  # _fold_test is non-None exactly when recipe_only is False (same condition as this ternary)
        for acol in agg_num_cols:
            if acol == gcol:
                continue
            if pairs is not None and (gcol, acol) not in pairs:
                continue  # PRE-CAP: only compute OOF for the kept top-max_pairs (bit-identical output)
            _avc = _av_cache.get(acol)
            if _avc is None:
                av = np.asarray(X[acol].to_numpy(), dtype=np.float64)
                finite = np.isfinite(av)
                finite_count = int(np.count_nonzero(finite))
                _av_cache[acol] = (av, finite, finite_count)
            else:
                av, finite, finite_count = _avc
            _gk = (acol, tuple(kept_stats))
            globals_ = _globals_cache.get(_gk)
            if globals_ is None:
                globals_ = _global_stats_all(av[finite], kept_stats)
                _globals_cache[_gk] = globals_
            # Full-data moments, needed anyway for the ``full``/``lut`` recipe lookup below.
            full_cnt, full_mean, full_cm2, full_cm3, full_cm4 = _per_cell_moments_stable(codes[finite], av[finite], n_cells)
            if not recipe_only:
                # RECIPE_ONLY (device-born binagg) skips the 5-fold OOF feat-column build - the
                # per-fold gather + np.where over the full n rows, the FE scan's single largest GPU-idle host
                # stage. The device-born path (binned_numeric_agg_with_recipes) fits recipes-only, gates on the
                # device from those recipes, then builds the OOF for the FEW survivors - so the OOF of the
                # dropped candidates is never computed. The ``full`` per-cell lookup + globals (the recipe
                # fields) are cheap 1-pass njit and are always built.
                assert _fold_test is not None and _ct_by_fold is not None  # populated whenever recipe_only is False
                oof = {s: np.full(n, globals_[s], dtype=np.float64) for s in kept_stats}
                finite_idx = np.where(finite)[0]
                fold_of_finite = fold_ids[finite_idx]
                for f in range(int(n_folds)):
                    test = _fold_test[f]
                    ct = _ct_by_fold[f]
                    test_fin = test[finite[test]]
                    # Equivalent to a ``(fold_ids != f) & finite`` full-array-AND + ``.any()`` gate (an
                    # O(n) scan done once per (gcol, acol, fold), G*A*n_folds times total) without
                    # materialising it: train has zero finite rows iff ALL finite rows fell into this
                    # fold's test set, i.e. ``test_fin`` (already computed, O(n/n_folds)) covers every
                    # finite row.
                    if test_fin.size == finite_count:
                        continue
                    # CENTERED moments are NOT additive across row subsets (a subset's own mean differs
                    # from the full-data mean, so full - test is invalid for cm2/cm3/cm4, unlike the old
                    # raw-power form) -- compute TRAIN directly on its own rows instead of full-minus-test
                    # (mirrors the same correctness-over-the-old-buggy-optimization tradeoff already made
                    # for target-encoding's per-category moments -- see _target_encoding_fe.py).
                    train_fin_idx = finite_idx[fold_of_finite != f]
                    t_cnt, t_mean, t_cm2, t_cm3, t_cm4 = _per_cell_moments_stable(codes[train_fin_idx], av[train_fin_idx], n_cells)
                    per = _derive_cell_stats(t_cnt, t_mean, t_cm2, t_cm3, t_cm4, kept_stats)
                    for s in kept_stats:
                        vals = per[s][ct]
                        oof[s][test] = np.where(np.isfinite(vals), vals, globals_[s])
            full = _derive_cell_stats(full_cnt, full_mean, full_cm2, full_cm3, full_cm4, kept_stats)
            for s in kept_stats:
                name = engineered_name_binned_agg(acol, gcol, s)
                if not recipe_only:
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


def _auto_detect_numeric_cols(X: pd.DataFrame) -> list:
    """RAW numeric columns eligible as group / aggregate sources (finite, non-constant)."""
    out = []
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            continue
        v = X[c].to_numpy()
        if not np.isfinite(np.asarray(v, dtype=np.float64)).all():
            continue
        # non-constant check: min != max on an all-finite column == np.unique(v).size >= 2, without the
        # full-n unique SORT (O(n) vs O(n log n); the sort was a measurable host sink at 1M x many columns).
        if np.min(v) == np.max(v):
            continue
        out.append(c)
    return out


def _family_wise_null_z(n_candidates: int, alpha: float = 0.05, min_z: float = 2.0) -> float:
    """One-sided normal quantile at the Bonferroni level ``alpha / n_candidates``, floored at ``min_z``."""
    from scipy.stats import norm

    return max(float(min_z), float(norm.isf(float(alpha) / max(1, int(n_candidates)))))


def binned_numeric_agg_with_recipes(
    X: pd.DataFrame, y: np.ndarray, *,
    group_num_cols: Sequence[str] | None = None, agg_num_cols: Sequence[str] | None = None,
    stats: Sequence[str] = SUPPORTED_STATS, nbins_base: int = 10,
    n_folds: int = 5, random_state: int = 0, max_pairs: int = 64,
    max_group_cols: int = 16, max_agg_cols: int = 16,
    mi_gate: bool = True, redundancy_gate: bool = True,
    min_cmi_gain: float = 0.005, reject_sink=None,
) -> tuple:
    """End-to-end: relevance-select numeric group/aggregate columns, OOF-fit per-cell stats, append ``binagg_*``
    columns and build replay recipes. Returns ``(X_aug, appended, recipes)`` mirroring ``kfold_target_encode_with_recipes``.

    SCALABLE + CHEAP-PROBE selection (replaces the prior arbitrary top-K-by-variance, which did not scale past a
    handful of columns - at 10k features the G*A pair space is ~1e8):
    * GROUP columns ranked by ``MI(qbin(g); y)`` - a group key only helps if its cells separate y (O(p) cheap MIs);
    * AGG columns ranked by variance (a near-constant column has no per-cell shape to aggregate);
    * top ``max_group_cols`` x top ``max_agg_cols`` bounds the computed pair set; ``max_pairs`` caps it further;
    * PROBE-GATE (``mi_gate``, default ON): the emitted columns pass the shipped ``local_mi_gate`` MI floor, so a
      target with NO cell-conditional structure yields ZERO columns - the family probes cheaply and exits without
      perturbing selection, which is what makes an ON-by-default flip safe. On a positive response it keeps the
      MI-relevant survivors (escalation is then just the standard MRMR screen over them)."""
    cols = _auto_detect_numeric_cols(X)
    gcands = [c for c in (group_num_cols or cols) if c in X.columns]
    acands = [c for c in (agg_num_cols or cols) if c in X.columns]
    if not gcands or not acands:
        return X, [], []

    # GROUP pre-selection by cheap MI(qbin(g); y). y discretised once (deciles) for the relevance proxy.
    y_arr = np.asarray(y.to_numpy() if hasattr(y, "to_numpy") else y, dtype=np.float64).ravel()
    if np.unique(y_arr).size <= 20:  # already categorical / few classes
        _, y_codes = np.unique(y_arr, return_inverse=True)
    else:
        y_codes = np.searchsorted(quantile_edges(y_arr, 10), y_arr, side="right")
    g_mi = _cheap_mi_group_selection(X, gcands, y_codes)
    gsel = sorted([g for g in gcands if g_mi[g] > 0.0], key=lambda g: g_mi[g], reverse=True)[: max(1, int(max_group_cols))]
    # AGG pre-selection by variance (unsupervised - the aggregated column needs spread to have per-cell shape).
    # Vectorised over columns (one np.var(axis=0) call) instead of a per-column Python loop -- each column's
    # variance is independent of every other, the classic "fuse into one batched call" pattern (measured
    # 1.94x at p=300 cols/n=1500, max diff 3.3e-15 vs the per-column loop -- machine-epsilon float-summation-
    # order noise, not a behavioural change; only feeds a descending sort so exact equality isn't even load-
    # bearing here).
    _a_var_vals = np.var(X[acands].to_numpy(dtype=np.float64), axis=0)
    a_var = dict(zip(acands, _a_var_vals.tolist()))
    asel = sorted(acands, key=lambda a: a_var.get(a, 0.0), reverse=True)[: max(1, int(max_agg_cols))]
    if not gsel or not asel:
        return X, [], []

    # PRE-CAP (perf): the OOF fit below previously computed all gsel x asel pairs and only
    # then capped to top-``max_pairs`` by ``pair_rank`` (group MI, then agg variance) - both already
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
    # max_pairs cap on the (group, agg) pairs, by descending group MI then agg variance.
    pair_rank = {(g, a): (g_mi.get(g, 0.0), a_var.get(a, 0.0)) for g in gsel for a in asel}

    def _cap_names(names, recipes_map):
        """Group ``names`` by their (group_col, agg_col) source pair and keep only the top ``max_pairs`` pairs by ``pair_rank``, preserving per-pair column order."""
        nbp: dict = {}
        for _nm in names:
            nbp.setdefault((recipes_map[_nm]["group_col"], recipes_map[_nm]["agg_col"]), []).append(_nm)
        _tops = sorted(nbp, key=lambda p: pair_rank.get(p, (0.0, 0.0)), reverse=True)[: max(1, int(max_pairs))]
        return [_nm for p in _tops for _nm in nbp[p]]

    # DEVICE-BORN binagg fit: when the device MI gate is available, fit RECIPES-ONLY (skip the 5-fold
    # OOF host loop - the FE scan's largest GPU-idle host stage, ~5s at 1M rows), gate on the device from those
    # recipes, then build the host OOF for the FEW survivors only. The device gate scores an OOF rebuilt ON-device
    # from the recipes (it never reads host OOF values), so the survivor set - and thus the survivor OOF the
    # redundancy gate + the append consume - is BYTE-IDENTICAL to fitting every candidate on the host; only the
    # OOF of the DROPPED candidates is never computed. Any device-gate None / failure -> the exact host path below.
    _device_binagg = False
    if mi_gate:
        try:
            from ._gpu_strict_fe import fe_gpu_device_born_binagg_enabled
            _device_binagg = bool(fe_gpu_device_born_binagg_enabled())
        except Exception as e:
            logger.debug("fe_gpu_device_born_binagg_enabled() check failed, defaulting to False: %s", e)
            _device_binagg = False

    feat_df = None
    raw = None
    _dev_cands: "Optional[DeviceOofCandidates]" = None
    state = "unavailable"
    if _device_binagg:
        state, raw, feat_df, _dev_cands = device_born_candidates(
            fit_binned_numeric_agg, X, y, gsel, asel, stats, nbins_base, n_folds, random_state, _precap_pairs, _cap_names, redundancy_gate, reject_sink,
        )
        if state == "empty":
            return X, [], []

    if state == "unavailable":
        # HOST path (unchanged behaviour): build ALL capped OOF columns, then the host CPU MI gate.
        feat_df, raw = fit_binned_numeric_agg(
            X, y, group_num_cols=gsel, agg_num_cols=asel,
            stats=stats, nbins_base=nbins_base, n_folds=n_folds, random_state=random_state,
            pairs=_precap_pairs,
        )
        if feat_df.shape[1] == 0:
            return X, [], []
        feat_df = feat_df[_cap_names(list(feat_df.columns), raw)]
        # PROBE-GATE: keep only columns clearing the local MI floor vs the raw baseline (cheap exit if no signal).
        if mi_gate and feat_df.shape[1] > 0:
            from ._unified_fe_gate import local_mi_gate
            survivors_set = set(local_mi_gate(feat_df, y, raw_X=X, reject_sink=reject_sink))
            feat_df = feat_df[[c for c in feat_df.columns if c in survivors_set]]
            if feat_df.shape[1] == 0:
                return X, [], []

    # REDUNDANCY GATE: a ``binagg_*`` column ``stat(a | qbin(g))`` is a deterministic function of its source
    # columns ``(g, a)``. The Tier-1 MI floor above keeps it whenever MI(col; y) clears the raw noise floor,
    # but that fires even when the column carries NO information about y beyond ``g``/``a`` themselves - e.g. on a
    # linearly-separable target where the raw source already explains y, the binned aggregate is a redundant
    # re-encoding of raw signal. Keep a column only when ``CMI(col; y | g, a) >= min_cmi_gain``, conditioning on
    # its OWN sources (cheap: at most two raw columns). Collapses spurious appends to zero on data with no
    # residual per-cell structure while preserving genuine cell-conditional features (where the per-cell shape
    # adds information the raw marginals cannot).
    assert raw is not None and (feat_df is not None or _dev_cands is not None)  # the host branch fills feat_df, the device branch fills it or leaves the lazy device source
    if redundancy_gate and (_dev_cands is not None or (feat_df is not None and feat_df.shape[1] > 0)):
        from ._mi_greedy_cmi_fe import _quantile_bin
        from ._unified_fe_gate import _coerce_y_classes

        y_cls = _coerce_y_classes(y)
        _src_bin_cache: dict[str, np.ndarray] = {}
        # Permutation-null floor. The plug-in CMI estimator is positively biased at finite n: a binagg column that is
        # genuinely redundant with its sources still scores a small POSITIVE CMI that a fixed threshold cannot separate
        # from real signal (the bias grows as n shrinks / the conditioning support fragments, and the OOF per-fold
        # aggregate adds further sampling noise). Calibrate per-candidate: score the SAME candidate against shuffled-y
        # under the same conditioning; the max over a handful of permutations is the candidate's own noise ceiling.
        # Keep it only when its observed CMI clears BOTH the absolute floor AND that ceiling - genuine cell-conditional
        # signal sits far above the null, redundant re-encodings sit at it.
        #
        # FWER fix: the ceiling was the raw MAX over only 15 permutations. With ~1/(n_perm+1) effective
        # alpha per candidate and many (group, agg, stat) candidates over many fits, a high-variance noise stat
        # (kurt sits at the top of the moment ladder) eventually clears the noisy max by luck - measured on the
        # all-noise null frame (seed=6, clf): binagg_kurt(n2|qbin(n4)) cmi=0.02507 vs max-of-15=0.02279 (PASS by
        # luck) yet max-of-60=0.02490 and the candidate sits squarely INSIDE the null. Replace the unstable raw
        # max with a robust one-sided z-ceiling (mean + _NULL_Z * std of the permutation CMIs): a stable estimate
        # of the null upper tail that does not depend on catching the single luckiest permutation. Genuine signal
        # sits far above the null so this does not suppress it; it collapses the borderline noise FPs to zero.
        _n_perm = 30
        # Family-wise z: the ceiling is applied to EVERY surviving (group, agg, stat) candidate (up to max_pairs x
        # stats = 256 at the defaults), so a fixed per-candidate z=2 (~2.3% one-sided) lets ~6 pure-noise columns
        # through on a wide noise frame (measured: binagg_std(noise_48|qbin(x5)) survived on the wide-synergy
        # fixture and fed spurious engineered composites). Bonferroni over the tested candidates at alpha=0.05,
        # never below the old per-candidate z.
        cands = _dev_cands if _dev_cands is not None else HostCandidates(feat_df, _quantile_bin, _candidate_codes)
        _NULL_Z = _family_wise_null_z(len(cands.names))
        _rng = np.random.default_rng(int(random_state))

        def _src_bins(col: str) -> np.ndarray:
            """Quantile-bin codes for a raw source column, memoized in ``_src_bin_cache`` since the same source column conditions many candidate binagg columns."""
            b = _src_bin_cache.get(col)
            if b is None:
                b = _quantile_bin(X[col].to_numpy(dtype=np.float64), nbins=nbins_base)
                _src_bin_cache[col] = b
            return b

        kept_cols: list[Any] = []
        _build_binned_agg_recipes(cands, raw, X, _src_bins, nbins_base, y_cls, min_cmi_gain, _rng, _n_perm, _NULL_Z, kept_cols, reject_sink)
        if _dev_cands is not None:
            # Host out-of-fold columns for the KEPT pairs only (usually none): same values the all-survivors fit would have produced for them.
            if not kept_cols:
                return X, [], []
            _kept_pairs = set((raw[_nm]["group_col"], raw[_nm]["agg_col"]) for _nm in kept_cols)
            feat_df, _ = fit_binned_numeric_agg(
                X, y, group_num_cols=gsel, agg_num_cols=asel,
                stats=stats, nbins_base=nbins_base, n_folds=n_folds, random_state=random_state,
                pairs=_kept_pairs,
            )
            _have = set(feat_df.columns)
            kept_cols = [c for c in kept_cols if c in _have]
        assert feat_df is not None
        feat_df = feat_df[kept_cols]
        if feat_df.shape[1] == 0:
            return X, [], []

    X_aug = pd.concat([X, feat_df], axis=1)
    recipes = [build_binned_numeric_agg_recipe(n, raw[n]) for n in feat_df.columns]
    return X_aug, list(feat_df.columns), recipes


from ._binned_numeric_agg_redundancy import _candidate_codes, _build_binned_agg_recipes
