"""A/B bench for removing the wasted full-array ``(fold_ids != f) & finite`` gate in
``fit_binned_numeric_agg``'s per-fold OOF loop.

The old gate built a full ``(n,)`` boolean array and called ``.any()`` on it just to check
whether a fold's training split has any finite rows -- an O(n) scan repeated once per
(group_col, agg_col, fold), i.e. G*A*n_folds times. The new gate reuses ``test_fin`` (already
computed for the per-fold moment call, O(n/n_folds)) and a cached finite-count, replacing the
O(n) scan with an O(n/n_folds) one. Validates bit-identical output (pure refactor, same
skip decisions) and measures the real wall-clock win at 2M rows with several group/agg columns.
"""

import time

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._binned_numeric_agg_fe import (
    SUPPORTED_STATS,
    _derive_cell_stats,
    _global_stat,
    _raw_moments,
    engineered_name_binned_agg,
    fit_binned_numeric_agg,
    quantile_edges,
    resolve_nbins_and_stats,
)


def _fit_binned_numeric_agg_old(X, y, *, group_num_cols, agg_num_cols, stats=SUPPORTED_STATS, nbins_base=10, n_folds=5, random_state=0, pairs=None, recipe_only=False):
    """Pre-fix reference: full-array ``(fold_ids != f) & finite`` + ``.any()`` gate, materialised every (gcol, acol, fold)."""
    n = len(X)
    rng = np.random.default_rng(int(random_state))
    fold_ids = np.empty(n, dtype=np.int64)
    fold_ids[rng.permutation(n)] = np.arange(n) % int(n_folds)
    feat_cols = {}
    recipes = {}
    _av_cache = {}
    _globals_cache = {}
    _fold_ne = None if recipe_only else [fold_ids != f for f in range(int(n_folds))]
    _fold_test = None if recipe_only else [np.where(fold_ids == f)[0] for f in range(int(n_folds))]
    for gcol in group_num_cols:
        gvals = np.asarray(X[gcol].to_numpy(), dtype=np.float64)
        if not np.isfinite(gvals).all():
            continue
        nbins, kept_stats = resolve_nbins_and_stats(n, stats, nbins_base, k=1)
        edges = quantile_edges(gvals, nbins)
        if edges.size == 0:
            continue
        codes = np.searchsorted(edges, gvals, side="right")
        n_cells = int(codes.max()) + 1
        _ct_by_fold = None if recipe_only else [codes[_ft] for _ft in _fold_test]
        for acol in agg_num_cols:
            if acol == gcol:
                continue
            if pairs is not None and (gcol, acol) not in pairs:
                continue
            _avc = _av_cache.get(acol)
            if _avc is None:
                av = np.asarray(X[acol].to_numpy(), dtype=np.float64)
                finite = np.isfinite(av)
                _av_cache[acol] = (av, finite)
            else:
                av, finite = _avc
            _gk = (acol, tuple(kept_stats))
            globals_ = _globals_cache.get(_gk)
            if globals_ is None:
                globals_ = {s: _global_stat(av[finite], s) for s in kept_stats}
                _globals_cache[_gk] = globals_
            full_cnt, full_s1, full_s2, full_s3, full_s4 = _raw_moments(codes[finite], av[finite], n_cells)
            if not recipe_only:
                oof = {s: np.full(n, globals_[s], dtype=np.float64) for s in kept_stats}
                for f in range(int(n_folds)):
                    tr = _fold_ne[f] & finite
                    if not tr.any():
                        continue
                    test = _fold_test[f]
                    ct = _ct_by_fold[f]
                    test_fin = test[finite[test]]
                    t_cnt, t_s1, t_s2, t_s3, t_s4 = _raw_moments(codes[test_fin], av[test_fin], n_cells)
                    per = _derive_cell_stats(full_cnt - t_cnt, full_s1 - t_s1, full_s2 - t_s2, full_s3 - t_s3, full_s4 - t_s4, kept_stats)
                    for s in kept_stats:
                        vals = per[s][ct]
                        oof[s][test] = np.where(np.isfinite(vals), vals, globals_[s])
            full = _derive_cell_stats(full_cnt, full_s1, full_s2, full_s3, full_s4, kept_stats)
            for s in kept_stats:
                name = engineered_name_binned_agg(acol, gcol, s)
                if not recipe_only:
                    feat_cols[name] = oof[s]
                lut = np.where(np.isfinite(full[s]), full[s], globals_[s]).astype(np.float64)
                recipes[name] = {"group_col": gcol, "agg_col": acol, "stat": s, "edges": edges, "lookup": lut, "global": float(globals_[s])}
    feat_df = pd.DataFrame(feat_cols, index=X.index)
    return feat_df, recipes


def _make_frame(n, n_group_cols, n_agg_cols, seed):
    rng = np.random.default_rng(seed)
    cols = {}
    for i in range(n_group_cols):
        cols[f"g{i}"] = rng.uniform(0, 1, n)
    for i in range(n_agg_cols):
        cols[f"a{i}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    y = rng.standard_normal(n)
    return X, y


def main():
    n, n_group_cols, n_agg_cols = 2_000_000, 16, 16
    X, y = _make_frame(n, n_group_cols, n_agg_cols, seed=0)
    group_cols = [f"g{i}" for i in range(n_group_cols)]
    agg_cols = [f"a{i}" for i in range(n_agg_cols)]

    # warm JIT
    Xw, yw = _make_frame(2000, n_group_cols, n_agg_cols, seed=1)
    fit_binned_numeric_agg(Xw, yw, group_num_cols=group_cols, agg_num_cols=agg_cols)
    _fit_binned_numeric_agg_old(Xw, yw, group_num_cols=group_cols, agg_num_cols=agg_cols)

    t0 = time.perf_counter()
    feat_old, recipes_old = _fit_binned_numeric_agg_old(X, y, group_num_cols=group_cols, agg_num_cols=agg_cols)
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    feat_new, recipes_new = fit_binned_numeric_agg(X, y, group_num_cols=group_cols, agg_num_cols=agg_cols)
    t_new = time.perf_counter() - t0

    assert set(feat_old.columns) == set(feat_new.columns)
    worst = max(np.max(np.abs(feat_old[c].to_numpy() - feat_new[c].to_numpy())) for c in feat_old.columns)
    print(f"old: {t_old:.3f}s")
    print(f"new: {t_new:.3f}s")
    print(f"speedup: {t_old / t_new:.2f}x")
    print(f"worst abs diff: {worst:.3e}")
    print(f"bit-identical: {worst == 0.0}")


if __name__ == "__main__":
    main()
