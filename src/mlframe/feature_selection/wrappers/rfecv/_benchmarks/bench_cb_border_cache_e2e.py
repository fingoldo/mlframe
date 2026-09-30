"""End-to-end RFECV A/B for the CatBoost cached-quantization-borders fast path (``_cb_border_cache``).

Runs a whole ``RFECV.fit`` (CatBoost CPU) with ``cb_cached_borders`` True vs False on the same data/seed, paired and interleaved, best-of-``reps``,
and checks the selected subsets and per-N CV scores are identical. Prints wall, speedup, number of fold fits and the border-table memory.

Run: python -m mlframe.feature_selection.wrappers.rfecv._benchmarks.bench_cb_border_cache_e2e n p [iters] [cv] [max_refits] [reps] [with_cat]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd


def make_data(n: int, p: int, with_cat: bool = False, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)).astype(np.float32), columns=[f"f{i}" for i in range(p)])
    X.iloc[::11, 3] = np.nan
    cats = []
    if with_cat:
        X["cat0"] = pd.Categorical(rng.choice(list("abcde"), n))
        cats = ["cat0"]
    y = pd.Series(((X.f0 + X.f1 * X.f2 + 0.5 * X.f4.fillna(0) + rng.normal(size=n)) > 0).astype(int), name="y")
    return X, y, cats


def run(X, y, cats, *, iters: int, cv: int, max_refits: int, cached: bool, seed: int = 0):
    from catboost import CatBoostClassifier

    from mlframe.feature_selection.wrappers import RFECV

    est = CatBoostClassifier(iterations=iters, depth=6, verbose=0, allow_writing_files=False, random_seed=seed)
    r = RFECV(estimator=est, cat_features=cats or None, cv=cv, max_refits=max_refits, verbose=0, leakage_corr_threshold=None, random_state=seed)
    r.cb_cached_borders = cached
    from mlframe.feature_selection.wrappers.rfecv import _cb_border_cache as m

    m.TOTALS.update(hits=0, misses=0, fallbacks=0)
    t0 = time.perf_counter()
    r.fit(X, y)
    r._cb_totals = dict(m.TOTALS)
    return time.perf_counter() - t0, r


def main(n=20_000, p=30, iters=50, cv=3, max_refits=8, reps=2, with_cat=0):
    X, y, cats = make_data(n, p, bool(with_cat))
    run(X.iloc[:2000], y.iloc[:2000], cats, iters=5, cv=3, max_refits=2, cached=True)  # warm
    best = {True: float("inf"), False: float("inf")}
    res = {}
    for _ in range(reps):
        for cached in (False, True):
            t, r = run(X, y, cats, iters=iters, cv=cv, max_refits=max_refits, cached=cached)
            best[cached] = min(best[cached], t)
            res[cached] = r
    a, b = res[False], res[True]
    ka = a.cv_results_
    same = list(a.support_) == list(b.support_) and all(np.array_equal(np.asarray(ka[k]), np.asarray(b.cv_results_[k])) for k in ka if k.startswith("mean") or k.startswith("std"))
    nfits = len(next(iter(ka.values()))) * cv
    print(f"n={n} p={p} its={iters} cv={cv} refits={max_refits} cat={with_cat}: off={best[False]:.1f}s on={best[True]:.1f}s "
          f"speedup={best[False] / best[True]:.3f}x saved={best[False] - best[True]:.1f}s ({100 * (1 - best[True] / best[False]):.1f}%) identical={same} fast_path={b._cb_totals} off_path={a._cb_totals} ~fold_fits={nfits}")


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
