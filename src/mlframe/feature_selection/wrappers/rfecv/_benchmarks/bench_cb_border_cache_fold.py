"""Paired per-fold-fit A/B for the CatBoost cached-borders fast path: generic ``fit(X, y)`` vs Pool + cached borders, on shrinking column subsets.

Interleaved trials, median of ``reps``, measured in process CPU time (wall is meaningless at load average >> cores on the shared box); the first (cache-miss) call is timed separately because it pays the border save. Also reports the border
table memory. Complements ``bench_cb_border_cache_e2e`` (whole ``RFECV.fit`` wall), which is too noisy on a shared box to resolve a few percent.

Measured (process CPU time, 4-core shared host at load ~19, CatBoost CPU, depth 6, 50 its + early stopping, subset-fit medians of 3-5 interleaved trials):
    n=20k  p=30: -1.6% (noise, no gain)      n=100k p=30: 7.6%      n=100k p=88: 11.3%      n=375k p=88: 13.7%
Full-width (cache-miss) fit costs +0..8% (border save) once per fold; the border tables are ~0.1-0.3 MB per fold. Whole-run saving (full-width iteration has none)
is ~0.72 x the subset saving: ~5-6% (100k x 30), ~8% (100k x 88), ~10% (375k x 88). Wired into RFECV as ``_cb_border_cache`` (default on for CatBoost CPU).

Run: python -m mlframe.feature_selection.wrappers.rfecv._benchmarks.bench_cb_border_cache_fold n p [iters] [reps]
"""
from __future__ import annotations

import statistics
import sys
import time

import numpy as np
import pandas as pd


def main(n=100_000, p=30, iters=50, reps=5):
    from catboost import CatBoostClassifier

    from mlframe.feature_selection.wrappers.rfecv import _cb_border_cache as cbc

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, p)).astype(np.float32), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(((X.f0 + X.f1 * X.f2 + rng.normal(size=n)) > 0).astype(int))
    nv = n // 10
    Xt, yt, Xv, yv = X.iloc[nv:], y.iloc[nv:], X.iloc[:nv], y.iloc[:nv]
    rows = np.arange(len(Xt))
    cols_all = list(X.columns)
    subsets = [cols_all[: max(2, int(p * f))] for f in (1.0, 0.7, 0.5, 0.35, 0.25, 0.15)]

    def mk():
        return CatBoostClassifier(iterations=iters, depth=6, verbose=0, allow_writing_files=False, random_seed=0)

    def generic(cols):
        t0 = time.process_time()
        mk().fit(Xt[cols], yt, eval_set=(Xv[cols], yv), use_best_model=True, early_stopping_rounds=20)
        return time.process_time() - t0

    src = type("Src", (), {})()

    def cached(cols):
        t0 = time.process_time()
        ok = cbc.fit_catboost_with_cached_borders(
            mk(), source=src, X_train=Xt[cols], y_train=yt, fit_features=cols, train_rows=rows,
            fit_params={"eval_set": (Xv[cols], yv), "use_best_model": True, "early_stopping_rounds": 20},
        )
        assert ok
        return time.process_time() - t0

    generic(subsets[-1])
    print(f"n={n} p={p} its={iters}")
    miss = cached(subsets[0])
    print(f"  full-width miss (border compute+save): {miss:.2f}s vs generic {statistics.median(generic(subsets[0]) for _ in range(reps)):.2f}s")
    tot_g = tot_c = 0.0
    for cols in subsets[1:]:
        g, c = [], []
        for _ in range(reps):
            g.append(generic(cols))
            c.append(cached(cols))
        mg, mc = statistics.median(g), statistics.median(c)
        tot_g += mg
        tot_c += mc
        print(f"  k={len(cols):3d}: generic {mg:.2f}s cached {mc:.2f}s saved {mg - mc:+.3f}s ({100 * (1 - mc / mg):+.1f}%)")
    print(f"  subset-fit total: generic {tot_g:.2f}s cached {tot_c:.2f}s saving {100 * (1 - tot_c / tot_g):.1f}%")
    fold = next(iter(cbc._CACHES.values())).folds
    print(f"  border table memory: {sum(sum(len(r) + 1 for v in f.lines.values() for r in v) for f in fold.values()) / 1e3:.0f} KB (text form)")


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
