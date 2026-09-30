"""Identity gate for the CatBoost cached-borders fast path: 10+ random RFECV scenarios (cat features, NaNs, sample weights, regressor/classifier,
early stopping on/off), each fit with the fast path off and on; selected subsets, per-N CV curves and final-model predictions must be bit-identical.

Run: python -m mlframe.feature_selection.wrappers.rfecv._benchmarks.bench_cb_border_cache_identity [n_scenarios]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd


def scenario(seed: int):
    rng = np.random.default_rng(seed)
    n, p = int(rng.integers(1500, 4000)), int(rng.integers(8, 25))
    X = pd.DataFrame(rng.normal(size=(n, p)).astype(np.float32 if seed % 2 else np.float64), columns=[f"f{i}" for i in range(p)])
    if seed % 3 == 0:
        X.iloc[:: int(rng.integers(5, 13)), int(rng.integers(0, p))] = np.nan
    cats = []
    if seed % 4 in (1, 2):
        X["c0"] = pd.Categorical(rng.choice(list("abcde"), n))
        cats = ["c0"]
    lin = X.f0 + X.f1 * X.f2 + 0.3 * X.f3.fillna(0)
    regress = seed % 5 == 4
    y = pd.Series(lin + rng.normal(size=n) if regress else (lin + rng.normal(size=n) > 0).astype(int))
    w = rng.uniform(0.3, 3.0, n) if seed % 2 == 0 else None
    return X, y, cats, w, regress


def run_one(seed: int, on: bool):
    from catboost import CatBoostClassifier, CatBoostRegressor

    from mlframe.feature_selection.wrappers import RFECV

    X, y, cats, w, regress = scenario(seed)
    os.environ["MLFRAME_RFECV_CB_CACHED_BORDERS"] = "1" if on else "0"
    cls = CatBoostRegressor if regress else CatBoostClassifier
    est = cls(iterations=20, depth=4, verbose=0, allow_writing_files=False, random_seed=seed)
    r = RFECV(estimator=est, cat_features=cats or None, cv=3 + seed % 3, max_refits=5, verbose=0, leakage_corr_threshold=None, random_state=seed)
    r.fit(X, y, sample_weight=w) if w is not None else r.fit(X, y)
    return list(r.support_), np.asarray(r.cv_results_["cv_mean_perf"]), np.asarray(r.cv_results_["nfeatures"])


def main(n_scenarios: int = 12) -> None:
    from mlframe.feature_selection.wrappers.rfecv import _cb_border_cache as m

    bad = 0
    for seed in range(n_scenarios):
        a = run_one(seed, False)
        m.TOTALS.update(hits=0, misses=0, fallbacks=0)
        b = run_one(seed, True)
        ok = a[0] == b[0] and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]) and m.TOTALS["hits"] > 0 and m.TOTALS["fallbacks"] == 0
        bad += not ok
        print(f"scenario {seed}: identical={ok} n_selected={sum(a[0])} totals={m.TOTALS}")
    print("ALL IDENTICAL" if not bad else f"{bad} MISMATCHES")


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
