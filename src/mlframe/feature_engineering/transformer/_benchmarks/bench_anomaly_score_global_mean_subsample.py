"""Microbench: anomaly_score_features.py's
``_fit_anomaly_predict`` computed ``global_mean_train`` by rescoring the FULL ``Xt`` with both
IsolationForest models (O(n_train) per model per fold) purely to get a scalar mean. The fix subsamples
``Xt`` down to ``max_samples`` (the same sample size each tree was already fit on) for that one scalar.

Run:  python bench_anomaly_score_global_mean_subsample.py
"""
import time

import numpy as np
from sklearn.ensemble import IsolationForest


def _old_global_mean(iso1, iso2, Xt):
    return float(((-iso1.score_samples(Xt) + -iso2.score_samples(Xt)) / 2.0).mean())


def _new_global_mean(iso1, iso2, Xt, max_samples, seed):
    if Xt.shape[0] > max_samples:
        sub_idx = np.random.default_rng(int(seed) + 97).choice(Xt.shape[0], size=max_samples, replace=False)
        X_sub = Xt[sub_idx]
    else:
        X_sub = Xt
    return float(((-iso1.score_samples(X_sub) + -iso2.score_samples(X_sub)) / 2.0).mean())


def best_of(fn, n=5):
    ts = []
    r = None
    for _ in range(n):
        t = time.perf_counter()
        r = fn()
        ts.append(time.perf_counter() - t)
    return min(ts), r


def run():
    print("=== anomaly_score_features.py global_mean_train: full-Xt rescore (OLD) vs max_samples subsample (NEW) ===")
    rng = np.random.default_rng(0)
    for n_train in (5_000, 20_000, 80_000):
        Xt = rng.normal(size=(n_train, 20)).astype(np.float32)
        max_samples = min(256, n_train)
        iso1 = IsolationForest(n_estimators=100, contamination="auto", random_state=0, n_jobs=-1, max_samples=max_samples).fit(Xt)
        iso2 = IsolationForest(n_estimators=100, contamination="auto", random_state=41, n_jobs=-1, max_samples=max_samples).fit(Xt)

        t_old, r_old = best_of(lambda: _old_global_mean(iso1, iso2, Xt), n=3)
        t_new, r_new = best_of(lambda: _new_global_mean(iso1, iso2, Xt, max_samples, seed=0), n=3)
        print(
            f"  n_train={n_train:>7}: OLD {t_old * 1e3:8.2f}ms (mean={r_old:+.4f})  "
            f"NEW {t_new * 1e3:8.2f}ms (mean={r_new:+.4f})  speedup {t_old / t_new:6.2f}x"
        )


if __name__ == "__main__":
    run()
