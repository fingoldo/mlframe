"""CPX29 bench: calibrate_conformal_mondrian per-group residual slicing.

OLD: per-alpha loop over G unique groups, each building a fresh ``g == u`` boolean
mask over all n residuals -> O(G*n) per alpha.
NEW: factorize + single stable argsort groups residuals into contiguous blocks once;
each group is a cheap O(n_g) slice. O(n log n) once + O(n) total slicing per alpha.

Run: CUDA_VISIBLE_DEVICES="" python bench_cpx29_conformal_mondrian.py
Warm best-of-N wall A/B; identity gate on the full per-alpha {group: radius} dicts.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from mlframe.training.composite import conformal as conf  # noqa: E402


class _Stub:
    """Minimal CompositeTargetEstimator-like object: calibrate_conformal_mondrian
    reads only ``estimator_`` (presence check) and ``predict``."""

    def __init__(self, y_pred):
        self.estimator_ = object()
        self._y_pred = y_pred

    def predict(self, X):
        return self._y_pred


def _make_data(n, n_groups, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.normal(size=n)
    y_pred = y_true + rng.normal(scale=1.0 + rng.random(n), size=n)
    groups = rng.integers(0, n_groups, size=n)
    return y_true, y_pred, groups


def _run(self, X, y, groups, alphas):
    return conf.calibrate_conformal_mondrian(self, X, y, groups, alpha=alphas)


def _best_of(fn, n_iter=7):
    best = float("inf")
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    alphas = [0.05, 0.1, 0.2]
    for n, ng in [(100_000, 500), (300_000, 2000), (500_000, 5000)]:
        y_true, y_pred, groups = _make_data(n, ng)
        stub = _Stub(y_pred)
        # warm
        _run(stub, None, y_true, groups, alphas)
        t = _best_of(lambda: _run(stub, None, y_true, groups, alphas))
        out = stub._mondrian_q_
        ngroups = len(out[round(0.1, 6)]) - 1  # minus the None key
        print(f"n={n:>7} groups={ng:>5}  ->  {t*1000:8.2f} ms  (effective_groups={ngroups})")


if __name__ == "__main__":
    main()
