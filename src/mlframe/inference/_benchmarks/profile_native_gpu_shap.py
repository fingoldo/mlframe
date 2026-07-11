"""cProfile + size-sweep harness for ``inference.native_gpu_shap.native_xgboost_gpu_shap_contribs``.

Run: python -m mlframe.inference._benchmarks.profile_native_gpu_shap

Two parts:
1. ``sweep()`` -- warm + best-of-5 A/B of the native booster path vs ``shap.Explainer`` on the SAME GPU-fit
   model, across a wide n_rows/n_cols range. This is the follow-up to the original honest-negative (measured
   at a single shape, n=30000 f=20) -- looking for a crossover point the original A/B could have missed.
   Measured on this hardware (2026-07-11): ratio stayed in {0.99x, 1.01x, 1.04x, 1.01x, 1.09x} across
   n_rows in {30000, 200000, 1000000} and n_cols in {20, 100, 300} -- noise-level everywhere, no crossover.
2. ``profile()`` -- cProfile of a representative call, to check for any avoidable overhead in the wrapper
   itself (DMatrix construction, split-off of the trailing base-value column).
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np


def make_data(n: int, f: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f)).astype(np.float32)
    logit = X[:, 0] * 1.5 - X[:, 1] * 0.7 + 0.3 * X[:, 2] * X[:, 3]
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return X, y


def _best_of(fn, n: int = 5) -> float:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def sweep():
    import shap
    import xgboost as xgb

    from mlframe.inference.native_gpu_shap import native_xgboost_gpu_shap_contribs

    def run(n_rows, n_cols):
        X, y = make_data(n_rows, n_cols)
        model = xgb.XGBClassifier(n_estimators=150, max_depth=5, device="cuda", tree_method="hist")
        model.fit(X, y)
        native_xgboost_gpu_shap_contribs(model, X)  # warm
        explainer = shap.Explainer(model)
        explainer(X)  # warm

        t_native = _best_of(lambda: native_xgboost_gpu_shap_contribs(model, X))
        t_shap = _best_of(lambda: explainer(X))
        ratio = t_shap / t_native
        print(f"n={n_rows:>8} f={n_cols:>4}  native={t_native:8.4f}s  shap.Explainer={t_shap:8.4f}s  ratio={ratio:6.2f}x")

    print("--- n_rows sweep (f=20) ---")
    for n in (30_000, 200_000, 1_000_000):
        run(n, 20)

    print("--- n_cols sweep (n=200000) ---")
    for f in (20, 100, 300):
        run(200_000, f)


def profile():
    from mlframe.inference.native_gpu_shap import native_xgboost_gpu_shap_contribs

    import xgboost as xgb

    X, y = make_data(200_000, 100)
    model = xgb.XGBClassifier(n_estimators=150, max_depth=5, device="cuda", tree_method="hist")
    model.fit(X, y)
    native_xgboost_gpu_shap_contribs(model, X)  # warm

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(5):
        native_xgboost_gpu_shap_contribs(model, X)
    profiler.disable()

    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(30)
    print(buf.getvalue())


if __name__ == "__main__":
    sweep()
    profile()
