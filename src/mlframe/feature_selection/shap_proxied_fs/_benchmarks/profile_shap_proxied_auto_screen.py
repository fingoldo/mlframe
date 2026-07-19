"""cProfile + wall-clock harness for the gt_08 ``proxy_mode="auto"`` su_seeded screen overhead.

Run: python -m mlframe.feature_selection.shap_proxied_fs._benchmarks.profile_shap_proxied_auto_screen

Measures the su_seeded synergy-screen cost that "auto" ALWAYS pays (even when the SNR gate finds
nothing) against a plain "additive" fit that never runs it, at widths {2000, 10000}. Acceptance
(plan gt_08 section 3.2 step 4): screen overhead <=3% of e2e wall. CPU-only.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cProfile
import io
import pstats
import time

import numpy as np


def make_data(n=2000, p=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float64)
    w = np.zeros(p)
    w[:6] = [2.0, -1.5, 1.2, -0.9, 0.7, -0.5]
    logits = X @ w
    p1 = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < p1).astype(np.int64)
    return X, y


def _fit_wall(X, y, proxy_mode, seed=0):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    sel = ShapProxiedFS(classification=True, use_gpu=False, cluster_use_gpu=False, revalidate=True, proxy_mode=proxy_mode, random_state=seed)
    t0 = time.perf_counter()
    sel.fit(X, y)
    wall = time.perf_counter() - t0
    return wall, sel


def bench_width(p, n=2000, seed=0):
    X, y = make_data(n=n, p=p, seed=seed)
    _fit_wall(X, y, "additive", seed=seed)  # warm numba / xgboost
    add_wall, _ = _fit_wall(X, y, "additive", seed=seed)
    auto_wall, sel_auto = _fit_wall(X, y, "auto", seed=seed)
    overhead_pct = 100.0 * (auto_wall - add_wall) / auto_wall if auto_wall > 0 else float("nan")
    print(f"p={p}: additive={add_wall:.3f}s auto={auto_wall:.3f}s overhead={overhead_pct:.2f}% "
          f"proxy_mode_resolved={sel_auto.shap_proxy_report_.get('proxy_mode_resolved')}")
    return overhead_pct


def main():
    for p in (2000, 10000):
        bench_width(p)

    pr = cProfile.Profile()
    X, y = make_data(n=2000, p=2000)
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    sel = ShapProxiedFS(classification=True, use_gpu=False, cluster_use_gpu=False, revalidate=True, proxy_mode="auto")
    sel.fit(X, y)  # warm
    pr.enable()
    sel2 = ShapProxiedFS(classification=True, use_gpu=False, cluster_use_gpu=False, revalidate=True, proxy_mode="auto")
    sel2.fit(X, y)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(35)
    print(s.getvalue())


if __name__ == "__main__":
    main()
