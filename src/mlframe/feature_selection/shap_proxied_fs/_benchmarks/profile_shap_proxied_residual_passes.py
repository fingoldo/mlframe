"""cProfile + wall-clock harness for gt_09 two-phase residual attribution (``residual_passes``).

Run: python -m mlframe.feature_selection.shap_proxied_fs._benchmarks.profile_shap_proxied_residual_passes

Measures 0-pass vs 1-pass wall at widths {2000, 10000} (gt_09 sec 6 acceptance criterion) and
profiles the residual pass's own cost breakdown via cProfile at p=2000. CPU-only.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cProfile
import io
import pstats
import time

import numpy as np


def make_data(n=2000, p=2000, seed=0, n_strong=6, n_weak=6, strong_weight=1.0, weak_weight=0.25):
    """Mixed-strength synthetic (matches the gt_09 biz_val fixture, sized down for the profile)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float64)
    strong = list(range(n_strong))
    weak = list(range(min(50, p - n_weak), min(50, p - n_weak) + n_weak))
    logit = strong_weight * X[:, strong].sum(axis=1) + weak_weight * X[:, weak].sum(axis=1)
    logit = logit / logit.std() * 2.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int64)
    return X, y


def _fit_wall(X, y, residual_passes, seed=0):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    sel = ShapProxiedFS(
        classification=True, use_gpu=False, cluster_use_gpu=False, revalidate=True,
        residual_passes=residual_passes, residual_merge="rescue", random_state=seed,
    )
    t0 = time.perf_counter()
    sel.fit(X, y)
    wall = time.perf_counter() - t0
    return wall, sel


def bench_width(p, n=2000, seed=0):
    X, y = make_data(n=n, p=p, seed=seed)
    _fit_wall(X, y, 0, seed=seed)  # warm numba / xgboost
    base_wall, _ = _fit_wall(X, y, 0, seed=seed)
    res_wall, sel_res = _fit_wall(X, y, 1, seed=seed)
    ratio = res_wall / base_wall if base_wall > 0 else float("nan")
    print(f"p={p}: 0-pass={base_wall:.3f}s 1-pass={res_wall:.3f}s ratio={ratio:.2f}x "
          f"(gt_09 sec 5 acceptance: <=1.8x)")
    return ratio


def main():
    for p in (2000, 10000):
        bench_width(p)

    pr = cProfile.Profile()
    X, y = make_data(n=2000, p=2000)
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    sel = ShapProxiedFS(classification=True, use_gpu=False, cluster_use_gpu=False, revalidate=True, residual_passes=1)
    sel.fit(X, y)  # warm
    pr.enable()
    sel2 = ShapProxiedFS(classification=True, use_gpu=False, cluster_use_gpu=False, revalidate=True, residual_passes=1)
    sel2.fit(X, y)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())


if __name__ == "__main__":
    main()
