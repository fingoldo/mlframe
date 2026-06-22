"""cProfile harness for the ShapProxiedFS fit + transform hot path.

Run: D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection.shap_proxied_fs._benchmarks.profile_shap_proxied_fit

Profiles a representative C3-tier fit (n=2000, p=200 with a handful of informative + correlated cluster
columns) so the OOF-SHAP, cluster, prefilter, search and revalidate stages all engage. CPU-only.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cProfile
import pstats
import io

import numpy as np


def make_data(n=2000, p=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float64)
    # informative
    w = np.zeros(p)
    w[:6] = [2.0, -1.5, 1.2, -0.9, 0.7, -0.5]
    # correlated cluster: cols 10..13 echo col 0
    for c in (10, 11, 12, 13):
        X[:, c] = X[:, 0] + 0.05 * rng.standard_normal(n)
    logits = X @ w
    p1 = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < p1).astype(np.int64)
    return X, y


def run():
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = make_data()
    sel = ShapProxiedFS(classification=True, use_gpu=False, cluster_use_gpu=False, revalidate=True)
    sel.fit(X, y)
    sel.transform(X)
    return sel


def main():
    run()  # warm numba / shap import
    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(35)
    ps.sort_stats("tottime").print_stats(25)
    print(s.getvalue())


if __name__ == "__main__":
    main()
