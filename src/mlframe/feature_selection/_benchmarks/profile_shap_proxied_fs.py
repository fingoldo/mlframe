"""Benchmark + cProfile harness for ShapProxiedFS.

Answers three questions:
  1. How fast is one proxy subset-evaluation vs one honest retrain? (the core ~1000x claim)
  2. Where does the wall-clock go in ``fit`` -- SHAP compute, the subset scan, trust guard, or
     honest re-validation? (the pragmatic-critic prediction: SHAP + re-validation dominate, NOT the
     scan).
  3. cProfile top cumulative hotspots, to optimise the stage that actually costs.

Run:  python -m mlframe.feature_selection._benchmarks.profile_shap_proxied_fs
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def make_data(n=20000, n_inf=8, n_noise=5, n_corr=2, seed=0):
    rng = np.random.default_rng(seed)
    inf = rng.normal(size=(n, n_inf))
    noise = rng.normal(size=(n, n_noise))
    corr = inf[:, :n_corr] + 0.3 * rng.normal(size=(n, n_corr))
    X = pd.DataFrame(
        np.column_stack([inf, noise, corr]),
        columns=[f"inf{i}" for i in range(n_inf)] + [f"noise{i}" for i in range(n_noise)]
        + [f"corr{i}" for i in range(n_corr)],
    )
    coefs = np.linspace(1.0, 0.3, n_inf)
    logit = inf @ coefs
    y = (logit + 0.4 * rng.normal(size=n) > 0).astype(int)
    return X, y


def bench_proxy_vs_honest(X, y):
    """Time one proxy subset-eval vs one honest retrain on the same subset."""
    from sklearn.model_selection import train_test_split

    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import compute_shap_matrix, make_default_estimator
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_objective import subset_loss
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import _honest_loss

    model = make_default_estimator(classification=True)
    Xs, Xh, ys, yh = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    Xs, Xh = Xs.reset_index(drop=True), Xh.reset_index(drop=True)

    t0 = time.perf_counter()
    phi, base, y_phi = compute_shap_matrix(model, Xs, ys, classification=True, out_of_fold=True, n_splits=5)
    t_shap = time.perf_counter() - t0

    idx = list(range(min(8, X.shape[1])))
    reps = 200
    t0 = time.perf_counter()
    for _ in range(reps):
        subset_loss(phi, base, y_phi, idx, "brier")
    t_proxy = (time.perf_counter() - t0) / reps

    t0 = time.perf_counter()
    _honest_loss(model, Xs, ys, Xh, yh, idx, classification=True, metric="brier")
    t_honest = time.perf_counter() - t0

    print(f"\n=== proxy vs honest (n={len(X)}, f={X.shape[1]}) ===")
    print(f"OOF SHAP compute (5-fold, once): {t_shap:7.3f} s")
    print(f"one proxy subset eval          : {t_proxy*1000:7.4f} ms")
    print(f"one honest retrain+eval        : {t_honest*1000:7.2f} ms")
    print(f"proxy speedup per subset       : {t_honest/max(t_proxy,1e-12):9.0f}x")


def bench_stage_breakdown(X, y):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    sel = ShapProxiedFS(classification=True, metric="brier", optimizer="bruteforce", max_features=10,
                        top_n=30, n_splits=5, n_revalidation_models=3, random_state=0, verbose=False)
    t0 = time.perf_counter()
    sel.fit(X, y)
    t_fit = time.perf_counter() - t0
    print(f"\n=== full fit wall-clock: {t_fit:.2f} s; selected {len(sel.selected_features_)} features ===")
    print(f"trust spearman={sel.shap_proxy_report_['trust']['spearman']:.3f}; "
          f"proxy_wins_vs_importance={sel.shap_proxy_report_['importance_ablation']['proxy_wins']}")
    return sel


def profile_fit(X, y):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    sel = ShapProxiedFS(classification=True, metric="brier", optimizer="bruteforce", max_features=10,
                        top_n=30, n_splits=5, n_revalidation_models=3, random_state=0, verbose=False)
    pr = cProfile.Profile()
    pr.enable()
    sel.fit(X, y)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
    print("\n=== cProfile (top 25 by cumulative) ===")
    print(s.getvalue())


if __name__ == "__main__":
    X, y = make_data()
    bench_proxy_vs_honest(X, y)
    bench_stage_breakdown(X, y)
    profile_fit(X, y)
