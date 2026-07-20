"""cProfile + wall-clock harness for gt_02's least-core / nucleolus stability refine (``refine_mode="core"``).

Run: python -m mlframe.feature_selection.shap_proxied_fs._benchmarks.profile_core_refine

Two measurements per the gt_02 plan sec 4 step 4:
  1. Isolated ``least_core_allocation`` LP cost across k in {10, 20, 30} players x n_coalitions in
     {256, 512, 1024} (the cost driver is n_coalitions x O(n) proxy-loss evals + one HiGHS LP).
  2. End-to-end ``ShapProxiedFS.fit`` wall, ``refine_mode="core"`` vs ``"greedy"``, on a fixture sized
     to produce k~17 winning members -- the gt_02 target: core wall <= greedy wall (greedy pays k^2
     honest FITS across its stage-2b single-drop rounds; core pays sampled proxy evals + ONE honest fit).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cProfile
import io
import pstats
import time

import numpy as np


class _SyntheticLossEvaluator:
    """Deterministic synthetic proxy-loss oracle: fixed per-unit weight + small subset-interaction noise.

    Mimics the shape of a real ``_Evaluator.loss`` call (memoised, O(1) amortised per repeat query,
    O(|subset|) on a cache miss) without requiring a real SHAP fit -- isolates the LP/sampling cost.
    """

    def __init__(self, k, seed=0):
        rng = np.random.default_rng(seed)
        self.weights = rng.uniform(0.1, 1.0, size=k)
        self.cache = {}
        self.n_evals = 0

    def loss(self, idx):
        """Return a cached or freshly-computed synthetic loss for the given unit-index subset."""
        key = tuple(sorted(idx))
        if key in self.cache:
            return self.cache[key]
        self.n_evals += 1
        val = -float(sum(self.weights[j] for j in key))  # more units -> lower ("better") synthetic loss
        self.cache[key] = val
        return val


def bench_lp_grid():
    """Sweep k x n_coalitions, report LP wall + n_evals for each cell."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_core_stability import least_core_allocation

    print("k, n_coalitions, wall_s, lp_status, binding_coalitions")
    for k in (10, 20, 30):
        for n_coalitions in (256, 512, 1024):
            ev = _SyntheticLossEvaluator(k)
            rng = np.random.default_rng(0)
            least_core_allocation(ev, tuple(range(k)), n_coalitions=n_coalitions, rng=rng)  # warm
            ev = _SyntheticLossEvaluator(k)
            rng = np.random.default_rng(0)
            t0 = time.perf_counter()
            _x, _eps, info = least_core_allocation(ev, tuple(range(k)), n_coalitions=n_coalitions, rng=rng)
            wall = time.perf_counter() - t0
            print(f"{k}, {n_coalitions}, {wall:.4f}, {info['lp_status']}, {info['binding_coalitions']}")


def _make_data(n=2000, p=2000, seed=0, n_strong=6, n_weak=11):
    """Mixed-strength synthetic fixture sized to produce ~17 winning members (6 strong + 11 weak)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float64)
    strong = list(range(n_strong))
    weak = list(range(50, 50 + n_weak))
    logit = X[:, strong].sum(axis=1) + 0.25 * X[:, weak].sum(axis=1)
    logit = logit / logit.std() * 2.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int64)
    return X, y


def _fit_wall(X, y, refine_mode, seed=0):
    """Fit ShapProxiedFS with the given refine_mode and return (wall_seconds, fitted_estimator)."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    sel = ShapProxiedFS(classification=True, prescreen_ladder_mode="off", n_jobs=1, refine_mode=refine_mode, random_state=seed)
    t0 = time.perf_counter()
    sel.fit(X, y)
    wall = time.perf_counter() - t0
    return wall, sel


def bench_end_to_end():
    """Compare refine_mode='core' vs 'greedy' wall at the k~17 target regime."""
    X, y = _make_data()
    _fit_wall(X, y, "greedy")  # warm numba / xgboost
    greedy_wall, sel_greedy = _fit_wall(X, y, "greedy")
    core_wall, sel_core = _fit_wall(X, y, "core")
    n_before = sel_greedy.shap_proxy_report_["within_cluster_refine"]["before"]
    print(f"end-to-end at k(before-refine)={n_before}: greedy={greedy_wall:.3f}s core={core_wall:.3f}s "
          f"ratio(core/greedy)={core_wall / greedy_wall:.3f} (gt_02 target: <= 1.0)")
    return greedy_wall, core_wall


def profile_core_refine_cost():
    """cProfile the isolated LP grid (k=17-ish default n_coalitions=512), top 25 by cumtime."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_core_stability import least_core_allocation

    ev = _SyntheticLossEvaluator(17)
    rng = np.random.default_rng(0)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(20):
        least_core_allocation(ev, tuple(range(17)), n_coalitions=512, rng=rng)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(25)
    print(s.getvalue())


def main():
    """Run the LP grid sweep, the cProfile breakdown, and the end-to-end core-vs-greedy wall comparison."""
    print("=== LP grid (k x n_coalitions) ===")
    bench_lp_grid()
    print("\n=== cProfile: least_core_allocation @ k=17, n_coalitions=512, x20 ===")
    profile_core_refine_cost()
    print("\n=== end-to-end fit wall: core vs greedy @ k~17 ===")
    bench_end_to_end()


if __name__ == "__main__":
    main()
