"""cProfile + wall-clock harness for gt_03's MSR-Banzhaf prescreen ranking (``prescreen_ranking="banzhaf"``).

Run: python -m mlframe.feature_selection.shap_proxied_fs._benchmarks.profile_banzhaf_ranking

Two measurements per the gt_03 plan sec 4 step 4:
  1. Isolated ``banzhaf_msr`` batched-matmul wall at (n_samples, n_features, n_coalitions) in
     {(3000, 112, 4096), (10000, 112, 4096)}.
  2. End-to-end ``ShapProxiedFS`` prescreen STAGE wall, ``prescreen_ranking="banzhaf"`` vs
     ``"mean_abs_phi"``, on a fixture sized to force the prescreen block -- asserts the banzhaf
     stage adds <= 0.5s over mean_abs_phi at the first (3000, 112) point.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cProfile
import io
import pstats
import time

import numpy as np


def _make_phi_fixture(n_samples: int, n_features: int, seed: int = 0):
    """Build a synthetic (phi, base, y) proxy-game fixture of the given shape, ~10% informative columns."""
    rng = np.random.default_rng(seed)
    n_informative = max(2, n_features // 10)
    weights = np.zeros(n_features)
    weights[:n_informative] = rng.uniform(0.5, 2.0, size=n_informative)
    phi = rng.normal(0, 0.3, size=(n_samples, n_features)) * weights[None, :]
    base = rng.normal(0, 0.1, size=n_samples)
    y = (base + phi.sum(axis=1) + rng.normal(0, 0.5, size=n_samples) > 0).astype(np.float64)
    return phi, base, y


def bench_isolated_wall():
    """Wall-clock the batched ``banzhaf_msr`` estimator across the plan's two (n, P, m) grid points."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_banzhaf import banzhaf_msr

    print("n_samples, n_features, n_coalitions, wall_s")
    for n_samples, n_features in ((3000, 112), (10000, 112)):
        phi, base, y = _make_phi_fixture(n_samples, n_features)
        rng = np.random.default_rng(0)
        banzhaf_msr(phi, base, y, classification=True, metric=None, n_coalitions=4096, rng=rng)  # warm (numba JIT)
        rng = np.random.default_rng(0)
        t0 = time.perf_counter()
        banzhaf_msr(phi, base, y, classification=True, metric=None, n_coalitions=4096, rng=rng)
        wall = time.perf_counter() - t0
        print(f"{n_samples}, {n_features}, 4096, {wall:.4f}")


def bench_prescreen_stage_delta():
    """End-to-end fit-stage wall: banzhaf vs mean_abs_phi prescreen at (3000, 112) -- asserts <= 0.5s delta."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    n_samples, n_features = 3000, 112
    phi, base, y = _make_phi_fixture(n_samples, n_features)
    X = base[:, None] + phi + np.random.default_rng(1).normal(0, 0.05, size=phi.shape)
    stage_walls = {}
    for ranking in ("mean_abs_phi", "banzhaf"):
        fs = ShapProxiedFS(
            classification=True, prescreen_top=40, prescreen_ranking=ranking,
            optimizer="beam", top_n=5, n_jobs=1, tqdm=False, random_state=0,
        )
        fs._stage_timings = {}
        fs.fit(X, y.astype(int))  # warm
        fs2 = ShapProxiedFS(
            classification=True, prescreen_top=40, prescreen_ranking=ranking,
            optimizer="beam", top_n=5, n_jobs=1, tqdm=False, random_state=0,
        )
        fs2._stage_timings = {}
        fs2.fit(X, y.astype(int))
        wall = fs2._stage_timings.get("prescreen", 0.0)
        stage_walls[ranking] = wall
        print(f"prescreen_ranking={ranking}: prescreen stage wall {wall:.4f}s")
    delta = stage_walls["banzhaf"] - stage_walls["mean_abs_phi"]
    print(f"delta (banzhaf - mean_abs_phi): {delta:.4f}s")
    assert delta <= 0.5, f"banzhaf prescreen stage added {delta:.4f}s > 0.5s budget at (n=3000, P=112)"


def bench_cprofile():
    """cProfile the isolated estimator at the larger grid point, print top-25 by cumtime."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_banzhaf import banzhaf_msr

    phi, base, y = _make_phi_fixture(10000, 112)
    rng = np.random.default_rng(0)
    pr = cProfile.Profile()
    pr.enable()
    banzhaf_msr(phi, base, y, classification=True, metric=None, n_coalitions=4096, rng=rng)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    ps.print_stats(25)
    print(s.getvalue())


if __name__ == "__main__":
    bench_isolated_wall()
    bench_prescreen_stage_delta()
    bench_cprofile()
