"""cProfile + wall-clock harness for gt_01's ``proxy_mode="faith_interaction"`` (order-2 Faith-Shap surrogate).

Run: python -m mlframe.feature_selection.shap_proxied_fs._benchmarks.profile_faith_interactions

Measures ``faith_shap_order2``/``faith_interaction_top_n`` wall vs ``n_coalitions`` at proxy widths
{28, 112} (plan section 4 step 4's grid); the regression solve is tiny -- the loop cost is
``n_coalitions * O(n_samples)`` subset_loss reductions via the memoised ``_Evaluator``. Target: <=2s
at n_samples=3000, width 112, 2048 coalitions.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np


def _make_fixture(n_samples: int, n_features: int, seed: int = 0):
    """Synthetic (phi, base, y) proxy-game fixture, ~10% features informative with pairwise-interacting operands among them."""
    rng = np.random.default_rng(seed)
    weights = np.zeros(n_features)
    n_informative = max(4, n_features // 10)
    weights[:n_informative] = rng.uniform(0.5, 2.0, size=n_informative)
    phi = rng.normal(0, 0.3, size=(n_samples, n_features)) * weights[None, :]
    base = rng.normal(0, 0.1, size=n_samples)
    y = (base + phi.sum(axis=1) + rng.normal(0, 0.5, size=n_samples) > 0).astype(np.float64)
    candidate_pairs = [(0, 1), (2, 3)]
    return phi, base, y, candidate_pairs


def bench_wall_vs_n_coalitions():
    """Wall-clock faith_interaction_top_n across the plan's (n_coalitions, width) grid."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_faith_interactions import faith_interaction_top_n

    print("n_samples, n_features, n_coalitions, wall_s")
    for n_features in (28, 112):
        phi, base, y, candidate_pairs = _make_fixture(3000, n_features)
        for n_coalitions in (512, 2048, 8192):
            rng = np.random.default_rng(0)
            faith_interaction_top_n(
                phi, base, y, classification=True, metric=None, candidate_pairs=candidate_pairs,
                min_card=2, max_card=8, top_n=10, n_coalitions=n_coalitions, rng=rng,
            )  # warm
            rng = np.random.default_rng(0)
            t0 = time.perf_counter()
            faith_interaction_top_n(
                phi, base, y, classification=True, metric=None, candidate_pairs=candidate_pairs,
                min_card=2, max_card=8, top_n=10, n_coalitions=n_coalitions, rng=rng,
            )
            wall = time.perf_counter() - t0
            print(f"3000, {n_features}, {n_coalitions}, {wall:.4f}")


def bench_cprofile():
    """cProfile the largest grid point (n_samples=3000, width=112, n_coalitions=2048), print top-25 by cumtime."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_faith_interactions import faith_interaction_top_n

    phi, base, y, candidate_pairs = _make_fixture(3000, 112)
    rng = np.random.default_rng(0)
    faith_interaction_top_n(phi, base, y, classification=True, metric=None, candidate_pairs=candidate_pairs, min_card=2, max_card=8, top_n=10, n_coalitions=2048, rng=rng)  # warm
    pr = cProfile.Profile()
    pr.enable()
    faith_interaction_top_n(
        phi, base, y, classification=True, metric=None, candidate_pairs=candidate_pairs,
        min_card=2, max_card=8, top_n=10, n_coalitions=2048, rng=np.random.default_rng(1),
    )
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    ps.print_stats(25)
    print(s.getvalue())


if __name__ == "__main__":
    bench_wall_vs_n_coalitions()
    bench_cprofile()
