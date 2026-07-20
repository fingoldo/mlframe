"""cProfile + wall-clock harness for gt_05's ``shapley_model_values`` / ``shapley_blend``.

Run: python -m mlframe.votenrank._benchmarks.profile_shapley_blend

Measures wall at (n_models, n_rows) in {(7, 4000), (20, 50000)} (plan section 5's grid), with and
without ``score_subsample``, to show the subsample knob actually bounds the AUC-sort cost at large
n_rows.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np


def _make_pool(n_models: int, n_rows: int, seed: int = 0):
    """Synthetic binary-target model pool: a mix of correlated-with-margin and pure-noise predictors."""
    rng = np.random.default_rng(seed)
    y_margin = rng.standard_normal(n_rows)
    y = (y_margin > 0).astype(np.float64)
    n_informative = max(1, n_models // 3)
    preds = []
    for i in range(n_models):
        if i < n_informative:
            preds.append(0.6 * y_margin + 0.4 * rng.standard_normal(n_rows))
        else:
            preds.append(rng.standard_normal(n_rows))
    return np.stack(preds), y


def bench_wall_grid():
    """Wall-clock shapley_model_values across the plan's (n_models, n_rows) grid, with/without subsampling."""
    from mlframe.votenrank.shapley_blend import shapley_model_values

    print("n_models, n_rows, score_subsample, wall_s")
    for n_models, n_rows in ((7, 4000), (20, 50000)):
        preds, y = _make_pool(n_models, n_rows)
        for subsample in (None, 5000):
            rng = np.random.default_rng(1)
            shapley_model_values(preds, y, n_permutations=20, score_subsample=subsample, rng=rng)  # warm
            rng = np.random.default_rng(1)
            t0 = time.perf_counter()
            shapley_model_values(preds, y, n_permutations=50, score_subsample=subsample, rng=rng)
            wall = time.perf_counter() - t0
            print(f"{n_models}, {n_rows}, {subsample}, {wall:.4f}")


def bench_cprofile():
    """cProfile the largest grid point (unsampled), print top-25 by cumtime."""
    from mlframe.votenrank.shapley_blend import shapley_model_values

    preds, y = _make_pool(20, 50000)
    rng = np.random.default_rng(0)
    shapley_model_values(preds, y, n_permutations=5, rng=rng)  # warm
    pr = cProfile.Profile()
    pr.enable()
    shapley_model_values(preds, y, n_permutations=30, rng=np.random.default_rng(0))
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    ps.print_stats(25)
    print(s.getvalue())


if __name__ == "__main__":
    bench_wall_grid()
    bench_cprofile()
