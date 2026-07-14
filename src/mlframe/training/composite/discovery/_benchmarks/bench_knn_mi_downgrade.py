"""Benchmark: knn-MI auto-downgrade (G8) -- full-fit wall of knn vs the budget-downgraded bin path.

The Kraskov estimator runs one per-column sweep per (base, transform) work item; on a moderate
screen it is orders of magnitude slower than bin-MI. ``knn_mi_auto_downgrade`` (default ON,
``knn_mi_budget_seconds=600``) probes one column's real cost and downgrades knn -> bin for the fit
when the extrapolated sweep exceeds the budget, logging a WARNING.

Measured on this machine (n=20_000 rows, 10 features, 2 bases x 6 transforms, screening="mi"):

    mi_estimator="knn", downgrade disabled : 21.86 s / fit
    mi_estimator="knn", budget=5s (fires)  :  7.00 s / fit  -> 3.1x speedup

An earlier docstring claimed 59.9x from an unverified estimate written before the benchmark was
actually run; the real number above supersedes it. 3.1x still clears the budget-downgrade's purpose
(bounding worst-case knn sweep cost), just far more modestly than the unverified guess implied.

The downgraded fit selects with bin-MI numerics (documented semantic change under the budget; the
warning names the knob to keep knn: raise ``knn_mi_budget_seconds`` or set
``knn_mi_auto_downgrade=False``).

Cross-transform T-side neighbor caching was investigated and REJECTED as infeasible with the
sklearn estimator: Kraskov radii live in the per-pair JOINT (x_j, T) space, T differs per transform,
and ``mutual_info_regression`` injects seeded jitter internally with no tree-injection API -- a
re-implementation would not be bit-identical and could alter selection. The sound reuse seams are
already wired (per-fit ``_per_feat_y_knn_full`` hoist, per-valid-mask ``_mi_y_compare_memo``, and
the honest-holdout per-(base, mask) ``mi_y`` memo added alongside this bench).

Run: python -m mlframe.training.composite.discovery._benchmarks.bench_knn_mi_downgrade
"""
from __future__ import annotations

import statistics
from timeit import default_timer as timer

import numpy as np
import pandas as pd


def _frame(n: int = 20_000, seed: int = 0):
    """Synthetic frame with two correlated bases, an orthogonal signal feature, and pure noise columns."""
    rng = np.random.default_rng(seed)
    b0 = rng.uniform(0.0, 1000.0, n)
    x0 = rng.normal(size=n)
    y = b0 + 30.0 * x0 + rng.normal(0.0, 1.0, n)
    cols = {"b0": b0, "b1": b0 + rng.normal(0.0, 100.0, n), "x0": x0}
    for j in range(7):
        cols[f"noise{j}"] = rng.normal(size=n)
    cols["y"] = y
    return pd.DataFrame(cols)


def _fit_once(df, *, auto_downgrade: bool, budget: float, reps: int = 3) -> float:
    """Median full-fit wall time (over ``reps`` fits) for one knn-downgrade-guard setting."""
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    feats = [c for c in df.columns if c != "y"]
    times = []
    for _ in range(reps):
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, random_state=0, screening="mi", mi_estimator="knn",
            base_candidates=["b0", "b1"], honest_holdout_frac=0.2, auto_base_null_perms=0,
            multi_base_enabled=False, interaction_base_discovery_enabled=False,
            auto_chain_discovery_enabled=False,
            transforms=["diff", "additive_residual", "linear_residual", "ratio", "monotonic_residual", "asinh_residual"],
            knn_mi_auto_downgrade=auto_downgrade, knn_mi_budget_seconds=budget,
        )
        disc = CompositeTargetDiscovery(cfg)
        t0 = timer()
        disc.fit(df, "y", feats, np.arange(len(df)))
        times.append(timer() - t0)
    return statistics.median(times)


def main() -> None:
    """Run the disabled-vs-firing knn-downgrade comparison and print the timing summary."""
    df = _frame()
    t_knn = _fit_once(df, auto_downgrade=False, budget=0.0, reps=1)
    t_down = _fit_once(df, auto_downgrade=True, budget=5.0)
    print(f"knn, downgrade disabled: {t_knn:.2f} s/fit")
    print(f"knn, budget=5s (fires) : {t_down:.2f} s/fit")
    print(f"speedup: {t_knn / t_down:.1f}x")


if __name__ == "__main__":
    main()
