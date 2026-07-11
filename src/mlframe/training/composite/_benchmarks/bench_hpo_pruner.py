"""Wall-time A/B: ``optimize_composite`` with vs without the default MedianPruner.

Question: does the default-on pruner actually reduce wasted CV compute (the whole point of pruning), and does
it change the winning selection score? Run: ``python -m mlframe.training.composite._benchmarks.bench_hpo_pruner``
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from mlframe.training.composite.hpo import HPOSpace, optimize_composite


def _make_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0, 1, n)) + 100.0
    feat = rng.uniform(-2, 2, n)
    resid = np.where(feat > 0, 3.0, -3.0) + 0.1 * rng.normal(0, 1, n)
    y = base + resid
    X = pd.DataFrame({"base": base, "feat": feat})
    return X, y


if __name__ == "__main__":
    X, y = _make_data(3000, seed=0)
    spaces = {"max_depth": HPOSpace("int", low=1, high=10)}

    for pruner_label, pruner_value in [("no_pruner", None), ("auto_median_pruner", "auto")]:
        t0 = time.perf_counter()
        res = optimize_composite(
            X, y, base_column="base",
            transform_candidates=("diff", "linear_residual", "ratio"),
            inner_factory=lambda: DecisionTreeRegressor(random_state=0),
            inner_spaces=spaces,
            n_trials=40, cv=6, prefer_optuna=True, random_state=0,
            pruner=pruner_value,
        )
        wall = time.perf_counter() - t0
        print(f"{pruner_label:>20} -> {wall * 1000:9.2f} ms, completed_trials={len(res.trials):>3}/40, selection_score={res.selection_score:.4f}")
