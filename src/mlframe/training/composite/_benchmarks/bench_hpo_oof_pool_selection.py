"""cProfile: ``optimize_composite(collect_oof_pool=True)`` + ``select_oof_pool_ensemble`` over the pool.

Question: where does the wall time go once a caller both harvests the trial OOF pool AND runs the new
``select_ensemble_from_pool`` / ``select_oof_pool_ensemble`` convenience wiring over it -- is the added
stepwise-selection pass negligible next to the HPO search itself (which it should be: selection runs
directly on already-computed OOF arrays, no refit)? Run:
``python -m mlframe.training.composite._benchmarks.bench_hpo_oof_pool_selection``
"""
from __future__ import annotations

import cProfile
import pstats
import time

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from mlframe.training.composite.hpo import HPOSpace, optimize_composite, select_oof_pool_ensemble


def _make_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    y = base + np.sin(2.0 * f1) + 0.5 * f2 + 0.2 * rng.standard_normal(n)
    X = pd.DataFrame({"base": base, "f1": f1, "f2": f2})
    return X, y


def _run(X, y, spaces):
    result = optimize_composite(
        X, y, base_column="base",
        transform_candidates=("diff", "ratio", "linear_residual"),
        inner_factory=lambda: DecisionTreeRegressor(random_state=0),
        inner_spaces=spaces,
        n_trials=25, cv=4, prefer_optuna=False, collect_oof_pool=True, random_state=11,
    )
    selection = select_oof_pool_ensemble(result, y)
    return result, selection


if __name__ == "__main__":
    X, y = _make_data(600, seed=42)
    spaces = {"max_depth": HPOSpace("int", low=1, high=15)}

    # Warm (JIT / import costs) before timing.
    _run(X, y, spaces)

    t0 = time.perf_counter()
    result, selection = _run(X, y, spaces)
    wall = time.perf_counter() - t0
    print(f"end-to-end wall (search + collect_oof_pool + select_oof_pool_ensemble): {wall * 1000:.2f} ms")
    print(f"n_trials={len(result.trials)}, kept={len(selection.kept_trial_indices)}, selection_score={selection.score:.4f}")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(X, y, spaces)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(30)

    # Isolate the NEW path's own cost against the HPO search it rides on top of.
    result_only = optimize_composite(
        X, y, base_column="base",
        transform_candidates=("diff", "ratio", "linear_residual"),
        inner_factory=lambda: DecisionTreeRegressor(random_state=0),
        inner_spaces=spaces,
        n_trials=25, cv=4, prefer_optuna=False, collect_oof_pool=True, random_state=11,
    )
    t1 = time.perf_counter()
    select_oof_pool_ensemble(result_only, y)
    selection_only_wall = time.perf_counter() - t1
    print(f"select_oof_pool_ensemble-only wall (pool already collected): {selection_only_wall * 1000:.3f} ms")
