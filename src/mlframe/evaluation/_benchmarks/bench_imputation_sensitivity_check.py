"""cProfile harness for ``evaluation.imputation_sensitivity_check.imputation_sensitivity_check``.

Run: ``python -m mlframe.evaluation._benchmarks.bench_imputation_sensitivity_check``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from mlframe.evaluation.imputation_sensitivity_check import imputation_sensitivity_check


def _make_data(n: int, n_features: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = (X.iloc[:, :3].sum(axis=1) + rng.normal(scale=0.5, size=n)).to_numpy()
    return X, y


def _run(n: int, n_features: int, n_variants: int, with_shift: bool = False) -> None:
    X, y = _make_data(n, n_features)
    variants = {f"variant_{i}": X for i in range(n_variants)}
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    shift_split = (np.arange(0, n // 2), np.arange(n // 2, n)) if with_shift else None
    imputation_sensitivity_check(Ridge(alpha=0.1), variants, y, r2_score, cv=cv, shift_split=shift_split)


if __name__ == "__main__":
    for n, n_features, n_variants in [(500, 15, 3), (2000, 15, 3), (2000, 15, 8)]:
        t0 = time.perf_counter()
        _run(n, n_features, n_variants)
        wall = time.perf_counter() - t0
        print(f"n={n:>5} n_features={n_features:>3} n_variants={n_variants:>2} -> {wall * 1000:9.2f} ms")

    # shift-aware path: same fold-CV cost plus one extra fit/predict per variant against the shift holdout.
    for n, n_features, n_variants in [(500, 15, 3), (2000, 15, 3), (2000, 15, 8)]:
        t0 = time.perf_counter()
        _run(n, n_features, n_variants, with_shift=True)
        wall = time.perf_counter() - t0
        print(f"[shift_split] n={n:>5} n_features={n_features:>3} n_variants={n_variants:>2} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(2000, 15, 8)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run(2000, 15, 8, with_shift=True)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("[shift_split]")
    print(buf.getvalue())
