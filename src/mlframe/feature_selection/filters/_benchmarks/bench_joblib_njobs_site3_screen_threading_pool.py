"""Measurement-only bench for joblib site 3: ``_screen_predictors.py:620``'s
``Parallel(n_jobs=n_workers, backend="threading")`` pool dispatching ``evaluate_candidates``
(``_evaluation_driver.py:186``) over per-round candidate workloads.

Calls the REAL ``screen_predictors`` entry point end-to-end (not a toy stand-in) at
n_workers=1/2/4, sweeping column count (candidate-pool size) around the realistic
wellbore-100k initial-column count (~519) at a reduced row count for benchable wall
time, plus a smaller "one round" scale matching ``tests/feature_selection/test_perf_regression.py``'s
n=1000/m=10 baseline fixture.

Run: PYTHONPATH=src CUDA_VISIBLE_DEVICES="" python src/mlframe/feature_selection/filters/_benchmarks/bench_joblib_njobs_site3_screen_threading_pool.py
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.feature_selection.filters.discretization import discretize_array
from mlframe.feature_selection.filters.screen import screen_predictors


def _build_screen_inputs(n, n_noise, n_signal, seed=42):
    rng = np.random.default_rng(seed)
    sig = rng.normal(size=(n, n_signal))
    y = (sig[:, 0] + sig[:, 1] > 0).astype(np.int32)
    noise = rng.normal(size=(n, n_noise))
    x_cont = np.column_stack([sig, noise])
    x_disc = np.column_stack([discretize_array(arr=x_cont[:, j], n_bins=10, method="quantile", dtype=np.int32) for j in range(x_cont.shape[1])])
    factors_data = np.column_stack([x_disc, y]).astype(np.int32)
    factors_nbins = np.array([10] * x_disc.shape[1] + [2], dtype=np.int64)
    names = [f"F{i}" for i in range(factors_data.shape[1])]
    target_idx = factors_data.shape[1] - 1
    return factors_data, factors_nbins, names, target_idx


def _run(n_workers, factors_data, factors_nbins, names, target_idx):
    t0 = time.perf_counter()
    screen_predictors(
        factors_data=factors_data, factors_nbins=factors_nbins, factors_names=names,
        y=(target_idx,), full_npermutations=3, baseline_npermutations=2,
        n_workers=n_workers, verbose=0,
    )
    return time.perf_counter() - t0


def main():
    for n, n_noise, n_signal, label in (
        (1000, 8, 2, "small round (m=10, matches existing perf-regression fixture)"),
        (99401, 300, 20, "wellbore-scale round (n=99401, m=320 candidates)"),
    ):
        factors_data, factors_nbins, names, target_idx = _build_screen_inputs(n, n_noise, n_signal)
        print(f"\n=== {label} ===")
        # warm-up (numba JIT)
        _run(1, factors_data, factors_nbins, names, target_idx)
        t1 = min(_run(1, factors_data, factors_nbins, names, target_idx) for _ in range(3))
        for n_workers in (2, 4):
            _run(n_workers, factors_data, factors_nbins, names, target_idx)  # warm pool spawn
            t = min(_run(n_workers, factors_data, factors_nbins, names, target_idx) for _ in range(3))
            print(f"n_workers=1: {t1*1e3:8.2f} ms   n_workers={n_workers}: {t*1e3:8.2f} ms   speedup={t1/t:.2f}x")


if __name__ == "__main__":
    main()
