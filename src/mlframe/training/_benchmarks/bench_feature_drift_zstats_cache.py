"""Microbench: target-invariant z-stats cache in compute_feature_distribution_drift.

The per-feature train/val/test z-stats + categorical PSI are functions of the
(train, val, test) frames ONLY -- they do NOT depend on feature_importance /
target_type / linear_shape. Only the FI-weighted aggregate + override decision
vary per target. compute_feature_distribution_drift is called once per target
(_phase_diagnostics.run_per_target_diagnostics), so the z-stats are recomputed
N times for N targets on the SAME filtered frames.

This bench measures wall for 1 vs 3 vs 5 targets with and without the cache to
quantify the per-target-multiplied recompute. Run:

    CUDA_VISIBLE_DEVICES="" python -m mlframe.training._benchmarks.bench_feature_drift_zstats_cache
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mlframe.training import feature_drift_report as _fdr
from mlframe.training.feature_drift_report import compute_feature_distribution_drift


def _make_frames(n: int, k_num: int, k_cat: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    data = {}
    for j in range(k_num):
        data[f"num_{j}"] = rng.standard_normal(n)
    for j in range(k_cat):
        data[f"cat_{j}"] = rng.integers(0, 30, size=n).astype(str)
    train = pd.DataFrame(data)
    val = pd.DataFrame({c: rng.permutation(train[c].to_numpy()) for c in train.columns})
    test = pd.DataFrame({c: rng.permutation(train[c].to_numpy()) for c in train.columns})
    return train, val, test


def _fi_for_target(cols, seed):
    rng = np.random.default_rng(1000 + seed)
    return {c: float(abs(rng.standard_normal())) for c in cols}


def _time_n_targets(train, val, test, n_targets: int, iters: int, *, cache: bool) -> float:
    """Wall for one suite run = N per-target drift calls on the SAME frames.

    Each iter starts cold (cache cleared) so the measurement is 1 cold target +
    (N-1) warm targets -- exactly what one suite run sees. ``cache=False`` clears
    before EVERY call to emulate the pre-cache all-fresh path.
    """
    num_cols = [c for c in train.columns if c.startswith("num_")]
    fis = [_fi_for_target(num_cols, t) for t in range(n_targets)]
    best = float("inf")
    for _ in range(iters):
        _fdr._DRIFT_INVARIANT_CACHE.clear()
        t0 = time.perf_counter()
        for t in range(n_targets):
            if not cache:
                _fdr._DRIFT_INVARIANT_CACHE.clear()
            compute_feature_distribution_drift(
                train, val, test,
                feature_importance=fis[t],
                target_type="regression",
            )
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def main() -> None:
    # Production-ish: 80k rows, 30 numeric + 8 categorical features.
    n, k_num, k_cat = 80_000, 30, 8
    train, val, test = _make_frames(n, k_num, k_cat)
    print(f"frames: n={n} k_num={k_num} k_cat={k_cat}")
    print(f"{'targets':>8} {'fresh_ms':>10} {'cached_ms':>10} {'speedup':>9}")
    for n_targets in (1, 3, 5):
        fresh = _time_n_targets(train, val, test, n_targets, iters=5, cache=False)
        cached = _time_n_targets(train, val, test, n_targets, iters=5, cache=True)
        print(f"{n_targets:>8} {fresh:>10.2f} {cached:>10.2f} {fresh / cached:>8.2f}x")


if __name__ == "__main__":
    main()
