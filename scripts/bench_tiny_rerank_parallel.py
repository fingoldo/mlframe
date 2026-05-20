"""Quick wall-time benchmark for the parallel ``_tiny_model_rerank`` path.

Measures serial (tiny_rerank_n_jobs=1) vs parallel (tiny_rerank_n_jobs=N)
on a realistic-shape problem: subsample=200k, K kept specs, 1-3 families,
deterministic LightGBM CV. Shared base/x_matrix arrays mean each parallel
spec contributes ~1 LightGBM CV worth of work to the threading pool.

Usage: python scripts/bench_tiny_rerank_parallel.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlframe.training.composite_discovery import CompositeTargetDiscovery
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def build_problem(n: int = 200_000, n_features: int = 12, seed: int = 11):
    rng = np.random.default_rng(seed)
    cols = {}
    base1 = rng.normal(50.0, 10.0, n).clip(min=1.0)
    base2 = rng.normal(20.0, 5.0, n).clip(min=1.0)
    cols["base1"] = base1
    cols["base2"] = base2
    for i in range(n_features):
        cols[f"f{i}"] = rng.normal(size=n)
    # Multiplicative + linear mixture so ratio / logratio / linear_residual
    # all have signal to recover.
    y = (
        base1 * (1.0 + 0.05 * cols["f0"])
        + 0.4 * cols["f1"]
        - 0.3 * cols["f2"]
        + 0.2 * cols["f3"] * cols["f4"]
        + rng.normal(0.0, 1.0, n)
    )
    df = pd.DataFrame(cols)
    return df, y


def run_once(n_jobs: int, df: pd.DataFrame, y: np.ndarray,
             tiny_model_sample_n: int = 50_000,
             discovery_n_jobs: int = 1) -> tuple[float, dict]:
    df_with_y = df.copy()
    df_with_y["y"] = y
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True,
        transforms=[
            "linear_residual", "diff", "ratio", "logratio",
            "monotonic_residual", "quantile_residual",
        ],
        mi_nbins=8,
        mi_estimator="bin",
        top_k_after_mi=8,
        eps_mi_gain=-1.0,
        random_state=11,
        discovery_n_jobs=discovery_n_jobs,
        mi_gain_bootstrap_n=0,
        detect_linear_residual_alpha_drift=False,
        base_candidates=["base1", "base2"],
        screening="tiny_model",
        tiny_screening_models="single_lgbm",
        tiny_model_sample_n=tiny_model_sample_n,
        tiny_model_n_estimators=60,
        tiny_model_cv_folds=3,
        tiny_model_n_seed_repeats=1,
        use_wilcoxon_gate=False,
        deterministic_screening_models=True,
        require_beats_raw_baseline=False,
        tiny_rerank_n_jobs=n_jobs,
    )
    discovery = CompositeTargetDiscovery(config=cfg)
    n = len(df)
    train_idx = np.arange(int(0.8 * n))
    feature_cols = list(df.columns)
    t0 = time.perf_counter()
    discovery.fit(
        df=df_with_y,
        target_col="y",
        feature_cols=feature_cols,
        train_idx=train_idx,
    )
    elapsed = time.perf_counter() - t0
    scores = dict(getattr(discovery, "_tiny_rerank_scores", {}))
    return elapsed, scores


def main():
    print("Building n=200k synthetic problem...")
    df, y = build_problem(n=200_000, n_features=12)
    print(f"  shape: {df.shape}, target std={float(np.std(y)):.3f}")

    print()
    print("Warm-up (JIT + cache) ...")
    run_once(n_jobs=1, df=df, y=y)

    print()
    print("Serial run (tiny_rerank_n_jobs=1) ...")
    t_serial_1, scores_serial = run_once(n_jobs=1, df=df, y=y)
    print(f"  wall-time: {t_serial_1:.2f}s  n_specs={len(scores_serial)}")

    print()
    print("Parallel run (tiny_rerank_n_jobs=4) ...")
    t_par_4, scores_par_4 = run_once(n_jobs=4, df=df, y=y)
    print(f"  wall-time: {t_par_4:.2f}s  n_specs={len(scores_par_4)}")

    print()
    print("Parallel run (tiny_rerank_n_jobs=8) ...")
    t_par_8, scores_par_8 = run_once(n_jobs=8, df=df, y=y)
    print(f"  wall-time: {t_par_8:.2f}s  n_specs={len(scores_par_8)}")

    print()
    print("Combined: discovery_n_jobs=4 + tiny_rerank_n_jobs=4 ...")
    t_combo, scores_combo = run_once(
        n_jobs=4, df=df, y=y, discovery_n_jobs=4,
    )
    print(f"  wall-time: {t_combo:.2f}s  n_specs={len(scores_combo)}")

    print()
    print("=== Speedup summary ===")
    print(f"  serial:        {t_serial_1:.2f}s")
    print(f"  parallel n=4:  {t_par_4:.2f}s  ({t_serial_1 / max(t_par_4, 1e-9):.2f}x)")
    print(f"  parallel n=8:  {t_par_8:.2f}s  ({t_serial_1 / max(t_par_8, 1e-9):.2f}x)")
    print(f"  combo n=4+4:   {t_combo:.2f}s  ({t_serial_1 / max(t_combo, 1e-9):.2f}x)")
    print()
    print("=== Equivalence check ===")
    all_keys = set(scores_serial) | set(scores_par_4) | set(scores_par_8)
    max_diff = 0.0
    for k in all_keys:
        s = scores_serial.get(k, float("nan"))
        p4 = scores_par_4.get(k, float("nan"))
        p8 = scores_par_8.get(k, float("nan"))
        if np.isfinite(s) and np.isfinite(p4):
            max_diff = max(max_diff, abs(s - p4))
        if np.isfinite(s) and np.isfinite(p8):
            max_diff = max(max_diff, abs(s - p8))
    print(f"  max RMSE diff vs serial: {max_diff:.2e}")
    if max_diff > 1e-6:
        print("  WARNING: parallel results diverged from serial!")
        for k in sorted(all_keys):
            print(f"    {k}: serial={scores_serial.get(k)} "
                  f"par4={scores_par_4.get(k)} par8={scores_par_8.get(k)}")


if __name__ == "__main__":
    main()
