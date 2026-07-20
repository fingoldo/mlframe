"""Benchmark: matrix-based Rényi alpha-entropy MI vs plug-in / mixed-KSG accuracy on small n.

Run: PYTHONPATH=src python -m mlframe.feature_selection._benchmarks.bench_renyi_alpha_small_n

Backs the MRMR_RESEARCH.md "still forward research" item #5 (matrix-based Rényi alpha-entropy CMI
estimator, Yu et al. 2020) now that it's wired as ``estimator='renyi_alpha'`` in ``_mi_dispatch.py``.
The paper's own claim is that this estimator beats plug-in on SMALL n (<500) where the discretization
bias of a histogram-based MI is worst. Ground truth here is the closed-form bivariate Gaussian MI:
``I(X;Y) = -0.5 * log(1 - rho^2)`` (nats), converted to bits for comparison with the bit-valued
estimators in ``_mi_dispatch.py``.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mi_dispatch import score_pair_mi


def _true_gaussian_mi_bits(rho: float) -> float:
    """Closed-form MI (bits) for a bivariate Gaussian with correlation rho."""
    nats = -0.5 * np.log(1.0 - rho * rho)
    return float(nats / np.log(2.0))


def _make_correlated_gaussian(seed: int, n: int, rho: float):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = rho * x + np.sqrt(max(0.0, 1.0 - rho * rho)) * z
    return x, y


def main():
    """Sweep small-n regimes x correlation strengths across estimators, report bias vs the closed-form Gaussian MI + wall-time."""
    estimators = ["plug_in", "mixed_ksg", "renyi_alpha"]
    ns = [100, 200, 500, 1000]
    rhos = [0.3, 0.6, 0.9]
    seeds = [0, 1, 2]

    rows = []
    for n in ns:
        for rho in rhos:
            true_mi = _true_gaussian_mi_bits(rho)
            for estimator in estimators:
                errs, times = [], []
                for seed in seeds:
                    x, y = _make_correlated_gaussian(seed, n, rho)
                    t0 = time.perf_counter()
                    mi_hat = score_pair_mi(x, y, estimator=estimator)
                    dt = time.perf_counter() - t0
                    errs.append(abs(mi_hat - true_mi))
                    times.append(dt)
                rows.append(
                    dict(
                        n=n,
                        rho=rho,
                        estimator=estimator,
                        true_mi=round(true_mi, 4),
                        mean_abs_err=round(float(np.mean(errs)), 4),
                        mean_time_s=round(float(np.mean(times)), 4),
                    )
                )
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Headline: does renyi_alpha beat plug_in's mean_abs_err at the smallest n (100, 200)?
    small_n = df[df["n"].isin([100, 200])]
    pivot = small_n.pivot_table(index=["n", "rho"], columns="estimator", values="mean_abs_err")
    print("\nSmall-n (n<=200) mean_abs_err pivot:")
    print(pivot.to_string())
    if "renyi_alpha" in pivot.columns and "plug_in" in pivot.columns:
        wins = int((pivot["renyi_alpha"] < pivot["plug_in"]).sum())
        total = len(pivot)
        print(f"\nrenyi_alpha beats plug_in on {wins}/{total} small-n (n, rho) cells.")
    return df


if __name__ == "__main__":
    main()
