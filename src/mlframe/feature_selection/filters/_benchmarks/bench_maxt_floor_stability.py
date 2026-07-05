"""Run-to-run stability of the order-1 maxT noise floor vs the permutation count K.

The floor is the 95th percentile of a per-shuffle MAX corrected-MI over a noise pool -- an extreme upper-tail order statistic. At K draws only ~K*(1-q) land above
the 95th percentile, so a small K estimates that quantile from almost nothing and the floor wobbles run-to-run on the SAME data. This bench fixes a pure-noise pool
(true signal = 0), recomputes the floor over many independent permutation seeds at each K, and reports the across-seed std of the floor. It proves the new default
(K=200) yields a several-fold lower-variance floor than the legacy K=25, justifying the default flip.

Run: python -m mlframe.feature_selection.filters._benchmarks.bench_maxt_floor_stability
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._permutation_null import pooled_permutation_null_gain_floor


def _make_noise_pool(n: int, p: int, nb: int, seed: int):
    """A pool of p independent uniform-binned noise columns + an independent target (true MI = 0 for every column)."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, nb, size=(n, p + 1)).astype(np.int64)
    nbins = np.full(p + 1, nb, dtype=np.int64)
    return data, nbins, np.arange(p), p


def floor_std_over_seeds(n: int, p: int, nb: int, k: int, n_seeds: int) -> tuple[float, float]:
    """Mean and across-seed std of the floor at permutation count ``k`` on a FIXED noise pool, varying only the permutation RNG seed."""
    data, nbins, cand, y_idx = _make_noise_pool(n, p, nb, seed=12345)
    floors = np.empty(n_seeds)
    for s in range(n_seeds):
        floors[s] = pooled_permutation_null_gain_floor(data, nbins, cand, y_idx, n_permutations=k, quantile=0.95, random_seed=1000 + s)
    return float(floors.mean()), float(floors.std(ddof=1))


def main() -> None:
    n, p, nb, n_seeds = 1500, 40, 8, 40
    print(f"maxT order-1 noise-floor stability  (n={n}, p={p}, nbins={nb}, 95th pct, {n_seeds} seeds)\n")
    print(f"{'K':>5} {'mean floor':>14} {'std floor':>14}")
    results = {}
    for k in (25, 100, 200):
        mean, std = floor_std_over_seeds(n, p, nb, k, n_seeds)
        results[k] = std
        print(f"{k:>5} {mean:>14.6e} {std:>14.6e}")
    print(f"\nstd reduction K=25 -> K=200 : {results[25] / results[200]:.2f}x lower")
    print(f"std reduction K=25 -> K=100 : {results[25] / results[100]:.2f}x lower")


if __name__ == "__main__":
    main()
