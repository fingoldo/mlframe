"""Sweep the MAD gate factor for the robust float-ensemble aggregator.

``robust_float_ensemble`` drops, per output column, members whose |dev from column median| exceeds
``mad_factor`` * scaled-MAD, then averages survivors. A small ``mad_factor`` over-fires on normal fold
spread at small K and costs RMSE in the all-clean regime; a large one barely protects against corrupted
folds. This bench sweeps ``mad_factor`` over a grid and measures, per cell, BOTH:

  (a) clean-regime RMSE ratio vs plain mean (want <=1.01, i.e. <=1% cost), and
  (b) outlier-regime RMSE ratio vs plain mean (want a large protection, >=2x cut on 1-2 corrupted folds).

The smallest factor whose clean cost is <=1% on EVERY clean cell while keeping >=2x protection on the
corrupted cells is the production default candidate. Run::

    python -m mlframe.models.ensembling._benchmarks.bench_mad_factor_sweep
"""

from __future__ import annotations

import numpy as np

from mlframe.models.ensembling.float_aggregation import robust_float_ensemble

FACTORS = (3.5, 4.0, 4.5, 5.0, 6.0, 8.0)
SEEDS = (0, 1, 2, 3, 4)
CLEAN_K = (3, 5, 8)
OUTLIER_CELLS = ((5, 1), (5, 2), (8, 1), (8, 2))  # (K, n_bad)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _make_y(rng: np.random.Generator, n: int):
    x = rng.normal(size=(n, 4))
    return x @ np.array([1.5, -2.0, 0.7, 1.1]) + rng.normal(0.0, 0.5, size=n)


def _clean_members(rng, y, k):
    sd = float(np.std(y))
    return np.stack([y + rng.normal(0.0, 0.3 * sd, size=y.shape[0]) for _ in range(k)])


def _outlier_members(rng, y, k, n_bad):
    sd = float(np.std(y))
    n = y.shape[0]
    members = [y + rng.normal(0.0, 0.3 * sd, size=n) for _ in range(k - n_bad)]
    for j in range(n_bad):
        if j % 2 == 0:
            members.append(y + 4.0 * sd)            # biased outlier fold
        else:
            members.append(2.5 * y + rng.normal(0.0, 0.3 * sd, size=n))  # scale outlier fold
    return np.stack(members)


def run(n: int = 3000):
    clean = {f: [] for f in FACTORS}            # ratio robust/mean per clean cell
    outlier = {f: [] for f in FACTORS}          # ratio mean/robust (protection) per outlier cell

    for k in CLEAN_K:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            y = _make_y(rng, n)
            members = _clean_members(rng, y, k)
            rmse_mean = _rmse(members.mean(axis=0), y)
            for f in FACTORS:
                rmse_r = _rmse(robust_float_ensemble(members, mad_factor=f), y)
                clean[f].append(rmse_r / rmse_mean)

    for k, n_bad in OUTLIER_CELLS:
        for seed in SEEDS:
            rng = np.random.default_rng(100 + seed)
            y = _make_y(rng, n)
            members = _outlier_members(rng, y, k, n_bad)
            rmse_mean = _rmse(members.mean(axis=0), y)
            for f in FACTORS:
                rmse_r = _rmse(robust_float_ensemble(members, mad_factor=f), y)
                outlier[f].append(rmse_mean / max(rmse_r, 1e-12))

    print(f"n={n}, clean K={CLEAN_K} x {len(SEEDS)} seeds, outlier cells (K,n_bad)={OUTLIER_CELLS} x {len(SEEDS)} seeds")
    print("\nfactor  clean_ratio_max  clean_ratio_mean  clean_ok(<=1.01)  outlier_protect_min  outlier_protect_mean  protect_ok(>=2x)")
    chosen = None
    for f in FACTORS:
        cmax = max(clean[f]); cmean = float(np.mean(clean[f]))
        omin = min(outlier[f]); omean = float(np.mean(outlier[f]))
        clean_ok = cmax <= 1.01
        protect_ok = omin >= 2.0
        print(f"{f:>5.1f}  {cmax:>15.4f}  {cmean:>16.4f}  {str(clean_ok):>16}  {omin:>19.3f}  {omean:>20.3f}  {str(protect_ok):>16}")
        if chosen is None and clean_ok and protect_ok:
            chosen = f

    print(f"\nCHOSEN smallest factor (clean<=1% AND protect>=2x): {chosen}")
    return clean, outlier, chosen


if __name__ == "__main__":
    run()
