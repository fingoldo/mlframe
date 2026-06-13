"""Bench: mean vs median cross-member aggregation for the FLOAT (regression / quantile) ensemble.

The suite-wide float-prediction ensemble in ``_predict_main_suite`` / ``_predict_main_from_models``
aggregates per-model predictions with ``np.mean(axis=0)``. When some members are outlier folds /
seeds (a head that overfit one fold, an unstable seed, a degenerate config that survived), the
arithmetic mean is dragged toward the bad member; the cross-member MEDIAN is robust to a minority of
bad members and recovers the honest target better.

This bench measures honest-holdout RMSE of mean-aggregated vs median-aggregated ensemble predictions
across 5 synthetic regression scenarios x 3 seeds, where a configurable fraction of the K ensemble
members are corrupted (bias / scale / heavy-tail noise) to emulate outlier folds. A FLIP to median is
justified only on a MAJORITY win across the 15 cells.

Run::

    python -m mlframe.training._benchmarks.bench_ensemble_mean_vs_median_agg
"""

from __future__ import annotations

import numpy as np


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _make_members(rng: np.random.Generator, y: np.ndarray, k: int, n_bad: int, scenario: str):
    """Build K member-prediction arrays. ``k - n_bad`` are honest noisy estimators of ``y``;
    ``n_bad`` are corrupted outlier members (bias / scale / heavy tail)."""
    n = y.shape[0]
    members = []
    good_noise = 0.30 * float(np.std(y))
    for _ in range(k - n_bad):
        members.append(y + rng.normal(0.0, good_noise, size=n))
    for j in range(n_bad):
        if scenario == "bias":
            members.append(y + 4.0 * float(np.std(y)) + rng.normal(0.0, good_noise, size=n))
        elif scenario == "scale":
            members.append(2.5 * y + rng.normal(0.0, good_noise, size=n))
        elif scenario == "heavy_tail":
            members.append(y + rng.standard_t(df=2, size=n) * 3.0 * float(np.std(y)))
        elif scenario == "flip_sign":
            members.append(-y + rng.normal(0.0, good_noise, size=n))
        else:  # mixed
            if j % 2 == 0:
                members.append(y + 4.0 * float(np.std(y)))
            else:
                members.append(2.5 * y + rng.normal(0.0, good_noise, size=n))
    return np.stack(members)


SCENARIOS = ("bias", "scale", "heavy_tail", "flip_sign", "mixed")


def robust_aggregate(members: np.ndarray, mad_factor: float = 3.5) -> np.ndarray:
    """Mean unless a member is a per-column outlier (|member - median| > mad_factor * MAD); then
    drop the flagged member from that column's average. Falls back to mean where MAD==0."""
    med = np.median(members, axis=0)
    abs_dev = np.abs(members - med)
    mad = np.median(abs_dev, axis=0)
    scaled_mad = 1.4826 * mad
    safe = scaled_mad > 0
    thresh = np.where(safe, mad_factor * scaled_mad, np.inf)
    keep = abs_dev <= thresh
    keep_counts = keep.sum(axis=0)
    keep_counts = np.where(keep_counts == 0, members.shape[0], keep_counts)
    masked_sum = np.where(keep, members, 0.0).sum(axis=0)
    return masked_sum / keep_counts


def run(seeds=(0, 1, 2), k=6, n_bad=2, n=3000):
    rows = []
    mean_wins = 0
    median_wins = 0
    robust_wins = 0
    for scenario in SCENARIOS:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            x = rng.normal(size=(n, 4))
            y = x @ np.array([1.5, -2.0, 0.7, 1.1]) + rng.normal(0.0, 0.5, size=n)
            members = _make_members(rng, y, k, n_bad, scenario)
            mean_pred = members.mean(axis=0)
            median_pred = np.median(members, axis=0)
            robust_pred = robust_aggregate(members)
            rmse_mean = _rmse(mean_pred, y)
            rmse_median = _rmse(median_pred, y)
            rmse_robust = _rmse(robust_pred, y)
            trio = {"mean": rmse_mean, "median": rmse_median, "robust": rmse_robust}
            winner = min(trio, key=trio.get)
            mean_wins += winner == "mean"
            median_wins += winner == "median"
            robust_wins += winner == "robust"
            rows.append((scenario, seed, rmse_mean, rmse_median, rmse_robust, winner))

    print(f"K={k} members, n_bad={n_bad} corrupted, n={n}, honest-holdout RMSE (lower=better)")
    print("scenario     seed   mean_RMSE   median_RMSE   robust_RMSE   winner")
    for scenario, seed, rm, rmd, rr, w in rows:
        print(f"{scenario:<12} {seed:>4}   {rm:>9.4f}   {rmd:>11.4f}   {rr:>11.4f}   {w}")
    print(f"\nwins -- robust {robust_wins}/{len(rows)}, median {median_wins}/{len(rows)}, mean {mean_wins}/{len(rows)}")
    return rows, median_wins, mean_wins, robust_wins


if __name__ == "__main__":
    print("=== outlier-fold regime (n_bad=2 of K=6) ===")
    run(k=6, n_bad=2)
    print("\n=== clean regime (n_bad=0 of K=6) -- median must NOT hurt materially ===")
    run(k=6, n_bad=0)
    print("\n=== single-bad regime (n_bad=1 of K=5) ===")
    run(k=5, n_bad=1)
