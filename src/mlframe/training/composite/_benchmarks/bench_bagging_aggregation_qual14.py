"""qual-14 bench: BaggingEnsemble member-aggregation default -- mean vs median on honest holdout.

Question: ``BaggedCompositeEstimator.predict`` collapses the (n_members, n_rows) member-prediction matrix with ``mean(axis=0)``.
For bagging on heavy-tailed / outlier-contaminated regression targets, the across-member MEDIAN is more robust to the occasional
wild member prediction (a member that bootstrapped a pathological resample). This bench measures honest holdout RMSE + MAE for
mean vs median aggregation across >=5 seeds AND >=2 scenarios (heavy-tailed target/noise vs clean Gaussian) to decide the default.

Run: python -m mlframe.training.composite._benchmarks.bench_bagging_aggregation_qual14
"""
from __future__ import annotations

import sys

import numpy as np

sys.modules.setdefault("cupy", None)

from sklearn.tree import DecisionTreeRegressor

from mlframe.training.composite.bagging import BaggedCompositeEstimator


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def _make_data(scenario: str, seed: int, n: int = 1200, p: int = 8):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    truth = 2.0 * X[:, 0] + 1.5 * X[:, 1] * X[:, 2] - 1.0 * X[:, 3] + 0.7 * np.sin(2.0 * X[:, 4])
    if scenario == "heavy_tail":
        # Student-t(df=2) noise: heavy tails -> occasional huge residual, the regime where a robust aggregator should pay off.
        noise = rng.standard_t(df=2, size=n) * 0.8
    elif scenario == "outlier_contam":
        # Mostly-clean Gaussian noise with a 5% contamination of large-magnitude spikes.
        noise = rng.randn(n) * 0.5
        mask = rng.rand(n) < 0.05
        noise[mask] += rng.randn(mask.sum()) * 12.0
    else:  # clean
        noise = rng.randn(n) * 0.5
    y = truth + noise
    cut = int(n * 0.7)
    return X[:cut], y[:cut], X[cut:], truth[cut:]


def _trimmed_mean(members: np.ndarray, trim: float = 0.1) -> np.ndarray:
    """Symmetric trimmed mean across members: sort each column, drop the lowest/highest ``trim`` fraction, mean the rest."""
    m = members.shape[0]
    k = int(np.floor(trim * m))
    if k == 0:
        return members.mean(axis=0)
    s = np.sort(members, axis=0)
    return s[k : m - k].mean(axis=0)


def run():
    scenarios = ["heavy_tail", "outlier_contam", "clean"]
    seeds = [0, 1, 2, 3, 4, 5, 6]
    print(f"{'scenario':<16}{'seed':>5}{'rmse_mean':>11}{'rmse_med':>11}{'rmse_trim':>11}{'mae_mean':>11}{'mae_med':>11}{'mae_trim':>11}")
    agg = {s: {"med_r": 0, "trim_r": 0, "med_m": 0, "trim_m": 0, "n": 0} for s in scenarios}
    for sc in scenarios:
        for sd in seeds:
            Xtr, ytr, Xte, truth_te = _make_data(sc, sd)
            proto = DecisionTreeRegressor(max_depth=6, random_state=0)
            bag = BaggedCompositeEstimator(base_estimator=proto, n_estimators=25, random_state=sd)
            bag.fit(Xtr, ytr)
            members = bag._member_predictions(Xte)  # (n_members, n_rows)
            p_mean = members.mean(axis=0)
            p_med = np.median(members, axis=0)
            p_trim = _trimmed_mean(members, trim=0.1)
            rm, rmd, rt = _rmse(p_mean, truth_te), _rmse(p_med, truth_te), _rmse(p_trim, truth_te)
            am, amd, at = _mae(p_mean, truth_te), _mae(p_med, truth_te), _mae(p_trim, truth_te)
            agg[sc]["med_r"] += int(rmd < rm)
            agg[sc]["trim_r"] += int(rt < rm)
            agg[sc]["med_m"] += int(amd < am)
            agg[sc]["trim_m"] += int(at < am)
            agg[sc]["n"] += 1
            print(f"{sc:<16}{sd:>5}{rm:>11.4f}{rmd:>11.4f}{rt:>11.4f}{am:>11.4f}{amd:>11.4f}{at:>11.4f}")
    print("\n=== aggregate (wins-over-mean / total) ===")
    for sc in scenarios:
        a = agg[sc]
        print(f"{sc:<16} RMSE: median {a['med_r']}/{a['n']}, trim {a['trim_r']}/{a['n']}   MAE: median {a['med_m']}/{a['n']}, trim {a['trim_m']}/{a['n']}")


if __name__ == "__main__":
    run()
