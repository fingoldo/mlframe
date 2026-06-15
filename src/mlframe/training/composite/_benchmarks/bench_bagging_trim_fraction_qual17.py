"""qual-17 bench: BaggedCompositeEstimator trimmed-mean ``trim_fraction`` default sweep on honest holdout.

qual-14 set ``aggregation="trimmed_mean"`` with a fixed ``trim_fraction=0.1``. The open question (this iter): is a DIFFERENT
trim fraction a better default across MORE contamination levels? A larger trim is more robust to heavy contamination but bleeds
efficiency on clean data; a smaller trim is near-mean on clean data but under-protects under heavy outliers. We sweep
trim in {0.0(=mean), 0.05, 0.1, 0.15, 0.2, 0.25, 0.30} across contamination scenarios (clean / heavy-tail t(2) / 5% / 15% / 30%
outlier-contam) x >=5 seeds, honest holdout RMSE + MAE vs the noise-free truth. The winner = lowest mean honest RMSE summed
across scenarios, with a per-scenario per-seed win-count majority check vs the incumbent 0.1.

Run: python -m mlframe.training.composite._benchmarks.bench_bagging_trim_fraction_qual17
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
        noise = rng.standard_t(df=2, size=n) * 0.8
    elif scenario.startswith("contam"):
        frac = float(scenario.split("_")[1]) / 100.0
        noise = rng.randn(n) * 0.5
        mask = rng.rand(n) < frac
        noise[mask] += rng.randn(mask.sum()) * 12.0
    else:  # clean
        noise = rng.randn(n) * 0.5
    y = truth + noise
    cut = int(n * 0.7)
    return X[:cut], y[:cut], X[cut:], truth[cut:]


def _trim(members: np.ndarray, trim: float) -> np.ndarray:
    m = members.shape[0]
    k = int(np.floor(trim * m))
    if k == 0:
        return members.mean(axis=0)
    s = np.sort(members, axis=0)
    return s[k:m - k].mean(axis=0)


def run():
    scenarios = ["clean", "heavy_tail", "contam_05", "contam_15", "contam_30"]
    seeds = [0, 1, 2, 3, 4, 5, 6]
    trims = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    incumbent = 0.10

    # members[scenario][seed] is computed ONCE (same fitted bag) then all trims applied to the same matrix -> apples-to-apples.
    rmse_sum = {t: 0.0 for t in trims}
    mae_sum = {t: 0.0 for t in trims}
    # win counts vs incumbent per scenario (lower honest RMSE)
    wins = {t: {sc: 0 for sc in scenarios} for t in trims}
    totals = {sc: 0 for sc in scenarios}
    per_scen_rmse = {sc: {t: 0.0 for t in trims} for sc in scenarios}

    for sc in scenarios:
        for sd in seeds:
            Xtr, ytr, Xte, truth_te = _make_data(sc, sd)
            proto = DecisionTreeRegressor(max_depth=6, random_state=0)
            bag = BaggedCompositeEstimator(base_estimator=proto, n_estimators=25, random_state=sd)
            bag.fit(Xtr, ytr)
            members = bag._member_predictions(Xte)
            r_inc = _rmse(_trim(members, incumbent), truth_te)
            totals[sc] += 1
            for t in trims:
                pred = _trim(members, t)
                r = _rmse(pred, truth_te)
                a = _mae(pred, truth_te)
                rmse_sum[t] += r
                mae_sum[t] += a
                per_scen_rmse[sc][t] += r
                if r < r_inc:
                    wins[t][sc] += 1

    print(f"{'trim':>6}{'sum_rmse':>11}{'sum_mae':>11}   per-scenario mean RMSE")
    for t in trims:
        per = "  ".join(f"{sc}:{per_scen_rmse[sc][t]/totals[sc]:.4f}" for sc in scenarios)
        tag = "  <- incumbent" if abs(t - incumbent) < 1e-9 else ""
        print(f"{t:>6.2f}{rmse_sum[t]:>11.4f}{mae_sum[t]:>11.4f}   {per}{tag}")

    best = min(trims, key=lambda t: rmse_sum[t])
    print(f"\nbest sum_rmse trim = {best:.2f}  (incumbent {incumbent:.2f})")
    print("\n=== per-trim wins-over-incumbent(0.10) by scenario (lower honest RMSE), out of total seeds ===")
    for t in trims:
        if abs(t - incumbent) < 1e-9:
            continue
        ws = "  ".join(f"{sc}:{wins[t][sc]}/{totals[sc]}" for sc in scenarios)
        tot_w = sum(wins[t].values())
        tot_n = sum(totals.values())
        print(f"trim {t:.2f}: {ws}   TOTAL {tot_w}/{tot_n}")


if __name__ == "__main__":
    run()
