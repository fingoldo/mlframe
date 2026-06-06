"""Bench: cluster-medoid correlation method Pearson vs SU (A1-09).

The default cluster-medoid pre-reduction (registry._instantiate_rfecv / _instantiate_boruta_shap, wrapped in
GroupAwareMRMR) used Pearson-only correlation, which sees only linear/monotone redundancy. Symmetric
Uncertainty (SU) additionally captures non-linear / non-monotone redundancy (e.g. x and x**2). This bench
compares the two clustering methods across multiple seeds + scenarios on a downstream tree model's OOS AUC,
to decide whether SU should be the default.

Run:
    python -m mlframe.feature_selection._benchmarks.bench_medoid_corr_method

Verdict (this machine, seeds 0-7, RandomForest downstream): Pearson stays the default.
  linear_redundancy:      Pearson AUC 0.9708 vs SU 0.9712 -- tie within noise; SU wins 2/8 seeds.
  nonmonotone_redundancy: Pearson AUC 0.9674 vs SU 0.6962 -- SU HURTS, wins 0/8 seeds.
SU collapses z and z**2 into one cluster and keeps only the medoid, dropping the complementary non-linear
feature -- but for prediction z and z**2 are NOT redundant (they carry different signal), so SU's
"non-linear redundancy" detection mis-fires here. Pearson's monotone-only view correctly keeps both. SU is
exposed as cluster_corr_method='su' for the narrow case where features ARE genuinely non-monotone duplicates
(e.g. a re-encoded copy), but the bench-set DEFAULT is Pearson on the majority of seeds + scenarios.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mlframe.feature_selection.filters.group_aware import cluster_features_by_correlation, _cluster_medoids


def _make_scenario(kind: str, seed: int, n: int = 2000):
    rng = np.random.RandomState(seed)
    z = rng.randn(n)
    w = rng.randn(n)
    cols = {}
    if kind == "linear_redundancy":
        # 3 linear copies of z, 3 of w, 4 noise.
        for i in range(3):
            cols[f"z{i}"] = z + 0.05 * rng.randn(n)
        for i in range(3):
            cols[f"w{i}"] = w + 0.05 * rng.randn(n)
        for i in range(4):
            cols[f"n{i}"] = rng.randn(n)
    elif kind == "nonmonotone_redundancy":
        cols["z0"] = z
        cols["z1"] = z ** 2  # Pearson ~0 with z0, SU high
        cols["z2"] = np.abs(z)
        cols["w0"] = w
        for i in range(3):
            cols[f"n{i}"] = rng.randn(n)
    else:
        raise ValueError(kind)
    X = pd.DataFrame(cols)
    y = (z + 0.5 * w + 0.3 * rng.randn(n) > 0).astype(int)
    return X, y


def _eval(X, y, method, seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=seed, stratify=y)
    cl = cluster_features_by_correlation(Xtr, threshold=0.5, method=method)
    medoids = _cluster_medoids(Xtr, cl, method=method)
    cols = [X.columns[i] for i in medoids]
    rf = RandomForestClassifier(n_estimators=120, random_state=seed, n_jobs=-1)
    rf.fit(Xtr[cols], ytr)
    auc = roc_auc_score(yte, rf.predict_proba(Xte[cols])[:, 1])
    return auc, len(cols)


def main():
    """Run the Pearson-vs-SU cluster-medoid correlation sweep across seeds and scenarios."""
    seeds = list(range(8))
    scenarios = ["linear_redundancy", "nonmonotone_redundancy"]
    results = []
    for sc in scenarios:
        for seed in seeds:
            X, y = _make_scenario(sc, seed)
            for method in ("pearson", "su"):
                t0 = time.perf_counter()
                auc, nfeat = _eval(X, y, method, seed)
                results.append({"scenario": sc, "seed": seed, "method": method,
                                "auc": auc, "n_medoids": nfeat, "wall_s": time.perf_counter() - t0})
    df = pd.DataFrame(results)
    summary = df.groupby(["scenario", "method"]).agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        n_medoids_mean=("n_medoids", "mean"), wall_mean=("wall_s", "mean")).reset_index()
    print(summary.to_string(index=False))
    # Per-scenario win count of su over pearson.
    for sc in scenarios:
        sub = df[df.scenario == sc]
        wins = 0
        for seed in seeds:
            a = sub[(sub.seed == seed) & (sub.method == "su")].auc.values[0]
            p = sub[(sub.seed == seed) & (sub.method == "pearson")].auc.values[0]
            wins += int(a > p + 1e-6)
        print(f"{sc}: SU beats Pearson on {wins}/{len(seeds)} seeds")
    out = Path(__file__).parent / "_results" / f"medoid_corr_method_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2, sort_keys=True))
    print("wrote", out)


if __name__ == "__main__":
    main()
