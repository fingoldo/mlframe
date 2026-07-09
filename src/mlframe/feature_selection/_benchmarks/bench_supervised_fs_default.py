"""Bench: should the suite default-on a cheap supervised FS filter? (A1-02).

The training suite's default FS is unsupervised-only (variance==0 / nulls>99% pre-screen); no supervised filter
runs unless the operator opts into MRMR / RFECV / BorutaShap. A1-02 asks whether a CHEAP supervised filter
(univariate mutual-information top-k) default-on improves honest-holdout OOS on representative synthetics. Flip
the default only if it wins on the MAJORITY of seeds.

Compares, per seed/scenario, downstream OOS AUC with:
  - none      : all features (current default; only the trivial unsupervised pre-screen)
  - mi_topk   : sklearn mutual_info_classif top-k (cheap supervised filter)

Run:
    python -m mlframe.feature_selection._benchmarks.bench_supervised_fs_default

Verdict (this machine, seeds 0-5): default stays UNSUPERVISED-ONLY (no auto supervised FS).
mi_topk vs none, win count over 6 seeds per (model, scenario):
  logreg many_noise 6/6  wide_sparse 6/6  few_noise 2/6  -- linear downstream clearly benefits
  rf     many_noise 3/6  wide_sparse 5/6  few_noise 0/6  -- tree downstream noise-robust; HURT on low-noise
mi_topk does NOT win on the majority across BOTH model families (it hurts trees on few_noise: AUC 0.952 vs
0.975). The benefit is model- + data-dependent (linear + wide/noisy data), which is precisely why supervised FS
stays OPT-IN via MRMR / RFECV / BorutaShap rather than default-on. Documented in FeatureSelectionConfig.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _make(scenario, seed, n=2500):
    if scenario == "many_noise":
        X, y = make_classification(n_samples=n, n_features=80, n_informative=10, n_redundant=10, n_repeated=0, random_state=seed)
    elif scenario == "few_noise":
        X, y = make_classification(n_samples=n, n_features=25, n_informative=12, n_redundant=5, n_repeated=0, random_state=seed)
    elif scenario == "wide_sparse":
        X, y = make_classification(n_samples=n, n_features=150, n_informative=8, n_redundant=4, n_repeated=0, random_state=seed)
    else:
        raise ValueError(scenario)
    cols = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols), y


def _eval(X, y, fs, seed, model_kind):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=seed, stratify=y)
    if fs == "none":
        cols = list(X.columns)
    elif fs == "mi_topk":
        mi = mutual_info_classif(Xtr, ytr, random_state=seed)
        k = max(5, int(np.ceil((mi > mi.mean()).sum())))  # keep the above-mean-MI features (cheap, data-driven)
        cols = [X.columns[i] for i in np.argsort(mi)[::-1][:k]]
    else:
        raise ValueError(fs)
    if model_kind == "rf":
        model = RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=-1)
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    model.fit(Xtr[cols], ytr)
    auc = roc_auc_score(yte, model.predict_proba(Xte[cols])[:, 1])
    return auc, len(cols)


def main():
    """Run the cheap-supervised-FS-default benchmark across seeds and scenarios and print the verdict."""
    seeds = list(range(6))
    scenarios = ["many_noise", "few_noise", "wide_sparse"]
    results = []
    for model_kind in ("rf", "logreg"):
        for sc in scenarios:
            for seed in seeds:
                X, y = _make(sc, seed)
                for fs in ("none", "mi_topk"):
                    auc, nfeat = _eval(X, y, fs, seed, model_kind)
                    results.append({"model": model_kind, "scenario": sc, "seed": seed, "fs": fs, "auc": auc, "n_features": nfeat})
    df = pd.DataFrame(results)
    summary = df.groupby(["model", "scenario", "fs"]).agg(auc_mean=("auc", "mean"), n_mean=("n_features", "mean")).reset_index()
    print(summary.to_string(index=False))
    # Majority-of-seeds win count for mi_topk over none, per (model, scenario).
    print("\nmi_topk beats none (per model+scenario, seeds win count):")
    for model_kind in ("rf", "logreg"):
        for sc in scenarios:
            sub = df[(df.model == model_kind) & (df.scenario == sc)]
            wins = 0
            for seed in seeds:
                a = sub[(sub.seed == seed) & (sub.fs == "mi_topk")].auc.values[0]
                b = sub[(sub.seed == seed) & (sub.fs == "none")].auc.values[0]
                wins += int(a > b + 1e-6)
            print(f"  {model_kind:6s} {sc:12s}: {wins}/{len(seeds)}")
    out = Path(__file__).parent / "_results" / f"supervised_fs_default_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2, sort_keys=True))
    print("wrote", out)


if __name__ == "__main__":
    main()
