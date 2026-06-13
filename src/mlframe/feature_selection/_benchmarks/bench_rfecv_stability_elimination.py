"""Bench: elimination_rule='stability' vs 'importance' (legacy + importance_agg='dispatched').

SPECULATIVE innovation -- stability-aware elimination. Per-iteration, RFECV eliminates the
lowest-aggregated-importance feature(s). 'stability' instead ranks for elimination by
mean_importance * fold_selection_frequency (fraction of folds where the feature lands in the
top-N cut), protecting a steady-mid-rank true feature from one-fold-noise eviction by a
high-mean-but-high-variance noise feature that spiked in one fold.

This is RELATED to importance_agg='dispatched' (which discounts tree value-CV in the RANKING).
We bench BOTH against 'importance' to see whether stability adds value ON TOP of dispatched.

Beds emphasise the failure mode: a few STEADY mid-strength true features among noise + a few
high-variance "lucky-spike" noise features. honest holdout AUC, multi-scenario x multi-seed.

Run (host env):
  CUDA_VISIBLE_DEVICES= MLFRAME_NO_CUDA_AUTOCONFIG=1 MLFRAME_KEEP_BROKEN_CUPY=1 \
    PYTHONPATH=src python -m mlframe.feature_selection._benchmarks.bench_rfecv_stability_elimination
"""
from __future__ import annotations

import os
import warnings

os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mlframe.feature_selection.wrappers.rfecv import RFECV

SEEDS = [0, 1, 2]


def make_bed(name, seed):
    """Return (X, y) with STEADY mid-rank true features + lucky-spike noise features."""
    rng = np.random.default_rng(seed)
    n = 900
    if name == "steady_mid_among_noise":
        n_strong, n_steady, n_noise = 2, 4, 24
    elif name == "many_steady":
        n_strong, n_steady, n_noise = 1, 6, 20
    elif name == "few_steady_lots_noise":
        n_strong, n_steady, n_noise = 2, 3, 40
    elif name == "balanced_signal":
        n_strong, n_steady, n_noise = 3, 4, 16
    elif name == "weak_steady":
        n_strong, n_steady, n_noise = 1, 5, 28
    else:
        raise ValueError(name)
    cols = {}
    logit = np.zeros(n)
    for i in range(n_strong):
        x = rng.standard_normal(n)
        cols[f"strong_{i}"] = x
        logit += 1.3 * x
    for i in range(n_steady):
        x = rng.standard_normal(n)
        cols[f"steady_{i}"] = x
        logit += 0.45 * x  # mid-strength, steadily useful
    for i in range(n_noise):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    X = pd.DataFrame(cols)
    return X, y


def _new_est(seed):
    return RandomForestClassifier(n_estimators=80, max_depth=6, n_jobs=-1, random_state=seed)


def fit_select(X, y, rule, importance_agg, seed):
    r = RFECV(
        estimator=_new_est(seed), cv=3, scoring=None, verbose=0,
        max_refits=8, random_state=seed,
        importance_getter="feature_importances_",
        importance_agg=importance_agg, elimination_rule=rule,
        n_features_selection_rule="one_se_min",
    )
    r.fit(X, y)
    return [c for c in r.get_feature_names_out() if c in X.columns]


def holdout_auc(Xtr, Xte, ytr, yte, cols, seed):
    if not cols:
        return float("nan")
    m = RandomForestClassifier(n_estimators=250, max_depth=8, n_jobs=-1, random_state=seed)
    m.fit(Xtr[cols], ytr)
    return float(roc_auc_score(yte, m.predict_proba(Xte[cols])[:, 1]))


def true_recall(cols):
    sel = set(cols)
    true = [c for c in cols if c.startswith(("strong_", "steady_"))]
    return len(true) / max(1, len([c for c in sel]))


CONFIGS = [
    ("importance_legacy", "importance", "legacy"),
    ("importance_dispatched", "importance", "dispatched"),
    ("stability_legacy", "stability", "legacy"),
    ("stability_dispatched", "stability", "dispatched"),
]
# pairwise compares stability vs importance matched on importance_agg (both legacy & dispatched).
BEDS = ["steady_mid_among_noise", "many_steady", "few_steady_lots_noise", "balanced_signal", "weak_steady"]


def main():
    rows = []
    for bed in BEDS:
        for seed in SEEDS:
            X, y = make_bed(bed, seed)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
            for label, rule, iagg in CONFIGS:
                cols = fit_select(Xtr, ytr, rule, iagg, seed)
                auc = holdout_auc(Xtr, Xte, ytr, yte, cols, seed)
                rows.append((bed, seed, label, round(auc, 4), len(cols), round(true_recall(cols), 3)))
                print(f"{bed:24s} s{seed} {label:22s} auc={auc:.4f} n={len(cols):2d} purity={true_recall(cols):.2f}", flush=True)
    df = pd.DataFrame(rows, columns=["bed", "seed", "config", "auc", "nsel", "purity"])
    print("\n=== mean honest-holdout AUC per config ===")
    print(df.groupby("config")["auc"].mean().round(4).to_string())
    print("\n=== per-bed mean AUC ===")
    print(df.pivot_table(index="bed", columns="config", values="auc", aggfunc="mean").round(4).to_string())
    # Pairwise: stability vs importance (matched on importance_agg), per (bed,seed).
    piv = df.pivot_table(index=["bed", "seed"], columns="config", values="auc")
    for iagg in ("legacy", "dispatched"):
        a, b = f"stability_{iagg}", f"importance_{iagg}"
        d = piv[a] - piv[b]
        wins = int((d > 1e-4).sum()); losses = int((d < -1e-4).sum()); ties = int((d.abs() <= 1e-4).sum())
        print(f"\nstability vs importance ({iagg}): mean delta={d.mean():+.4f}  wins={wins} losses={losses} ties={ties} / {len(d)}")
    return df


if __name__ == "__main__":
    main()
