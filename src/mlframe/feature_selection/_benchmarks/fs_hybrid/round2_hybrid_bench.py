"""Focused HybridSelector combine-rule bench: does the FI-credibility guard cut the vote=1 noise leak while
keeping the recall champion's base recovery? Runs the hybrid variants on make_dataset (the big-benchmark dataset)
across 3 seeds and reports base_recall / noise / downstream AUC, to compare against the big-benchmark baselines
(rfecv_lgbm_perm auc_mean 0.792 base_recall 0.875; boruta 0.772/0.875; hybrid no-guard 0.786/0.958).
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth import make_dataset
from hybrid_selector import HybridSelector

SEEDS = [0, 1, 2]
VARIANTS = {
    "vote1_guard": dict(vote=1, fi_guard=True),
    "vote1_noguard": dict(vote=1, fi_guard=False),
    "vote1_guard_expand": dict(vote=1, fi_guard=True, expand_clusters=True),
    "vote2": dict(vote=2, fi_guard=True),
}


def downstream(Xtr, Xte, ytr, yte, cols):
    o = {}
    o["lgbm"] = roc_auc_score(yte, lgb.LGBMClassifier(n_estimators=300, verbose=-1).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    o["logit"] = roc_auc_score(yte, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    o["knn"] = roc_auc_score(yte, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    return {k: round(float(v), 4) for k, v in o.items()}


def main():
    rows = []
    for sd in SEEDS:
        X, y, truth = make_dataset(n_samples=5000, seed=sd)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=sd, stratify=y)
        base, noise = set(truth["base"]), set(truth["noise"])
        for name, kw in VARIANTS.items():
            t0 = time.time(); h = HybridSelector(random_state=0, **kw); h.fit(Xtr, ytr); dt = time.time() - t0
            sel = h.raw_selected_
            a = downstream(Xtr, Xte, ytr, yte, sel)
            am = round(float(np.mean(list(a.values()))), 4)
            row = dict(seed=sd, variant=name, n=len(sel), base=len(set(sel) & base),
                       base_recall=round(len(set(sel) & base) / len(base), 3),
                       noise=len(set(sel) & noise), fit_s=round(dt, 1), auc=a, auc_mean=am)
            rows.append(row)
            print(f"sd{sd} {name:20s} n={row['n']:2d} base={row['base']}/{len(base)} rec={row['base_recall']} "
                  f"noise={row['noise']} {dt:5.1f}s auc={a} mean={am}", flush=True)
    df = pd.DataFrame(rows)
    print("\n=== mean over seeds ===")
    print(df.groupby("variant").agg(auc_mean=("auc_mean", "mean"), base_recall=("base_recall", "mean"),
                                    noise=("noise", "mean"), n=("n", "mean"),
                                    lgbm=("auc", lambda s: round(np.mean([a["lgbm"] for a in s]), 4)),
                                    logit=("auc", lambda s: round(np.mean([a["logit"] for a in s]), 4)),
                                    knn=("auc", lambda s: round(np.mean([a["knn"] for a in s]), 4)),
                                    fit_s=("fit_s", "mean")).round(4).to_string())
    print("\nbaselines from big benchmark: rfecv_lgbm_perm auc_mean 0.7924 rec 0.875 noise 0.33 | boruta 0.7722/0.875/1.33")


if __name__ == "__main__":
    main()
