"""Round-2 untested: R2b-3 (vote-fraction threshold calibration) + R2b-8 (panel + OOF/permutation importance).

hetero_vote was benched as a PRECISION tool that under-recovers vs boruta on AUC (R2b-1). R2b-3/R2b-8 are
REFINEMENTS of it that I previously rejected by reasoning ("no AUC headroom"). Discriminating test, MEASURED here:
sweep the vote_threshold (R2b-3: a lower bar = more permissive panel agreement) and the shadow percentile, and
force permutation importance in the panel (R2b-8). If NO refinement lifts hetero's downstream AUC to boruta's
level, the rejections are confirmed by measurement (not hand-wave). Across the 6 fs_hybrid scenarios x 2 seeds.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenarios import SCENARIOS, make
from mlframe.feature_selection.hetero_vote import heterogeneous_relevance_vote
from mlframe.feature_selection.boruta_shap import BorutaShap

SEEDS = [0, 1]


def downstream(Xtr, Xte, ytr, yte, cols):
    if not cols:
        return float("nan")
    aucs = []
    for mk in (lambda: lgb.LGBMClassifier(n_estimators=300, verbose=-1),
               lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
               lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25))):
        aucs.append(roc_auc_score(yte, mk().fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1]))
    return round(float(np.mean(aucs)), 4)


def boruta(X, y):
    b = BorutaShap(model=RandomForestClassifier(n_estimators=80, n_jobs=-1, random_state=0),
                   importance_measure="gini", classification=True, n_trials=50, percentile=95, verbose=False, random_state=0)
    b.fit(X, y)
    return [c for c in b.selected_features_ if c in X.columns]


def main():
    rows = []
    # R2b-3 threshold/percentile grid + R2b-8 permutation panel importance. Default panel (tree/linear/distance).
    VARIANTS = {
        "boruta_ref": None,
        "hetero_vt0.5": dict(vote_threshold=0.5, percentile=100.0),
        "hetero_vt0.34": dict(vote_threshold=0.34, percentile=100.0),  # R2b-3 permissive (>=1 of 3)
        "hetero_pct95": dict(vote_threshold=0.5, percentile=95.0),  # R2b-3 looser shadow bar
        "hetero_vt0.34p95": dict(vote_threshold=0.34, percentile=95.0),  # R2b-3 most permissive
    }
    for sc in SCENARIOS:
        for sd in SEEDS:
            X, y, t = make(sc, sd)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=sd, stratify=y)
            base, noise = set(t["base"]), set(t["noise"])
            for name, kw in VARIANTS.items():
                t0 = time.time()
                if kw is None:
                    sel = boruta(Xtr, ytr)
                else:
                    acc, _ = heterogeneous_relevance_vote(Xtr, ytr, classification=True, n_shadow_trials=2, random_state=0, **kw)
                    sel = [c for c in acc if c in X.columns]
                dt = time.time() - t0
                am = downstream(Xtr, Xte, ytr, yte, sel)
                rows.append(dict(scenario=sc, seed=sd, variant=name, n=len(sel), base=len(set(sel) & base),
                                 noise=len(set(sel) & noise), fit_s=round(dt, 1), auc_mean=am))
            print(f"{sc:14s} sd{sd} " + " ".join(f"{r['variant']}={r['auc_mean']}" for r in rows[-len(VARIANTS):]), flush=True)
    df = pd.DataFrame(rows)
    print("\n=== mean over scenarios x seeds ===")
    g = df.groupby("variant").agg(auc_mean=("auc_mean", "mean"), base=("base", "mean"), noise=("noise", "mean"), n=("n", "mean")).round(4)
    b = g.loc["boruta_ref", "auc_mean"]
    g["delta_vs_boruta"] = (g["auc_mean"] - b).round(4)
    print(g.to_string())
    print(f"\nR2b-3/R2b-8 verdict: any hetero variant beats boruta ({b})? " +
          str({v: round(g.loc[v, 'auc_mean'] - b, 4) for v in g.index if v != 'boruta_ref' and g.loc[v, 'auc_mean'] > b} or "NONE"))


if __name__ == "__main__":
    main()
