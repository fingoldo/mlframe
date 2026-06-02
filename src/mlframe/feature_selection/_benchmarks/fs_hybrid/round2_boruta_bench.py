"""Round-2 BorutaShap bench (R2b-1, R2b-2): hetero_vote variants vs single-fit Boruta across scenarios.

Answers: (R2b-1) does cross-model hetero_vote match/beat single-fit Boruta on downstream honest-holdout AUC
while dropping the noise leak? (R2b-2) does cheap n_shadow_trials=2 match the default-ish 5? Writes a table.
"""
from __future__ import annotations
import os, sys, time, json
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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results")
os.makedirs(OUT, exist_ok=True)


def downstream(Xtr, Xte, ytr, yte, cols):
    if not cols:
        return {"lgbm": float("nan"), "logit": float("nan"), "knn": float("nan")}
    o = {}
    o["lgbm"] = roc_auc_score(yte, lgb.LGBMClassifier(n_estimators=300, verbose=-1).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    o["logit"] = roc_auc_score(yte, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    o["knn"] = roc_auc_score(yte, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    return {k: round(float(v), 4) for k, v in o.items()}


def boruta_single(X, y):
    b = BorutaShap(model=RandomForestClassifier(n_estimators=80, n_jobs=-1, random_state=0),
                   importance_measure="gini", classification=True, n_trials=60, percentile=95, verbose=False, random_state=0)
    b.fit(X, y)
    return [c for c in b.selected_features_ if c in X.columns]


def main():
    rows = []
    for sc in SCENARIOS:
        for sd in SEEDS:
            X, y, t = make(sc, sd)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=sd, stratify=y)
            noise = set(t["noise"]); base = set(t["base"])
            methods = {}
            t0 = time.time(); methods["boruta"] = (boruta_single(Xtr, ytr), time.time() - t0)
            for nst in (2, 5):
                t0 = time.time(); acc, _ = heterogeneous_relevance_vote(Xtr, ytr, classification=True, n_shadow_trials=nst, vote_threshold=0.5, random_state=0)
                methods[f"hetero{nst}"] = (acc, time.time() - t0)
            for name, (cols, dt) in methods.items():
                cols = [c for c in cols if c in X.columns]
                a = downstream(Xtr, Xte, ytr, yte, cols)
                am = round(float(np.nanmean([v for v in a.values()])), 4)
                row = dict(scenario=sc, seed=sd, method=name, n=len(cols), base=len(set(cols) & base),
                           noise=len(set(cols) & noise), fit_s=round(dt, 1), auc=a, auc_mean=am)
                rows.append(row)
                print(f"{sc:14s} sd{sd} {name:9s} n={row['n']:2d} base={row['base']}/{len(base)} noise={row['noise']:2d} {dt:5.1f}s auc={a} mean={am}", flush=True)
    with open(os.path.join(OUT, "round2_boruta.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # summary: mean auc_mean + mean noise + win count (best auc_mean per scenario,seed) per method
    df = pd.DataFrame(rows)
    print("\n=== mean over scenarios x seeds ===")
    print(df.groupby("method").agg(auc_mean=("auc_mean", "mean"), lgbm=("auc", lambda s: np.mean([a["lgbm"] for a in s])),
                                   noise=("noise", "mean"), n=("n", "mean"), fit_s=("fit_s", "mean")).round(4).to_string())
    wins = {}
    for (sc, sd), g in df.groupby(["scenario", "seed"]):
        b = g.loc[g["auc_mean"].idxmax(), "method"]; wins[b] = wins.get(b, 0) + 1
    print("win counts (best auc_mean):", wins)


if __name__ == "__main__":
    main()
