"""Round-2 ShapProxiedFS parsimony-cluster bench (R2s-3 gates R2s-1/4/5).

4 configs x 3 over-pruning-relevant scenarios x 2 seeds. Answers: (R2s-3) is within_cluster_refine net-positive,
or does refine=False + wider revalidation (top_n 20->40) win? (R2s-1/4) does a TIGHTER parsimony_tol (0.005)
with refine ON beat refine=False (i.e. is the over-pruning just a too-loose tol)? Downstream honest-holdout AUC.
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
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenarios import make
from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

SCEN = ["base", "manyredundant", "monotone"]
SEEDS = [0, 1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results")
CONFIGS = {
    "refineOFF_top20": dict(within_cluster_refine=False, top_n=20),
    "refineOFF_top40": dict(within_cluster_refine=False, top_n=40),
    "refineON_tol02": dict(within_cluster_refine=True, parsimony_tol=0.02, top_n=20),
    "refineON_tol005": dict(within_cluster_refine=True, parsimony_tol=0.005, top_n=20),
}
BASE = dict(classification=True, n_splits=3, min_features=1, prefilter_top=40, prefilter_n_estimators=50,
            oof_shap_n_estimators=50, revalidation_n_estimators=50, n_revalidation_models=2,
            trust_guard=True, trust_guard_n_estimators=15, cluster_features="auto", random_state=0, verbose=False)


def downstream(Xtr, Xte, ytr, yte, cols):
    if not cols:
        return {"lgbm": float("nan"), "logit": float("nan"), "knn": float("nan")}
    o = {}
    o["lgbm"] = roc_auc_score(yte, lgb.LGBMClassifier(n_estimators=300, verbose=-1).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    o["logit"] = roc_auc_score(yte, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    o["knn"] = roc_auc_score(yte, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    return {k: round(float(v), 4) for k, v in o.items()}


def main():
    rows = []
    for sc in SCEN:
        for sd in SEEDS:
            X, y, t = make(sc, sd)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=sd, stratify=y)
            base = set(t["base"])
            for name, kw in CONFIGS.items():
                try:
                    t0 = time.time(); s = ShapProxiedFS(**BASE, **kw); s.fit(Xtr, ytr); dt = time.time() - t0
                    cols = [c for c in s.selected_features_ if c in X.columns]
                    a = downstream(Xtr, Xte, ytr, yte, cols)
                    am = round(float(np.nanmean([v for v in a.values()])), 4)
                    row = dict(scenario=sc, seed=sd, config=name, n=len(cols), base=len(set(cols) & base), fit_s=round(dt, 1), auc=a, auc_mean=am)
                except Exception as e:
                    row = dict(scenario=sc, seed=sd, config=name, error=f"{type(e).__name__}: {e}")
                rows.append(row)
                print(f"{sc:14s} sd{sd} {name:16s} " + (row.get("error") or f"n={row['n']:2d} base={row['base']}/{len(base)} {row['fit_s']:5.1f}s auc={row['auc']} mean={row['auc_mean']}"), flush=True)
    with open(os.path.join(OUT, "round2_shap.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    df = pd.DataFrame([r for r in rows if not r.get("error")])
    print("\n=== mean over scenarios x seeds ===")
    print(df.groupby("config").agg(auc_mean=("auc_mean", "mean"), lgbm=("auc", lambda s: np.mean([a["lgbm"] for a in s])),
                                   n=("n", "mean"), base=("base", "mean"), fit_s=("fit_s", "mean")).round(4).to_string())
    wins = {}
    for (sc, sd), g in df.groupby(["scenario", "seed"]):
        b = g.loc[g["auc_mean"].idxmax(), "config"]; wins[b] = wins.get(b, 0) + 1
    print("win counts (best auc_mean):", wins)


if __name__ == "__main__":
    main()
