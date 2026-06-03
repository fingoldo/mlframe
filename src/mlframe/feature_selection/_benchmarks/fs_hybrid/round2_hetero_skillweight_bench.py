"""Round-2 BorutaShap R2b-4: CV-skill-weighted hetero_vote vs equal-weight vs single-fit Boruta.

hetero_vote's measured deficit (R2b-1) is RECALL: a panel member blind to a feature's functional form (linear on
monotone-nonlinear signal) casts a full-weight NO vote and sinks features the informative members confirm
(monotone sd1 collapsed hetero to base 3/8). R2b-4 weights each member's vote by its above-chance CV skill so a
structurally-blind member is downweighted. This benches whether skill-weighting recovers the lost recall / AUC
across the full bed (the monotone + weakmix cells are the discriminating ones). If it closes the gap to Boruta it
becomes the hetero default; if not, hetero stays the equal-weight precision tool and Boruta stays the AUC default.
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
            noise, base = set(t["noise"]), set(t["base"])
            methods, extra = {}, {}
            t0 = time.time(); methods["boruta"] = boruta_single(Xtr, ytr); extra["boruta"] = (time.time() - t0, "")
            t0 = time.time(); acc, _ = heterogeneous_relevance_vote(Xtr, ytr, classification=True, n_shadow_trials=2, weight_by_cv_skill=False, random_state=0); methods["hetero_eq"] = acc; extra["hetero_eq"] = (time.time() - t0, "")
            t0 = time.time(); acc, info = heterogeneous_relevance_vote(Xtr, ytr, classification=True, n_shadow_trials=2, weight_by_cv_skill=True, random_state=0); methods["hetero_skill"] = acc; extra["hetero_skill"] = (time.time() - t0, str(info.get("model_weights")))
            for name, cols in methods.items():
                dt, ex = extra[name]
                cols = [c for c in cols if c in X.columns]
                a = downstream(Xtr, Xte, ytr, yte, cols)
                am = round(float(np.nanmean(list(a.values()))), 4)
                row = dict(scenario=sc, seed=sd, method=name, n=len(cols), base=len(set(cols) & base),
                           noise=len(set(cols) & noise), fit_s=round(dt, 1), auc=a, auc_mean=am, weights=ex)
                rows.append(row)
                print(f"{sc:14s} sd{sd} {name:13s} n={row['n']:2d} base={row['base']}/{len(base)} noise={row['noise']:2d} {dt:5.1f}s mean={am} {ex}", flush=True)
    with open(os.path.join(OUT, "round2_hetero_skillweight.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    df = pd.DataFrame(rows)
    print("\n=== mean over scenarios x seeds ===")
    print(df.groupby("method").agg(auc_mean=("auc_mean", "mean"), base=("base", "mean"),
                                   noise=("noise", "mean"), n=("n", "mean"), fit_s=("fit_s", "mean")).round(4).to_string())
    wins = {}
    for (sc, sd), g in df.groupby(["scenario", "seed"]):
        b = g.loc[g["auc_mean"].idxmax(), "method"]; wins[b] = wins.get(b, 0) + 1
    print("win counts (best auc_mean):", wins)
    # the head-to-head that matters: does skill-weighting beat equal-weight hetero?
    he = df[df.method == "hetero_eq"].set_index(["scenario", "seed"]).auc_mean
    hs = df[df.method == "hetero_skill"].set_index(["scenario", "seed"]).auc_mean
    skill_better = int((hs > he + 1e-6).sum()); eq_better = int((he > hs + 1e-6).sum())
    print(f"R2b-4 skill vs equal hetero: skill better in {skill_better}/{len(he)} cells, equal better in {eq_better}; "
          f"mean skill {hs.mean():.4f} vs equal {he.mean():.4f} vs boruta {df[df.method=='boruta'].auc_mean.mean():.4f}")


if __name__ == "__main__":
    main()
