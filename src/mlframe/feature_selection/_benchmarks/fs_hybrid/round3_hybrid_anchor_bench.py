"""Round-3 fix #1: does anchor_fe make the FE-hybrid >= its own FE substrate (mrmr_fe), removing the measured
"hybrid trails mrmr_fe" defect? hybrid(anchor) vs hybrid(no-anchor, the prior combine) vs mrmr_fe, make_dataset 3 seeds.
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
import fs_selectors as S

SEEDS = [0, 1, 2]


def downstream(Ztr, Zte, ytr, yte):
    o = {}
    o["lgbm"] = roc_auc_score(yte, lgb.LGBMClassifier(n_estimators=300, verbose=-1).fit(Ztr, ytr).predict_proba(Zte)[:, 1])
    o["logit"] = roc_auc_score(yte, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Ztr, ytr).predict_proba(Zte)[:, 1])
    o["knn"] = roc_auc_score(yte, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)).fit(Ztr, ytr).predict_proba(Zte)[:, 1])
    return {k: round(float(v), 4) for k, v in o.items()}


def build(name):
    if name == "hybrid_anchor":   return HybridSelector(vote=1, use_fe=True, anchor_fe=True)
    if name == "hybrid_noanchor": return HybridSelector(vote=1, use_fe=True, anchor_fe=False)
    if name == "mrmr_fe":         return S.MRMRSel(fe=True)


def main():
    rows = []
    for sd in SEEDS:
        X, y, t = make_dataset(n_samples=5000, seed=sd)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=sd, stratify=y)
        base = set(t["base"])
        for name in ("mrmr_fe", "hybrid_noanchor", "hybrid_anchor"):
            t0 = time.time(); sel = build(name); sel.fit(Xtr, ytr); dt = time.time() - t0
            Ztr, Zte = sel.transform(Xtr), sel.transform(Xte)
            raw = [c for c in getattr(sel, "raw_selected_", []) if c in X.columns]
            a = downstream(Ztr, Zte, ytr, yte); am = round(float(np.nanmean(list(a.values()))), 4)
            rows.append(dict(seed=sd, strategy=name, n=int(Ztr.shape[1]), n_eng=int(getattr(sel, "n_engineered_", 0)),
                             base_recall=round(len(set(raw) & base) / len(base), 3), fit_s=round(dt, 1), auc_mean=am))
            print(f"sd{sd} {name:16s} n={rows[-1]['n']:2d} eng={rows[-1]['n_eng']:2d} rec={rows[-1]['base_recall']} {dt:5.1f}s mean={am} auc={a}", flush=True)
    df = pd.DataFrame(rows)
    print("\n=== mean over seeds ===")
    g = df.groupby("strategy").agg(auc_mean=("auc_mean", "mean"), base_recall=("base_recall", "mean"),
                                   n=("n", "mean"), n_eng=("n_eng", "mean"), fit_s=("fit_s", "mean")).round(4)
    m = g.loc["mrmr_fe", "auc_mean"]
    g["delta_vs_mrmr_fe"] = (g["auc_mean"] - m).round(4)
    print(g.to_string())


if __name__ == "__main__":
    main()
