"""Round-3 H3-1 bench: does feature-engineering integration close the HybridSelector's ~0.05 AUC gap to mrmr_fe?

Compares hybrid use_fe=True (MRMR engineers + shares engineered columns to all members) vs use_fe=False (the
current shipped behaviour) vs the mrmr_fe standalone baseline, on make_dataset (3 seeds). Measures downstream
honest-holdout AUC (LightGBM/Logistic/kNN), base recall, engineered-feature count, fit time. The thesis (H3-1):
FE-on should lift auc_mean from ~0.786 toward mrmr_fe's 0.835 while KEEPING the hybrid's recall championship.
KILL if FE-on auc_mean does not clear ~0.81, or base_recall drops below 0.90.
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


# Isolates two levers vs the current shipped hybrid (nofe_gini): the NOISE FIX (gini->permutation boruta driver)
# and the FE integration. mrmr_fe is the standalone ceiling.
def build(name):
    if name == "hybrid_nofe_gini":
        return HybridSelector(vote=1, use_fe=False, boruta_driver="gini", name=name)
    if name == "hybrid_nofe_perm":
        return HybridSelector(vote=1, use_fe=False, boruta_driver="permutation", name=name)
    if name == "hybrid_fe_perm":
        return HybridSelector(vote=1, use_fe=True, boruta_driver="permutation", name=name)
    if name == "mrmr_fe":
        return S.MRMRSel(fe=True)


def main():
    rows = []
    for sd in SEEDS:
        X, y, t = make_dataset(n_samples=5000, seed=sd)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=sd, stratify=y)
        base, noise = set(t["base"]), set(t["noise"])
        for name in ("hybrid_nofe_gini", "hybrid_nofe_perm", "hybrid_fe_perm", "mrmr_fe"):
            t0 = time.time()
            sel = build(name); sel.fit(Xtr, ytr); dt = time.time() - t0
            Ztr, Zte = sel.transform(Xtr), sel.transform(Xte)
            raw = [c for c in getattr(sel, "raw_selected_", []) if c in X.columns]
            a = downstream(Ztr, Zte, ytr, yte)
            am = round(float(np.nanmean(list(a.values()))), 4)
            row = dict(seed=sd, strategy=name, n=int(Ztr.shape[1]), n_eng=int(getattr(sel, "n_engineered_", 0)),
                       base=len(set(raw) & base), base_recall=round(len(set(raw) & base) / len(base), 3),
                       noise=len(set(raw) & noise), fit_s=round(dt, 1), auc=a, auc_mean=am)
            rows.append(row)
            print(f"sd{sd} {name:16s} n={row['n']:2d} eng={row['n_eng']:2d} base={row['base']}/{len(base)} "
                  f"rec={row['base_recall']} noise={row['noise']} {row['fit_s']:5.1f}s mean={am} auc={a}", flush=True)
    df = pd.DataFrame(rows)
    print("\n=== mean over seeds ===")
    print(df.groupby("strategy").agg(auc_mean=("auc_mean", "mean"), base_recall=("base_recall", "mean"),
                                     noise=("noise", "mean"), n=("n", "mean"), n_eng=("n_eng", "mean"),
                                     lgbm=("auc", lambda s: round(np.mean([a["lgbm"] for a in s]), 4)),
                                     logit=("auc", lambda s: round(np.mean([a["logit"] for a in s]), 4)),
                                     knn=("auc", lambda s: round(np.mean([a["knn"] for a in s]), 4)),
                                     fit_s=("fit_s", "mean")).round(4).to_string())


if __name__ == "__main__":
    main()
