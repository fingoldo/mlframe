"""Round-3 S3-1 probe: does interaction_aware=True let ShapProxiedFS recover the pure-interaction operands
(inf_4*inf_5, etc.) that the additive SHAP proxy misses (recall stalls at 5/8)? Off vs on, on the interaction-
heavy cells, measuring interaction-operand recovery + base recall + downstream honest-holdout AUC. The wired path
(shap_proxied_fs.py:1500) is the additivity COALITION objective (base+sum Phi_ik), NOT the closed interaction-
ranking (S2/B3/R1); candidates still go through honest revalidation. KILL if operands still absent or AUC flat.
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
from scenarios import make
from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

CASES = [("make_dataset", 0), ("make_dataset", 1), ("xor2", 0), ("xor2", 1)]
# interaction_on16 = the default gate (phi.shape[1] <= 16) -> never fires on these wide cells (== off, confirmed).
# interaction_on60 raises the gate so the coalition path ACTUALLY engages (phi units <= 60). This is the real test.
CONFIGS = {"interaction_off": dict(interaction_aware=False),
           "interaction_on60": dict(interaction_aware=True, max_interaction_features=60)}


def load(name, seed):
    if name == "make_dataset":
        return make_dataset(n_samples=5000, seed=seed)
    return make(name, seed)


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
    for name, sd in CASES:
        X, y, t = load(name, sd)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=sd, stratify=y)
        base, oper = set(t["base"]), set(t.get("interaction_operands", []))
        for cname, kw in CONFIGS.items():
            t0 = time.time()
            s = ShapProxiedFS(classification=True, n_splits=3, top_n=20, min_features=8,
                              prefilter_top=min(40, X.shape[1]), prefilter_n_estimators=60, oof_shap_n_estimators=60,
                              revalidation_n_estimators=60, n_revalidation_models=2, trust_guard=True,
                              trust_guard_n_estimators=20, cluster_features="auto", within_cluster_refine=True,
                              parsimony_tol=0.005, random_state=0, verbose=False, **kw)
            try:
                s.fit(Xtr, ytr); dt = time.time() - t0
                sel = [c for c in s.selected_features_ if c in X.columns]
                a = downstream(Xtr, Xte, ytr, yte, sel)
                am = round(float(np.nanmean(list(a.values()))), 4)
                row = dict(case=f"{name}_sd{sd}", config=cname, n=len(sel),
                           base=len(set(sel) & base), oper=len(set(sel) & oper), n_oper=len(oper),
                           fit_s=round(dt, 1), auc=a, auc_mean=am)
            except Exception as e:
                row = dict(case=f"{name}_sd{sd}", config=cname, error=f"{type(e).__name__}: {e}")
            rows.append(row)
            print(f"{row['case']:16s} {cname:15s} " + (row.get("error") or
                  f"n={row['n']:2d} base={row['base']}/{len(base)} oper={row['oper']}/{row['n_oper']} {row['fit_s']:5.1f}s mean={row['auc_mean']} auc={row['auc']}"), flush=True)
    df = pd.DataFrame([r for r in rows if not r.get("error")])
    print("\n=== mean over cases ===")
    print(df.groupby("config").agg(auc_mean=("auc_mean", "mean"), base=("base", "mean"),
                                   oper=("oper", "mean"), n=("n", "mean"), fit_s=("fit_s", "mean")).round(4).to_string())


if __name__ == "__main__":
    main()
