"""ShapProxiedFS levers -- MEASURED (round-3 S3-2/3/4 + round-2 R2s-1/2/4/5 parsimony-band family).

Discriminating tests (not deferrals):
  parsimony_tol GRID {0.02,0.01,0.005,0.002}: the band ideas (R2s-1 split-var, R2s-4 abs-floor, R2s-5 skill-norm,
     S3-2) all interpolate a per-dataset tol BETWEEN fixed values -- so if NO fixed tol on this grid beats the 0.02
     default by > cross-seed noise, an adaptive band cannot either (the grid bounds the band's achievable gain).
  n_reval 2 vs 5 (S3-4): does a stabler revalidation winner recover more?
  K-split vote (R2s-2): fit ShapProxiedFS on 3 distinct 80% subsamples, keep features chosen in >=2 -> AUC +
     selection stability vs single fit (variance reduction).
  prefilter diagnostic (S3-3): are the interaction operands present in the working set after prefilter?
Measures downstream honest-holdout AUC, base recall, noise, interaction-operand recovery.
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
from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

SEEDS = [0, 1, 2]
BASEKW = dict(classification=True, n_splits=3, top_n=20, min_features=8, prefilter_n_estimators=60,
              oof_shap_n_estimators=60, revalidation_n_estimators=60, trust_guard=True,
              trust_guard_n_estimators=20, cluster_features="auto", within_cluster_refine=True,
              random_state=0, verbose=False)


def mk(p, **over):
    kw = dict(BASEKW); kw["prefilter_top"] = min(40, p); kw.update(over)
    return ShapProxiedFS(**kw)


def downstream(Xtr, Xte, ytr, yte, cols):
    if not cols:
        return {"lgbm": float("nan"), "logit": float("nan"), "knn": float("nan")}
    o = {}
    o["lgbm"] = roc_auc_score(yte, lgb.LGBMClassifier(n_estimators=300, verbose=-1).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    o["logit"] = roc_auc_score(yte, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    o["knn"] = roc_auc_score(yte, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    return {k: round(float(v), 4) for k, v in o.items()}


def run(name, sel, Xtr, Xte, ytr, yte, base, noise, oper):
    t0 = time.time(); sel.fit(Xtr, ytr); dt = time.time() - t0
    cols = [c for c in sel.selected_features_ if c in Xtr.columns]
    a = downstream(Xtr, Xte, ytr, yte, cols); am = round(float(np.nanmean(list(a.values()))), 4)
    return dict(variant=name, n=len(cols), base=len(set(cols) & base), base_recall=round(len(set(cols) & base) / len(base), 3),
                noise=len(set(cols) & noise), oper=len(set(cols) & oper), fit_s=round(dt, 1), auc_mean=am)


def main():
    rows = []
    for sd in SEEDS:
        X, y, t = make_dataset(n_samples=5000, seed=sd)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=sd, stratify=y)
        base, noise, oper = set(t["base"]), set(t["noise"]), set(t.get("interaction_operands", []))
        p = X.shape[1]
        # parsimony_tol grid (R2s-1/4/5 + S3-2)
        for tol in (0.02, 0.01, 0.005, 0.002):
            r = run(f"tol_{tol}", mk(p, parsimony_tol=tol, n_revalidation_models=2), Xtr, Xte, ytr, yte, base, noise, oper)
            rows.append(r); print(f"sd{sd} {r['variant']:14s} n={r['n']:2d} base={r['base']}/{len(base)} oper={r['oper']}/{len(oper)} noise={r['noise']} {r['fit_s']:5.1f}s mean={r['auc_mean']}", flush=True)
        # n_reval 5 (S3-4)
        r = run("nreval_5", mk(p, parsimony_tol=0.02, n_revalidation_models=5), Xtr, Xte, ytr, yte, base, noise, oper)
        rows.append(r); print(f"sd{sd} {r['variant']:14s} n={r['n']:2d} base={r['base']}/{len(base)} oper={r['oper']}/{len(oper)} noise={r['noise']} {r['fit_s']:5.1f}s mean={r['auc_mean']}", flush=True)
        # prefilter diagnostic (S3-3): are interaction operands in the working set?
        s = mk(p, parsimony_tol=0.02, n_revalidation_models=2); s.fit(Xtr, ytr)
        rep = getattr(s, "shap_proxy_report_", {}) or {}
        pf = rep.get("prefilter", {})
        work = set()
        if isinstance(pf, dict):
            for k in ("working_cols", "kept", "kept_features", "survivors", "stage1_features"):
                v = pf.get(k)
                if isinstance(v, (list, tuple, set)):
                    work = set(v); break
        print(f"sd{sd} S3-3 prefilter: operands_in_working={sorted(oper & work) if work else 'report-key-absent (prefilter keys=' + str(list(pf.keys()) if isinstance(pf, dict) else type(pf).__name__) + ')'} (oper={sorted(oper)})", flush=True)
        # K-split vote (R2s-2): 3 subsample fits, keep features chosen in >=2
        from collections import Counter
        cnt = Counter()
        for b in range(3):
            rng = np.random.default_rng(100 * sd + b)
            idx = rng.choice(len(Xtr), int(0.8 * len(Xtr)), replace=False)
            sb = mk(p, parsimony_tol=0.02, n_revalidation_models=2)
            sb.fit(Xtr.iloc[idx].reset_index(drop=True), ytr.iloc[idx].reset_index(drop=True))
            cnt.update([c for c in sb.selected_features_ if c in Xtr.columns])
        ksel = [c for c in X.columns if cnt.get(c, 0) >= 2]
        a = downstream(Xtr, Xte, ytr, yte, ksel); am = round(float(np.nanmean(list(a.values()))), 4)
        rows.append(dict(variant="Ksplit_vote", n=len(ksel), base=len(set(ksel) & base), base_recall=round(len(set(ksel) & base) / len(base), 3),
                         noise=len(set(ksel) & noise), oper=len(set(ksel) & oper), fit_s=0.0, auc_mean=am))
        print(f"sd{sd} Ksplit_vote    n={len(ksel):2d} base={len(set(ksel)&base)}/{len(base)} noise={len(set(ksel)&noise)} mean={am}", flush=True)
    df = pd.DataFrame(rows)
    print("\n=== mean over seeds ===")
    g = df.groupby("variant").agg(auc_mean=("auc_mean", "mean"), base_recall=("base_recall", "mean"),
                                  oper=("oper", "mean"), noise=("noise", "mean"), n=("n", "mean"), fit_s=("fit_s", "mean")).round(4)
    print(g.to_string())


if __name__ == "__main__":
    main()
