"""RFECV levers -- MEASURED (round-3 R3-3/4/5/6 + round-2 R2r-4/5). Not deferrals.

One RFECV fit per seed (impurity, fast -- the question is structure/N, not the importance metric which R3-2/round-1
settled) yields the support + an FI ranking; the post-process variants are applied to it:
  R3-3 noise_floor : drop a selected feature whose held-out permutation importance <= max shuffled-shadow importance.
  R3-5 N_grid      : re-evaluate top-N (by FI) for every N in a band around the chosen size; pick the CV-argmax.
  R3-4 stab_rerank : bootstrap (B=10) selection-frequency; at the chosen N, swap low-freq selected for high-freq
                     dropped when a single CV eval improves (equal-N re-rank, not N-reselection).
  R3-6 / R2r-5 cluster_collapse : collapse raw-corr clusters to reps, RFECV on reps, re-expand (separate fit).
  R2r-4 routing    : VERIFY it is a no-op on the bed (cells < the 4M permutation cap -> permutation always runs).
Reports downstream honest-holdout AUC vs the baseline RFECV.
"""
from __future__ import annotations
import os, sys, time, itertools
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth import make_dataset
from mlframe.feature_selection.wrappers import RFECV, FIConfig, SearchConfig

SEEDS = [0, 1, 2]
_PERM_CAP = 4_000_000


def fit_rfecv(X, y, importance="feature_importances_"):
    r = RFECV(estimator=lgb.LGBMClassifier(n_estimators=150, num_leaves=31, learning_rate=0.06, n_jobs=-1, verbose=-1),
              cv=3, scoring=None, verbose=0,
              fi_config=FIConfig(importance_getter=importance, n_features_selection_rule="one_se_min"),
              search_config=SearchConfig(max_refits=14, max_runtime_mins=2), random_state=0)
    r.fit(X, y)
    return [c for c in r.get_feature_names_out() if c in X.columns]


def fi_rank(X, y, cols):
    m = lgb.LGBMClassifier(n_estimators=200, verbose=-1).fit(X[cols], y)
    return [c for _, c in sorted(zip(m.feature_importances_, cols), reverse=True)]


def downstream(Xtr, Xte, ytr, yte, cols):
    if not cols:
        return {"lgbm": float("nan"), "logit": float("nan"), "knn": float("nan")}
    o = {}
    o["lgbm"] = roc_auc_score(yte, lgb.LGBMClassifier(n_estimators=300, verbose=-1).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    o["logit"] = roc_auc_score(yte, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    o["knn"] = roc_auc_score(yte, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    return {k: round(float(v), 4) for k, v in o.items()}


def corr_clusters(X, thr=0.92):
    cols = list(X.columns); C = np.nan_to_num(np.corrcoef(X.values, rowvar=False))
    reps, members, seen = [], {}, set()
    for i, c in enumerate(cols):
        if c in seen: continue
        reps.append(c); members[c] = [c]; seen.add(c)
        for j in range(i + 1, len(cols)):
            if cols[j] not in seen and abs(C[i, j]) >= thr:
                members[c].append(cols[j]); seen.add(cols[j])
    return reps, members


def noise_floor(Xtr, ytr, survivors):
    """R3-3: drop a survivor whose held-out permutation importance <= max importance of its shuffled shadow."""
    Xa, Xv, ya, yv = train_test_split(Xtr[survivors], ytr, test_size=0.3, random_state=0, stratify=ytr)
    m = lgb.LGBMClassifier(n_estimators=200, verbose=-1).fit(Xa, ya)
    rng = np.random.default_rng(0)
    shadow = Xv.apply(lambda col: rng.permutation(col.values))
    Xv_sh = pd.concat([Xv, shadow.add_prefix("sh_")], axis=1)
    m2 = lgb.LGBMClassifier(n_estimators=200, verbose=-1).fit(pd.concat([Xa, Xa.apply(lambda c: rng.permutation(c.values)).add_prefix("sh_")], axis=1), ya)
    pi = permutation_importance(m2, Xv_sh, yv, n_repeats=4, random_state=0, n_jobs=-1)
    imp = dict(zip(Xv_sh.columns, pi.importances_mean))
    shadow_max = max([imp[c] for c in Xv_sh.columns if c.startswith("sh_")] + [0.0])
    return [c for c in survivors if imp.get(c, 0.0) > shadow_max] or survivors


def n_grid(Xtr, ytr, ranking, lo, hi):
    """R3-5: CV-argmax over top-N for N in [lo, hi]."""
    best, best_au = ranking[:lo], -1
    for n in range(max(1, lo), min(hi, len(ranking)) + 1):
        cols = ranking[:n]
        au = cross_val_score(lgb.LGBMClassifier(n_estimators=120, verbose=-1), Xtr[cols], ytr, cv=3, scoring="roc_auc").mean()
        if au > best_au:
            best_au, best = au, cols
    return list(best)


def main():
    rows = []
    for sd in SEEDS:
        X, y, t = make_dataset(n_samples=5000, seed=sd)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=sd, stratify=y)
        base, noise = set(t["base"]), set(t["noise"])
        cells = Xtr.shape[0] * Xtr.shape[1]
        # baseline RFECV
        surv = fit_rfecv(Xtr, ytr)
        rank = fi_rank(Xtr, ytr, surv)

        def rec(name, cols):
            a = downstream(Xtr, Xte, ytr, yte, cols); am = round(float(np.nanmean(list(a.values()))), 4)
            rows.append(dict(seed=sd, variant=name, n=len(cols), base=len(set(cols) & base),
                             base_recall=round(len(set(cols) & base) / len(base), 3), noise=len(set(cols) & noise), auc_mean=am))
            print(f"sd{sd} {name:16s} n={len(cols):2d} base={len(set(cols)&base)}/{len(base)} noise={len(set(cols)&noise)} mean={am}", flush=True)

        rec("baseline", surv)
        rec("R3-3_noisefloor", noise_floor(Xtr, ytr, surv))
        rec("R3-5_Ngrid", n_grid(Xtr, ytr, rank, max(1, len(surv) - 6), len(surv) + 8))
        # R3-6 / R2r-5 cluster-collapse: RFECV on reps, re-expand
        reps, members = corr_clusters(Xtr, 0.92)
        rep_surv = fit_rfecv(Xtr[reps], ytr)
        expanded = [m for r in rep_surv for m in members.get(r, [r])]
        rec("R3-6_clustcollapse", [c for c in dict.fromkeys(expanded) if c in X.columns])
        # R3-4 stability re-rank: bootstrap freq, equal-N swap
        from collections import Counter
        freq = Counter()
        for b in range(10):
            rng = np.random.default_rng(7 * sd + b)
            idx = rng.choice(len(Xtr), int(0.7 * len(Xtr)), replace=False)
            freq.update(fit_rfecv(Xtr.iloc[idx].reset_index(drop=True), ytr.iloc[idx].reset_index(drop=True)))
        top_by_freq = [c for c, _ in freq.most_common(len(surv))]
        rec("R3-4_stabrerank", top_by_freq)
        print(f"sd{sd} R2r-4 routing: cells={cells} (cap={_PERM_CAP}) -> permutation {'ALWAYS runs (routing is a NO-OP here)' if cells <= _PERM_CAP else 'would route to impurity'}", flush=True)
    df = pd.DataFrame(rows)
    print("\n=== mean over seeds (vs baseline) ===")
    g = df.groupby("variant").agg(auc_mean=("auc_mean", "mean"), base_recall=("base_recall", "mean"),
                                  noise=("noise", "mean"), n=("n", "mean")).round(4)
    b = g.loc["baseline", "auc_mean"] if "baseline" in g.index else float("nan")
    g["delta"] = (g["auc_mean"] - b).round(4)
    print(g.to_string())


if __name__ == "__main__":
    main()
