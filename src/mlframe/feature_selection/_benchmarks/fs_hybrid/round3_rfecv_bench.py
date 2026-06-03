"""Round-3 RFECV levers: R3-2 (one_se rule audit) + R3-1 (post-selection survivor pairwise-product FE).

R3-2: the RFECVSel adapter forces n_features_selection_rule='one_se_min' (parsimony); does 'one_se_max' / 'argmax'
recover more true bases / higher downstream AUC? (cheap config flip).
R3-1: RFECV eliminates by MARGINAL importance so it cannot keep a pure-interaction operand alone, and has no FE ->
+0.04 below mrmr_fe. Take the RFECV survivors, form pairwise products/squares of the top-k survivors, and KEEP a
product only if adding it lifts 3-fold CV AUC (vs the survivor set) by > 1 SE. This builds the interaction term from
the small survivor pool (where both operands usually survive), gated by honest CV -- sidestepping the closed
interaction-RANKING failure. Measures whether the true inf_4*inf_5 gets formed + the downstream AUC lift.
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
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth import make_dataset
from mlframe.feature_selection.wrappers import RFECV, FIConfig, SearchConfig

SEEDS = [0, 1, 2]


def downstream(Xtr, Xte, ytr, yte, cols):
    if not cols:
        return {"lgbm": float("nan"), "logit": float("nan"), "knn": float("nan")}
    o = {}
    o["lgbm"] = roc_auc_score(yte, lgb.LGBMClassifier(n_estimators=300, verbose=-1).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    o["logit"] = roc_auc_score(yte, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    o["knn"] = roc_auc_score(yte, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    return {k: round(float(v), 4) for k, v in o.items()}


def fit_rfecv(X, y, rule):
    r = RFECV(estimator=lgb.LGBMClassifier(n_estimators=150, num_leaves=31, learning_rate=0.06, n_jobs=-1, verbose=-1),
              cv=3, scoring=None, verbose=0,
              fi_config=FIConfig(importance_getter="permutation", n_features_selection_rule=rule),
              search_config=SearchConfig(max_refits=18, max_runtime_mins=3), random_state=0)
    r.fit(X, y)
    return [c for c in r.get_feature_names_out() if c in X.columns]


def survivor_fe(Xtr, ytr, survivors, fi_rank, k=8):
    """R3-1: form pairwise products + squares of the top-k survivors; keep those whose addition lifts 3-fold CV AUC
    over the survivor set by > 1 SE. Returns (added_product_frames dict name->series builder, names)."""
    top = fi_rank[:k]
    base_scores = cross_val_score(lgb.LGBMClassifier(n_estimators=120, verbose=-1), Xtr[survivors], ytr, cv=3, scoring="roc_auc")
    base_mu, base_se = base_scores.mean(), base_scores.std() / np.sqrt(len(base_scores))
    products = {}
    for a, b in list(itertools.combinations(top, 2)) + [(c, c) for c in top]:
        name = f"prod__{a}__{b}"
        col = (Xtr[a] * Xtr[b]).rename(name)
        aug = pd.concat([Xtr[survivors], col], axis=1)
        sc = cross_val_score(lgb.LGBMClassifier(n_estimators=120, verbose=-1), aug, ytr, cv=3, scoring="roc_auc")
        if sc.mean() > base_mu + base_se:   # > 1 SE improvement
            products[name] = (a, b, round(sc.mean() - base_mu, 4))
    return products


def main():
    rows = []
    for sd in SEEDS:
        X, y, t = make_dataset(n_samples=5000, seed=sd)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=sd, stratify=y)
        base, noise = set(t["base"]), set(t["noise"])
        oper = set(t.get("interaction_operands", []))
        # R3-2: rule audit
        for rule in ("one_se_min", "one_se_max", "argmax"):
            t0 = time.time(); sel = fit_rfecv(Xtr, ytr, rule); dt = time.time() - t0
            a = downstream(Xtr, Xte, ytr, yte, sel); am = round(float(np.nanmean(list(a.values()))), 4)
            rows.append(dict(seed=sd, config=f"rule_{rule}", n=len(sel), base=len(set(sel) & base),
                             base_recall=round(len(set(sel) & base) / len(base), 3), noise=len(set(sel) & noise),
                             fit_s=round(dt, 1), auc_mean=am))
            print(f"sd{sd} rule_{rule:11s} n={len(sel):2d} base={len(set(sel)&base)}/{len(base)} noise={len(set(sel)&noise)} {dt:5.1f}s mean={am}", flush=True)
        # R3-1: survivor-FE on the one_se_min survivors
        sel = fit_rfecv(Xtr, ytr, "one_se_min")
        # FI rank = lgbm importance on survivors
        m = lgb.LGBMClassifier(n_estimators=200, verbose=-1).fit(Xtr[sel], ytr)
        fi_rank = [c for _, c in sorted(zip(m.feature_importances_, sel), reverse=True)]
        prods = survivor_fe(Xtr, ytr, sel, fi_rank, k=min(8, len(sel)))
        # build augmented frames and score downstream honestly
        def add_prods(Xdf):
            extra = {n: (Xdf[a] * Xdf[b]) for n, (a, b, _) in prods.items()}
            return pd.concat([Xdf[sel]] + [v.rename(n) for n, v in extra.items()], axis=1) if extra else Xdf[sel]
        Ztr, Zte = add_prods(Xtr), add_prods(Xte)
        a = downstream(Ztr, Zte, ytr, yte, list(Ztr.columns)); am = round(float(np.nanmean(list(a.values()))), 4)
        # did we form a TRUE interaction product (both operands in oper)?
        true_prod = any({pa, pb} <= oper for (pa, pb, _) in prods.values()) if oper else False
        rows.append(dict(seed=sd, config="survivor_FE", n=Ztr.shape[1], base=len(set(sel) & base),
                         base_recall=round(len(set(sel) & base) / len(base), 3), noise=len(set(sel) & noise),
                         fit_s=0.0, auc_mean=am))
        print(f"sd{sd} survivor_FE     n={Ztr.shape[1]:2d} (+{len(prods)} prods) true_interaction_formed={true_prod} mean={am} auc={a}", flush=True)
        if prods:
            print(f"      products kept: {list(prods.items())[:6]}", flush=True)
    df = pd.DataFrame(rows)
    print("\n=== mean over seeds ===")
    print(df.groupby("config").agg(auc_mean=("auc_mean", "mean"), base_recall=("base_recall", "mean"),
                                   noise=("noise", "mean"), n=("n", "mean")).round(4).to_string())


if __name__ == "__main__":
    main()
