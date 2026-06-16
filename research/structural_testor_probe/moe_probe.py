"""MoE-over-predictions Phase-0 probe (research/MoE_Tabular.md).

Corrected design (user steer):
  * MoE sees ONLY the member models' predictions (+ row-wise aggregates over them), never raw X.
  * MoE/gate is TUNED on train via out-of-fold member predictions (honest, no leakage);
    members are refit on full train and predict test. EVERY number in the table is TEST-set.
  * Reference rows: single_model (the 1st pool model) and best_model (best individual member,
    chosen by train-OOF AUC, reported on test).

Question: does an input-dependent (per-row / per-region) combination of the N member predictions
beat MLFRAME'S simple prob-mean ensembles (combine_probs flavours, which use GLOBAL fixed weights)?

Pool members -> OOF train probs + test probs. Then:
  single_model / best_model                          -- references
  ens_<flavour>  (arithm/quad/qube/geo/harm/median/rrf) -- mlframe combine_probs, global weights
  stack_logit / stack_gbm                    -- learned global meta-combiner (stacking)
  moe_router_best / moe_router_softlogit             -- per-region (KMeans in pred-space) combiners
Datasets reused from the FS cross-selector bench + planted-heterogeneity synth. Multi-seed; TEST AUC.
"""
from __future__ import annotations
import os, sys, warnings, time
import numpy as np
warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold

from mlframe.models.ensembling.base import combine_probs
BENCH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "..", "src", "mlframe", "feature_selection",
                                      "_benchmarks", "fs_hybrid"))
sys.path.insert(0, BENCH)
import round4_broad_realdata_bench as B  # noqa: E402

FLAVOURS = ["arithm", "quad", "qube", "geo", "harm", "median", "rrf"]


def _members():
    return [
        ("lgbm",       lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05, n_jobs=-1, verbose=-1)),
        ("extratrees", ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=0)),
        ("rf",         RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)),
        ("hgb",        HistGradientBoostingClassifier(max_iter=300, random_state=0)),
        ("logit",      LogisticRegression(max_iter=1000)),
    ]


def _needs_scale(name):
    return name == "logit"


def member_predictions(Xtr, ytr, Xte, seed):
    """OOF train probs (honest, for gate training) + test probs, per member. Returns
    (train_p1 (Ntr,K), test_p1 (Nte,K), member_names, oof_auc list)."""
    cv = StratifiedKFold(4, shuffle=True, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    tr_list, te_list, names, oof_auc = [], [], [], []
    for name, m in _members():
        Xtr_m = sc.transform(Xtr) if _needs_scale(name) else Xtr
        Xte_m = sc.transform(Xte) if _needs_scale(name) else Xte
        oof = cross_val_predict(m, Xtr_m, ytr, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        m.fit(Xtr_m, ytr)
        te = m.predict_proba(Xte_m)[:, 1]
        tr_list.append(oof); te_list.append(te); names.append(name)
        oof_auc.append(roc_auc_score(ytr, oof))
    return np.column_stack(tr_list), np.column_stack(te_list), names, oof_auc


def _aggs(P):
    """Row-wise aggregates over member class-1 probs (N,K) -> (N,6)."""
    return np.column_stack([P.mean(1), P.std(1), P.min(1), P.max(1), P.max(1) - P.min(1), np.median(P, 1)])


def gate_feats(P):
    return np.hstack([P, _aggs(P)])


# --------------------------------------------------------------------------- MoE combiners (pred-space)
def stack_logit(trP, ytr, teP):
    g = LogisticRegression(max_iter=1000).fit(gate_feats(trP), ytr)
    return g.predict_proba(gate_feats(teP))[:, 1]


def stack_gbm(trP, ytr, teP):
    g = lgb.LGBMClassifier(n_estimators=150, num_leaves=15, learning_rate=0.05, n_jobs=-1, verbose=-1)
    g.fit(gate_feats(trP), ytr)
    return g.predict_proba(gate_feats(teP))[:, 1]


def moe_router_best(trP, ytr, teP, K=4, seed=0):
    """Per-region (KMeans in pred-space) HARD pick of the single best member (by in-region OOF AUC)."""
    km = KMeans(K, n_init=4, random_state=seed).fit(trP)
    lab_tr, lab_te = km.labels_, km.predict(teP)
    glob_best = int(np.argmax([roc_auc_score(ytr, trP[:, j]) for j in range(trP.shape[1])]))
    pick = {}
    for c in range(K):
        idx = lab_tr == c
        if idx.sum() < 20 or len(np.unique(ytr[idx])) < 2:
            pick[c] = glob_best; continue
        aucs = [roc_auc_score(ytr[idx], trP[idx, j]) for j in range(trP.shape[1])]
        pick[c] = int(np.argmax(aucs))
    out = np.empty(len(teP))
    for c in range(K):
        m = lab_te == c
        if m.any():
            out[m] = teP[m, pick.get(c, glob_best)]
    return out


def moe_router_softlogit(trP, ytr, teP, K=4, seed=0):
    """Per-region learned logistic combiner of members (soft, input-dependent weights)."""
    km = KMeans(K, n_init=4, random_state=seed).fit(trP)
    lab_tr, lab_te = km.labels_, km.predict(teP)
    glob = LogisticRegression(max_iter=1000).fit(trP, ytr)
    models = {}
    for c in range(K):
        idx = lab_tr == c
        models[c] = (LogisticRegression(max_iter=1000).fit(trP[idx], ytr[idx])
                     if (idx.sum() >= 50 and len(np.unique(ytr[idx])) == 2) else glob)
    out = np.empty(len(teP))
    for c in range(K):
        m = lab_te == c
        if m.any():
            out[m] = models.get(c, glob).predict_proba(teP[m])[:, 1]
    return out


# --------------------------------------------------------------------------- datasets
def make_two_regime(n=4000, p=20, seed=0):
    rng = np.random.default_rng(seed); X = rng.normal(size=(n, p)); g = X[:, -1]
    y = np.where(g < 0, (X[:, 0] + X[:, 1] + X[:, 2] > 0),
                 (np.sign(X[:, 3] * X[:, 4]) - X[:, 5] > 0)).astype(np.int64)
    return X, y


def make_homogeneous(n=4000, p=20, seed=0):
    rng = np.random.default_rng(seed); X = rng.normal(size=(n, p))
    y = (X[:, 0] + 0.7 * X[:, 1] + 0.5 * X[:, 2] + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    return X, y


# --------------------------------------------------------------------------- driver
def run(name, Xf, yf, seeds=(0, 1, 2)):
    Xf = np.asarray(Xf, float); yf = np.asarray(yf).astype(int)
    print(f"\n{'='*90}\nDATASET {name}  shape={Xf.shape}  pos={yf.mean():.3f}\n{'='*90}", flush=True)
    rows = (["single_model", "best_model"] + [f"ens_{f}" for f in FLAVOURS]
            + ["stack_logit", "stack_gbm", "moe_router_best", "moe_router_softlogit"])
    aucs = {m: [] for m in rows}
    for sd in seeds:
        Xtr, Xte, ytr, yte = train_test_split(Xf, yf, test_size=0.4, random_state=sd, stratify=yf)
        trP, teP, names, oof_auc = member_predictions(Xtr, ytr, Xte, sd)
        aucs["single_model"].append(roc_auc_score(yte, teP[:, 0]))            # 1st pool model (lgbm)
        aucs["best_model"].append(roc_auc_score(yte, teP[:, int(np.argmax(oof_auc))]))  # best by OOF, test AUC
        stacked = np.stack([np.column_stack([1 - teP[:, j], teP[:, j]]) for j in range(teP.shape[1])], 0)
        for fl in FLAVOURS:
            aucs[f"ens_{fl}"].append(roc_auc_score(yte, combine_probs(stacked, fl)[:, 1]))
        aucs["stack_logit"].append(roc_auc_score(yte, stack_logit(trP, ytr, teP)))
        aucs["stack_gbm"].append(roc_auc_score(yte, stack_gbm(trP, ytr, teP)))
        aucs["moe_router_best"].append(roc_auc_score(yte, moe_router_best(trP, ytr, teP, seed=sd)))
        aucs["moe_router_softlogit"].append(roc_auc_score(yte, moe_router_softlogit(trP, ytr, teP, seed=sd)))
    best_ens = max(np.mean(aucs[f"ens_{f}"]) for f in FLAVOURS)
    print(f"  {'method':<22}{'TEST AUC':>9}{'std':>7}{'  vs best_simple_ens':>20}", flush=True)
    for m in rows:
        a = np.array(aucs[m])
        tag = "" if (m.startswith("ens_") or m in ("single_model", "best_model")) else f"  {a.mean()-best_ens:+.4f}"
        star = "  *best simple ens" if (m.startswith("ens_") and abs(a.mean()-best_ens) < 1e-9) else ""
        print(f"  {m:<22}{a.mean():>9.4f}{a.std():>7.4f}{tag}{star}", flush=True)


def main():
    t0 = time.time()
    run("synth:two_regime(home-turf)", *make_two_regime())
    run("synth:homogeneous(control)", *make_homogeneous())
    for nm, kw in [("madelon", dict(name="madelon", version=1)),
                   ("gina_agnostic", dict(name="gina_agnostic", version=1)),
                   ("scene", dict(name="scene", version=1))]:
        try:
            X, y, note = B.load_one(nm, kw, 3000, 1200)
            run(f"bench:{nm}", X.to_numpy(), y.to_numpy())
        except Exception as e:
            print(f"[skip {nm}: {type(e).__name__}: {e}]", flush=True)
    try:
        X, y, note = B.fallback_breast_cancer()
        run("bench:breast_cancer", X.to_numpy(), y.to_numpy())
    except Exception as e:
        print(f"[skip breast_cancer: {e}]", flush=True)
    print(f"\n[total {time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
