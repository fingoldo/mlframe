"""Combined probe for two Zhuravlev-derived meta-learning ideas, reusing the leakage-safe
member-prediction harness from moe_probe (OOF train preds train the meta-learner; test preds score).

  research/Polynomial_Aggregators.md -> POLY block: polynomial expansion of base preds + Ridge/Lasso/
     Logistic. KEY benchmark = poly vs GBM stacking (if GBM already captures the pred-interactions, reject).
  research/Algebraic_Correction.md   -> CORRECTION block: residual correction (X-aware) + meta-correction
     (X+preds). KEY = must beat stacking.

Baselines: ens_arithm (mean), stack_logit (linear), stack_ridge, stack_gbm (the strong meta-learner).
All numbers TEST-set AUC, 3 seeds. Datasets reused from the FS bench + 2 synth.
"""
from __future__ import annotations
import os, sys, warnings, time
import numpy as np
warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from moe_probe import member_predictions, gate_feats, B, combine_probs, make_two_regime, make_homogeneous  # noqa: E402


def _poly_logit(trP, ytr, teP, degree, l1=False):
    pf = PolynomialFeatures(degree=degree, include_bias=False)
    Ztr = pf.fit_transform(trP); Zte = pf.transform(teP)
    sc = StandardScaler().fit(Ztr)
    g = LogisticRegression(max_iter=2000, penalty=("l1" if l1 else "l2"),
                           solver=("liblinear" if l1 else "lbfgs"), C=1.0)
    g.fit(sc.transform(Ztr), ytr)
    return g.predict_proba(sc.transform(Zte))[:, 1], int((np.abs(g.coef_) > 1e-6).sum())


def _stack_gbm(trP, ytr, teP):
    g = lgb.LGBMClassifier(n_estimators=150, num_leaves=15, learning_rate=0.05, n_jobs=-1, verbose=-1)
    g.fit(gate_feats(trP), ytr)
    return g.predict_proba(gate_feats(teP))[:, 1]


def _residual_correction(Xtr, ytr, Xte, base_oof, base_te):
    """Algebraic Type B: correct the base model's OOF residual with an X-aware GBM regressor.
    final = clip(base + corrector(X)). Honest: residual from OOF base preds."""
    resid = ytr - base_oof
    r = lgb.LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.05, n_jobs=-1, verbose=-1)
    r.fit(Xtr, resid)
    return np.clip(base_te + r.predict(Xte), 0, 1)


def _meta_correction(Xtr, ytr, Xte, trP, teP):
    """Algebraic Type D: GBM meta over [original X + member preds + aggregates]."""
    g = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05, n_jobs=-1, verbose=-1)
    g.fit(np.hstack([Xtr, gate_feats(trP)]), ytr)
    return g.predict_proba(np.hstack([Xte, gate_feats(teP)]))[:, 1]


def run(name, Xf, yf, seeds=(0, 1, 2)):
    Xf = np.asarray(Xf, float); yf = np.asarray(yf).astype(int)
    print(f"\n{'='*94}\nDATASET {name}  shape={Xf.shape}  pos={yf.mean():.3f}\n{'='*94}", flush=True)
    rows = ["ens_arithm", "stack_logit", "stack_ridge", "stack_gbm",
            "poly2_l2", "poly2_l1", "poly3_l2",
            "corr_residual", "corr_meta"]
    aucs = {m: [] for m in rows}; nterms = {"poly2_l1": []}
    for sd in seeds:
        Xtr, Xte, ytr, yte = train_test_split(Xf, yf, test_size=0.4, random_state=sd, stratify=yf)
        trP, teP, names, oof_auc = member_predictions(Xtr, ytr, Xte, sd)
        stacked = np.stack([np.column_stack([1 - teP[:, j], teP[:, j]]) for j in range(teP.shape[1])], 0)
        aucs["ens_arithm"].append(roc_auc_score(yte, combine_probs(stacked, "arithm")[:, 1]))
        aucs["stack_logit"].append(roc_auc_score(yte, LogisticRegression(max_iter=1000).fit(trP, ytr).predict_proba(teP)[:, 1]))
        rg = Ridge().fit(trP, ytr)
        aucs["stack_ridge"].append(roc_auc_score(yte, rg.predict(teP)))
        aucs["stack_gbm"].append(roc_auc_score(yte, _stack_gbm(trP, ytr, teP)))
        p, _ = _poly_logit(trP, ytr, teP, 2, l1=False); aucs["poly2_l2"].append(roc_auc_score(yte, p))
        p, nt = _poly_logit(trP, ytr, teP, 2, l1=True); aucs["poly2_l1"].append(roc_auc_score(yte, p)); nterms["poly2_l1"].append(nt)
        p, _ = _poly_logit(trP, ytr, teP, 3, l1=False); aucs["poly3_l2"].append(roc_auc_score(yte, p))
        aucs["corr_residual"].append(roc_auc_score(yte, _residual_correction(Xtr, ytr, Xte, trP[:, 0], teP[:, 0])))
        aucs["corr_meta"].append(roc_auc_score(yte, _meta_correction(Xtr, ytr, Xte, trP, teP)))
    base = np.mean(aucs["stack_gbm"])
    print(f"  {'method':<16}{'TEST AUC':>9}{'std':>7}{'  vs stack_gbm':>15}", flush=True)
    for m in rows:
        a = np.array(aucs[m]); tag = "" if m == "stack_gbm" else f"  {a.mean()-base:+.4f}"
        extra = f"  (~{np.mean(nterms['poly2_l1']):.0f} terms kept)" if m == "poly2_l1" else ""
        print(f"  {m:<16}{a.mean():>9.4f}{a.std():>7.4f}{tag}{extra}", flush=True)


def main():
    t0 = time.time()
    run("synth:two_regime", *make_two_regime())
    run("synth:homogeneous", *make_homogeneous())
    for nm, kw in [("madelon", dict(name="madelon", version=1)),
                   ("gina_agnostic", dict(name="gina_agnostic", version=1)),
                   ("scene", dict(name="scene", version=1))]:
        try:
            X, y, note = B.load_one(nm, kw, 3000, 1200); run(f"bench:{nm}", X.to_numpy(), y.to_numpy())
        except Exception as e:
            print(f"[skip {nm}: {type(e).__name__}: {e}]", flush=True)
    try:
        X, y, note = B.fallback_breast_cancer(); run("bench:breast_cancer", X.to_numpy(), y.to_numpy())
    except Exception as e:
        print(f"[skip breast_cancer: {e}]", flush=True)
    print(f"\n[total {time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
