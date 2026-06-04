"""Task B prototype bench: do MEDIAN-GATED operators beat threshold-0 and beat
plain products on designed beds y=(a>median_a)*b and y=(a>median_a)&(b>median_b)?

This is a STANDALONE prototype (no production edit). It simulates the leak-safe
stateful operator: the gate threshold (median) is computed on the TRAIN split only
and replayed on the holdout (exactly what a prewarp-style pseudo-unary would do in
the real pipeline). We then measure downstream AUC of appending each engineered
op family to the raw columns.

Verifies the WIN claim from agent E before recommending a (shared-file) pipeline change.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

try:
    from lightgbm import LGBMClassifier
    _HAVE_LGBM = True
except Exception:
    _HAVE_LGBM = False

N_JOBS = 2


def split(n, seed=7):
    rng = np.random.default_rng(seed)
    idx = np.arange(n); rng.shuffle(idx)
    cut = int(n * 0.6)
    return idx[:cut], idx[cut:]


# ---- designed beds (a,b are X[:,0], X[:,1]) ----
def bed_gated_med(n=6000, p=20, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    a, b = X[:, 0], X[:, 1]
    med_a = np.median(a)
    z = (a > med_a).astype(float) * b
    logit = 1.8 * z
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(np.int64)
    return X, y


def bed_thr_and_med(n=6000, p=20, seed=12):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    a, b = X[:, 0], X[:, 1]
    med_a, med_b = np.median(a), np.median(b)
    z = ((a > med_a) & (b > med_b)).astype(float)
    logit = 2.6 * (z - 0.25)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(np.int64)
    return X, y


def bed_proximity(n=6000, p=20, seed=13):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    a, b = X[:, 0], X[:, 1]
    logit = 1.8 * (-np.abs(a - b)) + 1.0
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(np.int64)
    return X, y


def eng_cols_leaksafe(Xtr, Xte, kind):
    """Build engineered column(s) for operand pair (0,1). Stateful thresholds
    (median) come from TRAIN only and are replayed on TEST (leak-safe)."""
    atr, btr = Xtr[:, 0], Xtr[:, 1]
    ate, bte = Xte[:, 0], Xte[:, 1]
    if kind == "product":
        return atr * btr, ate * bte
    if kind == "gated0":
        return (atr > 0).astype(float) * btr, (ate > 0).astype(float) * bte
    if kind == "gated_med":
        m = np.median(atr)            # TRAIN median, replayed on TEST
        return (atr > m).astype(float) * btr, (ate > m).astype(float) * bte
    if kind == "thr_and0":
        return (atr > 0).astype(float) * (btr > 0).astype(float), \
               (ate > 0).astype(float) * (bte > 0).astype(float)
    if kind == "thr_and_med":
        ma, mb = np.median(atr), np.median(btr)
        return (atr > ma).astype(float) * (btr > mb).astype(float), \
               (ate > ma).astype(float) * (bte > mb).astype(float)
    if kind == "absdiff":
        return np.abs(atr - btr), np.abs(ate - bte)
    raise ValueError(kind)


def auc_models(Xtr, ytr, Xte, yte):
    out = {}
    sc = StandardScaler()
    Xs_tr = sc.fit_transform(np.nan_to_num(Xtr))
    Xs_te = sc.transform(np.nan_to_num(Xte))
    lr = LogisticRegression(max_iter=500)
    lr.fit(Xs_tr, ytr); out["logit"] = roc_auc_score(yte, lr.predict_proba(Xs_te)[:, 1])
    kn = KNeighborsClassifier(n_neighbors=25, n_jobs=N_JOBS)
    kn.fit(Xs_tr, ytr); out["knn"] = roc_auc_score(yte, kn.predict_proba(Xs_te)[:, 1])
    if _HAVE_LGBM:
        gb = LGBMClassifier(n_estimators=150, num_leaves=31, n_jobs=N_JOBS,
                            verbose=-1, random_state=42)
        gb.fit(Xtr, ytr); out["lgbm"] = roc_auc_score(yte, gb.predict_proba(Xte)[:, 1])
    out["mean"] = float(np.mean(list(out.values())))
    return out


def run(bed_name, X, y):
    tr, te = split(len(X))
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    print(f"\n# bed={bed_name} n={len(X)} p={X.shape[1]} prev={y.mean():.3f}")
    base = auc_models(Xtr, ytr, Xte, yte)
    print(f"  {'raw_only':<14} logit={base['logit']:.4f} knn={base['knn']:.4f} "
          f"lgbm={base.get('lgbm',float('nan')):.4f} mean={base['mean']:.4f}")
    for kind in ("product", "gated0", "gated_med", "thr_and0", "thr_and_med", "absdiff"):
        ctr, cte = eng_cols_leaksafe(Xtr, Xte, kind)
        Xtr2 = np.column_stack([Xtr, ctr])
        Xte2 = np.column_stack([Xte, cte])
        a = auc_models(Xtr2, ytr, Xte2, yte)
        dm = a["mean"] - base["mean"]
        dl = a["logit"] - base["logit"]
        print(f"  +{kind:<13} logit={a['logit']:.4f} knn={a['knn']:.4f} "
              f"lgbm={a.get('lgbm',float('nan')):.4f} mean={a['mean']:.4f} "
              f"(d_mean={dm:+.4f} d_logit={dl:+.4f})")


def main():
    print(f"have_lgbm={_HAVE_LGBM}")
    run("gated_med  y=(a>med_a)*b", *bed_gated_med())
    run("thr_and_med y=(a>med_a)&(b>med_b)", *bed_thr_and_med())
    run("proximity  y=-|a-b|", *bed_proximity())


if __name__ == "__main__":
    main()
