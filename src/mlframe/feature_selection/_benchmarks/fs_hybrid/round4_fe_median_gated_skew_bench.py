"""Task B discriminating bench: where threshold-0 != median, does the TRAIN-median
gate beat threshold-0? Uses SHIFTED/SKEWED operands (median far from 0). Multi-seed.

Leak-safe: median from TRAIN only, replayed on TEST.
"""
from __future__ import annotations
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
try:
    from lightgbm import LGBMClassifier; _HAVE_LGBM = True
except Exception:
    _HAVE_LGBM = False
N_JOBS = 2
SEEDS = [1, 2, 3, 4]


def split(n, seed):
    rng = np.random.default_rng(2000 + seed); idx = np.arange(n); rng.shuffle(idx)
    c = int(n * 0.6); return idx[:c], idx[c:]


def bed_gated_skew(seed, n=6000, p=20):
    """a,b SHIFTED so median(a)~=3 (far from 0). y depends on (a>median_a)*b."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, 0] = X[:, 0] + 3.0     # shift operand a: median ~ 3, NOT 0
    a, b = X[:, 0], X[:, 1]
    z = (a > np.median(a)).astype(float) * b
    y = (rng.random(n) < 1/(1+np.exp(-1.8*z))).astype(np.int64)
    return X, y


def bed_thr_and_skew(seed, n=6000, p=20):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, 0] = np.exp(0.7 * X[:, 0])      # lognormal: median ~1, threshold-0 useless
    X[:, 1] = X[:, 1] + 2.0              # shifted: median ~2
    a, b = X[:, 0], X[:, 1]
    z = ((a > np.median(a)) & (b > np.median(b))).astype(float)
    y = (rng.random(n) < 1/(1+np.exp(-2.6*(z-0.25)))).astype(np.int64)
    return X, y


def eng(Xtr, Xte, kind):
    atr, btr, ate, bte = Xtr[:,0], Xtr[:,1], Xte[:,0], Xte[:,1]
    if kind == "product": return atr*btr, ate*bte
    if kind == "gated0":  return (atr>0).astype(float)*btr, (ate>0).astype(float)*bte
    if kind == "gated_med":
        m = np.median(atr); return (atr>m).astype(float)*btr, (ate>m).astype(float)*bte
    if kind == "thr_and0": return (atr>0).astype(float)*(btr>0).astype(float), (ate>0).astype(float)*(bte>0).astype(float)
    if kind == "thr_and_med":
        ma, mb = np.median(atr), np.median(btr)
        return (atr>ma).astype(float)*(btr>mb).astype(float), (ate>ma).astype(float)*(bte>mb).astype(float)
    raise ValueError(kind)


def auc(Xtr, ytr, Xte, yte):
    out = {}
    sc = StandardScaler(); a = sc.fit_transform(np.nan_to_num(Xtr)); b = sc.transform(np.nan_to_num(Xte))
    lr = LogisticRegression(max_iter=500); lr.fit(a, ytr); out["logit"] = roc_auc_score(yte, lr.predict_proba(b)[:,1])
    kn = KNeighborsClassifier(n_neighbors=25, n_jobs=N_JOBS); kn.fit(a, ytr); out["knn"] = roc_auc_score(yte, kn.predict_proba(b)[:,1])
    if _HAVE_LGBM:
        gb = LGBMClassifier(n_estimators=150, num_leaves=31, n_jobs=N_JOBS, verbose=-1, random_state=42)
        gb.fit(Xtr, ytr); out["lgbm"] = roc_auc_score(yte, gb.predict_proba(Xte)[:,1])
    out["mean"] = float(np.mean(list(out.values()))); return out


def run(name, mk, kinds):
    print(f"\n# bed={name} (multi-seed mean d vs raw_only)")
    acc = {k: [] for k in (["raw_only"] + kinds)}
    accl = {k: [] for k in (["raw_only"] + kinds)}
    for s in SEEDS:
        X, y = mk(s); tr, te = split(len(X), s)
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        base = auc(Xtr, ytr, Xte, yte)
        acc["raw_only"].append(base["mean"]); accl["raw_only"].append(base["logit"])
        for k in kinds:
            ctr, cte = eng(Xtr, Xte, k)
            a = auc(np.column_stack([Xtr, ctr]), ytr, np.column_stack([Xte, cte]), yte)
            acc[k].append(a["mean"]); accl[k].append(a["logit"])
    base_m = np.mean(acc["raw_only"]); base_l = np.mean(accl["raw_only"])
    print(f"  raw_only      mean={base_m:.4f} logit={base_l:.4f}")
    for k in kinds:
        dm = np.mean(acc[k]) - base_m; dl = np.mean(accl[k]) - base_l
        print(f"  +{k:<13} d_mean={dm:+.4f}  d_logit={dl:+.4f}  (mean={np.mean(acc[k]):.4f})")


def main():
    print(f"have_lgbm={_HAVE_LGBM} seeds={SEEDS}")
    run("gated_skew y=(a>med_a)*b, median_a~3", bed_gated_skew,
        ["product", "gated0", "gated_med"])
    run("thr_and_skew y=AND, a~lognormal b~shift", bed_thr_and_skew,
        ["product", "thr_and0", "thr_and_med"])


if __name__ == "__main__":
    main()
