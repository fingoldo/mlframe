"""Coreset / dataset-pruning Phase-0 go/no-go probe (research/CoreSet_Selection.md).

Success here is NOT higher quality -- it is SIMILAR quality + real END-TO-END speedup
(prune_time + fit_time vs full fit). The critical baselines are random + stratified sampling:
the ТЗ warns most sophisticated methods fail to beat them.

Methods (tabular-appropriate):
  random        -- uniform subsample (Baseline A; must always be present)
  stratified    -- class-proportion-preserving subsample (Baseline B)
  el2n          -- keep highest |p - y| from an OOF proxy GBM (GraNd ~ EL2N for binary logloss)
  forgetting    -- keep most-forgotten (correct->incorrect flips across boosting stages)
  kcenter       -- k-center greedy geometric coverage on scaled X (the canonical coreset)
  (CNN/k-medoids skipped: O(n^2), impractical at the n where speedup matters -- noted, not hidden.)

Pruning levels 10/30/50% kept; LightGBM (primary, timed) + Logistic. Multi-seed. Reports per
(dataset,method,frac): TEST AUC, quality retention vs full, prune_s, fit_s, total_s, end-to-end speedup.
"""
from __future__ import annotations
import warnings, time
import numpy as np
warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold

KEEP = [0.10, 0.30, 0.50]
SEEDS = (0, 1)


def _lgbm():
    return lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, n_jobs=-1, verbose=-1)


# --------------------------------------------------------------------------- scorers
def el2n_scores(X, y, seed):
    """|p - y| from a 3-fold OOF proxy GBM. High = hard/informative."""
    oof = cross_val_predict(_lgbm(), X, y, cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                            method="predict_proba", n_jobs=1)[:, 1]
    return np.abs(y - oof)


def forgetting_scores(X, y, seed):
    """Count correct->incorrect flips of a sample across boosting stages (staged_predict)."""
    m = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, random_state=seed, n_jobs=-1, verbose=-1)
    m.fit(X, y)
    flips = np.zeros(len(y))
    prev = None
    booster = m.booster_
    # staged raw scores every 25 trees (cheap) -> correctness sequence
    for it in range(25, 301, 25):
        raw = booster.predict(X, num_iteration=it)
        correct = ((raw >= 0.5).astype(int) == y)
        if prev is not None:
            flips += (prev & ~correct)  # was correct, now wrong
        prev = correct
    return flips


def kcenter_greedy(Xs, k, seed):
    """Greedy farthest-point coreset: iteratively add the point farthest from the current set."""
    n = Xs.shape[0]
    rng = np.random.default_rng(seed)
    first = int(rng.integers(n))
    chosen = [first]
    mind = np.linalg.norm(Xs - Xs[first], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(mind))
        chosen.append(nxt)
        np.minimum(mind, np.linalg.norm(Xs - Xs[nxt], axis=1), out=mind)
    return np.array(chosen)


# --------------------------------------------------------------------------- pruning dispatch
def prune(method, X, y, Xs, frac, seed):
    k = max(50, int(len(y) * frac))
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    if method == "random":
        idx = rng.choice(len(y), k, replace=False)
    elif method == "stratified":
        idx, _ = train_test_split(np.arange(len(y)), train_size=k, random_state=seed, stratify=y)
    elif method == "el2n":
        idx = np.argsort(-el2n_scores(X, y, seed))[:k]
    elif method == "forgetting":
        idx = np.argsort(-forgetting_scores(X, y, seed))[:k]
    elif method == "kcenter":
        idx = kcenter_greedy(Xs, k, seed)
    else:
        raise ValueError(method)
    return np.asarray(idx), time.perf_counter() - t0


# --------------------------------------------------------------------------- datasets
def make_large_synth(n=60000, p=30, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (X[:, 0] + 0.8 * X[:, 1] - 0.6 * X[:, 2] + 0.5 * X[:, 3] * X[:, 4] + 0.4 * rng.normal(size=n) > 0).astype(np.int64)
    return X, y


def load_openml(name, version, cap=60000):
    from sklearn.datasets import fetch_openml
    import pandas as pd
    d = fetch_openml(name, version=version, as_frame=True, parser="auto")
    X = d.data.copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "category":
            X[c] = pd.factorize(X[c])[0]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.Series(pd.factorize(np.asarray(d.target).ravel())[0])
    y = (y == y.value_counts().idxmax()).astype(int)
    X, y = X.to_numpy(float), y.to_numpy()
    if len(y) > cap:
        i, _ = train_test_split(np.arange(len(y)), train_size=cap, random_state=0, stratify=y)
        X, y = X[i], y[i]
    return X, y


# --------------------------------------------------------------------------- driver
def run(name, X, y):
    print(f"\n{'='*100}\nDATASET {name}  shape={X.shape}  pos={y.mean():.3f}\n{'='*100}", flush=True)
    methods = ["random", "stratified", "el2n", "forgetting", "kcenter"]
    # accumulate per (method,frac): auc, retention, prune_s, fit_s, total_s, speedup
    agg = {}
    full_auc, full_fit = [], []
    for sd in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=sd, stratify=y)
        Xs = StandardScaler().fit_transform(Xtr)
        # full reference
        t0 = time.perf_counter(); fm = _lgbm().fit(Xtr, ytr); ff = time.perf_counter() - t0
        fa = roc_auc_score(yte, fm.predict_proba(Xte)[:, 1])
        full_auc.append(fa); full_fit.append(ff)
        for meth in methods:
            for fr in KEEP:
                idx, pt = prune(meth, Xtr, ytr, Xs, fr, sd)
                t0 = time.perf_counter(); m = _lgbm().fit(Xtr[idx], ytr[idx]); ft = time.perf_counter() - t0
                a = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
                agg.setdefault((meth, fr), []).append((a, a / fa, pt, ft, pt + ft, ff / (pt + ft)))
    fa_m, ff_m = np.mean(full_auc), np.mean(full_fit)
    print(f"  FULL: auc={fa_m:.4f}  fit={ff_m:.2f}s  (n_train={int(len(y)*0.7)})", flush=True)
    print(f"  {'method':<12}{'keep':>6}{'AUC':>9}{'retain':>8}{'prune_s':>9}{'fit_s':>8}{'total_s':>9}{'e2e_speedup':>12}", flush=True)
    for meth in methods:
        for fr in KEEP:
            v = np.array(agg[(meth, fr)])
            a, ret, pt, ft, tot, sp = v.mean(0)
            print(f"  {meth:<12}{int(fr*100):>5}%{a:>9.4f}{ret:>8.3f}{pt:>9.2f}{ft:>8.2f}{tot:>9.2f}{sp:>12.2f}", flush=True)


def main():
    t0 = time.time()
    run("synth_large(60k)", *make_large_synth())
    for nm, ver in [("electricity", 1), ("adult", 2), ("bank-marketing", 1)]:
        try:
            X, y = load_openml(nm, ver)
            run(f"openml:{nm}", X, y)
        except Exception as e:
            print(f"[skip {nm}: {type(e).__name__}: {e}]", flush=True)
    print(f"\n[total {time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
