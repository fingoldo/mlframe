"""BROAD validation of cluster-medoid pre-reduction for wrapper selectors
(audit integration-defaults-3), across DIVERSE datasets -- to decide whether it
is SAFE (never materially hurts OOS AUC) and USEFUL (speedup) enough to default
ON. Uses sklearn RFECV as the wrapper proxy + expand=True (keep whole cluster
when its medoid is selected) per the product decision.

Datasets:
  * synthetic make_classification with varying n_redundant / n_repeated /
    class_sep / imbalance (realistic correlated redundancy),
  * a hand-built RISK case: a cluster whose y-signal lives in a NON-medoid
    member (expand=True should still keep it),
  * real bundled sklearn datasets (breast_cancer, wine->binary, digits->even/odd)
    which have genuine correlation structure.

For each: RFECV on full X vs RFECV on cluster medoids (corr_threshold=0.9) then
expand support back. Reports OOS AUC delta (reduced - full) and speedup. SAFE iff
AUC delta >= ~ -0.01 everywhere; USEFUL iff speedup > 1 on correlated data.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer, load_digits, load_wine, make_classification,
)
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlframe.feature_selection.filters.group_aware import (
    _cluster_medoids, cluster_features_by_correlation,
)

warnings.filterwarnings("ignore")


def _rfecv_full(Xtr, ytr):
    sel = RFECV(LogisticRegression(max_iter=500), step=0.1, cv=3, min_features_to_select=2, n_jobs=1)
    sel.fit(Xtr, ytr)
    return np.where(sel.support_)[0]


def _rfecv_reduced(Xtr, ytr, corr_threshold=0.9):
    cid = cluster_features_by_correlation(Xtr, threshold=corr_threshold, method="pearson")
    medoids = _cluster_medoids(Xtr, cid, method="pearson")
    Xm = Xtr.iloc[:, medoids]
    sel = RFECV(LogisticRegression(max_iter=500), step=0.1, cv=3, min_features_to_select=2, n_jobs=1)
    sel.fit(Xm, ytr)
    sel_clusters = {int(cid[medoids[i]]) for i in np.where(sel.support_)[0]}
    # expand=True: keep ALL members of any selected cluster.
    expanded = np.array([j for j in range(Xtr.shape[1]) if cid[j] in sel_clusters])
    return expanded, len(medoids)


def _auc(Xtr, ytr, Xte, yte, cols):
    if len(cols) == 0:
        cols = np.arange(Xtr.shape[1])
    clf = LogisticRegression(max_iter=1000).fit(Xtr.iloc[:, cols], ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte.iloc[:, cols])[:, 1])


def _run(name, X, y, seed=0):
    X = pd.DataFrame(StandardScaler().fit_transform(np.asarray(X, float)), columns=[f"f{i}" for i in range(np.asarray(X).shape[1])])
    y = pd.Series(np.asarray(y).astype(int)).reset_index(drop=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.35, random_state=seed, stratify=y)
    Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)
    ytr, yte = ytr.reset_index(drop=True), yte.reset_index(drop=True)
    t0 = time.perf_counter(); full = _rfecv_full(Xtr, ytr); tf = time.perf_counter() - t0
    t0 = time.perf_counter(); red, nmed = _rfecv_reduced(Xtr, ytr); tr = time.perf_counter() - t0
    af, ar = _auc(Xtr, ytr, Xte, yte, full), _auc(Xtr, ytr, Xte, yte, red)
    print(f"{name:<26} p={X.shape[1]:>3} | FULL auc={af:.4f} k={len(full):>3} t={tf:5.2f}s | "
          f"RED auc={ar:.4f} k={len(red):>3} t={tr:5.2f}s med={nmed:>3} | "
          f"dAUC={ar-af:+.4f} speedup={tf/max(tr,1e-9):4.2f}x")
    return ar - af, tf / max(tr, 1e-9)


def _risk_case(n=2000, seed=0):
    # A cluster of near-duplicates where the y-signal lives in a NON-medoid
    # member (a slightly-off member carries the label). expand=True must keep it.
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n)
    cols = {f"dup{i}": base + 0.03 * rng.standard_normal(n) for i in range(5)}
    signal_member = base + 0.03 * rng.standard_normal(n)
    cols["dup_signal"] = signal_member
    for j in range(10):
        cols[f"noise{j}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    y = (signal_member + 0.3 * rng.standard_normal(n) > 0).astype(int)
    return X, y


def main():
    daucs, sps = [], []
    # ---- synthetic make_classification (varied redundancy / sep / imbalance) ----
    configs = [
        dict(n_features=60, n_informative=8, n_redundant=30, n_repeated=6, class_sep=1.0, weights=None),
        dict(n_features=80, n_informative=10, n_redundant=10, n_repeated=0, class_sep=0.6, weights=None),
        dict(n_features=40, n_informative=6, n_redundant=20, n_repeated=4, class_sep=1.2, weights=[0.9, 0.1]),
        dict(n_features=100, n_informative=12, n_redundant=0, n_repeated=0, class_sep=0.8, weights=None),  # no clusters
    ]
    for i, cfg in enumerate(configs):
        X, y = make_classification(n_samples=2500, random_state=i, n_clusters_per_class=2, **cfg)
        d, s = _run(f"synth_{i}(red{cfg['n_redundant']}rep{cfg['n_repeated']})", X, y, seed=i)
        daucs.append(d); sps.append(s)
    # ---- risk case ----
    Xr, yr = _risk_case()
    d, s = _run("risk_signal_in_nonmedoid", Xr, yr); daucs.append(d); sps.append(s)
    # ---- real bundled datasets ----
    bc = load_breast_cancer(); d, s = _run("real_breast_cancer", bc.data, bc.target); daucs.append(d); sps.append(s)
    wn = load_wine(); d, s = _run("real_wine(0_vs_rest)", wn.data, (wn.target == 0).astype(int)); daucs.append(d); sps.append(s)
    dg = load_digits(); d, s = _run("real_digits(even_odd)", dg.data, (dg.target % 2)); daucs.append(d); sps.append(s)
    print("=" * 100)
    daucs = np.array(daucs)
    print(f"AUC delta (reduced - full): min={daucs.min():+.4f} mean={daucs.mean():+.4f} max={daucs.max():+.4f}")
    print(f"speedup: min={min(sps):.2f}x mean={np.mean(sps):.2f}x max={max(sps):.2f}x")
    print(f"SAFE (all dAUC >= -0.01)? {bool((daucs >= -0.01).all())}")
    bad = [i for i, d in enumerate(daucs) if d < -0.01]
    if bad:
        print(f"  HARM on dataset indices: {bad}")


if __name__ == "__main__":
    main()
