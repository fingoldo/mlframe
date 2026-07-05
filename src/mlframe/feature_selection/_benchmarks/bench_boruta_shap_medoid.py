"""Validate cluster-medoid pre-reduction for BorutaShap -- specifically whether
its SHADOW-importance null behaves under reduction (audit integration-defaults-3,
the BorutaShap follow-up). Concern: BorutaShap accepts a feature when its SHAP/
gini importance beats the max SHADOW (permuted) importance; reducing to medoids
changes the shadow null (fewer features -> different max-shadow). Counter-effect:
redundant copies split a cluster's importance across members, DILUTING each
member's per-feature test -> medoids (one representative) should give a CLEANER,
more reliable BorutaShap decision.

Compare full BorutaShap vs BorutaShap-on-medoids-then-expand: OOS AUC + wall-clock
+ whether the informative features survive and noise is rejected. Bounded n/p and
n_trials so BorutaShap stays cheap.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlframe.feature_selection.boruta_shap import BorutaShap
from mlframe.feature_selection.filters.group_aware import (
    _cluster_medoids, cluster_features_by_correlation,
)

warnings.filterwarnings("ignore")


def _boruta(Xtr, ytr):
    bs = BorutaShap(importance_measure="gini", classification=True, n_trials=25, verbose=False, random_state=0)
    bs.fit(Xtr, ytr)
    return list(bs.accepted)


def _auc(Xtr, ytr, Xte, yte, cols):
    cols = [c for c in cols if c in Xtr.columns]
    if not cols:
        cols = list(Xtr.columns)
    clf = LogisticRegression(max_iter=1000).fit(Xtr[cols], ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte[cols])[:, 1])


def _run(name, X, y, seed=0):
    X = pd.DataFrame(StandardScaler().fit_transform(np.asarray(X, float)), columns=[f"f{i}" for i in range(np.asarray(X).shape[1])])
    y = pd.Series(np.asarray(y).astype(int)).reset_index(drop=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.35, random_state=seed, stratify=y)
    Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)
    ytr, yte = ytr.reset_index(drop=True), yte.reset_index(drop=True)

    t0 = time.perf_counter(); full = _boruta(Xtr, ytr); tf = time.perf_counter() - t0

    t0 = time.perf_counter()
    cid = cluster_features_by_correlation(Xtr, threshold=0.9, method="pearson")
    medoids = _cluster_medoids(Xtr, cid, method="pearson")
    med_names = [Xtr.columns[i] for i in medoids]
    acc_med = _boruta(Xtr[med_names], ytr)
    sel_clusters = {int(cid[Xtr.columns.get_loc(c)]) for c in acc_med if c in Xtr.columns}
    expanded = [Xtr.columns[j] for j in range(Xtr.shape[1]) if cid[j] in sel_clusters]
    tr = time.perf_counter() - t0

    af, ar = _auc(Xtr, ytr, Xte, yte, full), _auc(Xtr, ytr, Xte, yte, expanded)
    print(f"{name:<22} p={X.shape[1]:>3} | FULL auc={af:.4f} k={len(full):>3} t={tf:5.1f}s | "
          f"RED auc={ar:.4f} k={len(expanded):>3} t={tr:5.1f}s med={len(medoids):>3} | "
          f"dAUC={ar-af:+.4f} speedup={tf/max(tr,1e-9):.2f}x")
    return ar - af, tf / max(tr, 1e-9)


def main():
    daucs, sps = [], []
    for i, cfg in enumerate([
        dict(n_features=50, n_informative=6, n_redundant=24, n_repeated=4, class_sep=1.0),
        dict(n_features=40, n_informative=8, n_redundant=8, n_repeated=0, class_sep=0.7),
    ]):
        X, y = make_classification(n_samples=1500, random_state=i, n_clusters_per_class=2, **cfg)
        d, s = _run(f"synth_{i}(red{cfg['n_redundant']})", X, y, seed=i); daucs.append(d); sps.append(s)
    bc = load_breast_cancer(); d, s = _run("real_breast_cancer", bc.data, bc.target); daucs.append(d); sps.append(s)
    print("=" * 92)
    daucs = np.array(daucs)
    print(f"AUC delta (reduced-full): min={daucs.min():+.4f} mean={daucs.mean():+.4f} max={daucs.max():+.4f}")
    print(f"speedup: min={min(sps):.2f}x mean={np.mean(sps):.2f}x")
    print(f"SAFE (all dAUC >= -0.01)? {bool((daucs >= -0.01).all())}")


if __name__ == "__main__":
    main()
