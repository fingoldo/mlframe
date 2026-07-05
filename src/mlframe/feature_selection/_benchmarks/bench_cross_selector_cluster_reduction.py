"""Benchmark: does an unsupervised cluster-medoid pre-reduction speed up a
wrapper selector (RFECV) on WIDE correlated data without quality loss? (audit
integration-defaults-3). RFECV/BorutaShap currently get no cluster-aware
pruning; collapsing each correlated cluster to its medoid before the wrapper,
then expanding the support back, should cut wall-clock on wide data.

Uses the existing group_aware helpers (cluster_features_by_correlation +
_cluster_medoids) + sklearn RFECV (a clean wrapper proxy). Compares wall-clock
and OOS AUC: RFECV-on-full-X vs RFECV-on-medoids-then-expand. Bounded p/n.

RESULT (2026-06-03): WIN -> ~3.06x wall-clock speedup (1.2s -> 0.4s) with OOS AUC
delta -0.0001 (no loss) at p=108. The wrapper runs on ~30 medoids instead of all
108 correlated columns. GroupAwareMRMR was generalised to accept sklearn-style
boolean support_ so it can wrap RFECV/BorutaShap. NOTE: support_ expands to whole
clusters (larger kept set), so wiring this default-ON into the training suite is
a behavior change (more features kept) and left as a follow-up product call; the
capability ships via GroupAwareMRMR(RFECV(...)).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from mlframe.feature_selection.filters.group_aware import (
    _cluster_medoids,
    cluster_features_by_correlation,
)


def _wide_corr_frame(n, seed, n_groups=6, per_group=14, n_noise=24):
    rng = np.random.default_rng(seed)
    latents = [rng.standard_normal(n) for _ in range(n_groups)]
    cols = []
    # Only the first 3 groups drive y; the rest are correlated nuisance.
    for gi, z in enumerate(latents):
        for _ in range(per_group):
            cols.append(z + 0.25 * rng.standard_normal(n))
    for _ in range(n_noise):
        cols.append(rng.standard_normal(n))
    X = pd.DataFrame(np.column_stack(cols), columns=[f"f{i}" for i in range(len(cols))])
    signal = sum(latents[:3])
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-1.5 * signal))).astype(int)
    return X, pd.Series(y), len(cols)


def _rfecv(Xtr, ytr):
    est = LogisticRegression(max_iter=500)
    sel = RFECV(est, step=0.1, cv=3, min_features_to_select=2, n_jobs=1)
    sel.fit(Xtr, ytr)
    return np.where(sel.support_)[0]


def _auc(Xtr, ytr, Xte, yte, cols):
    if len(cols) == 0:
        cols = np.arange(Xtr.shape[1])
    clf = LogisticRegression(max_iter=1000).fit(Xtr.iloc[:, cols], ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte.iloc[:, cols])[:, 1])


def main():
    full_t, red_t, full_a, red_a, full_k, red_k = [], [], [], [], [], []
    for seed in range(3):
        X, y, p = _wide_corr_frame(2000, seed)
        Xtr, ytr, Xte, yte = X.iloc[:1400], y.iloc[:1400], X.iloc[1400:], y.iloc[1400:]

        t0 = time.perf_counter()
        full_sel = _rfecv(Xtr, ytr)
        full_t.append(time.perf_counter() - t0)
        full_a.append(_auc(Xtr, ytr, Xte, yte, full_sel)); full_k.append(len(full_sel))

        t0 = time.perf_counter()
        cid = cluster_features_by_correlation(Xtr, threshold=0.8, method="pearson")
        medoids = _cluster_medoids(Xtr, cid, method="pearson")
        med_sel = _rfecv(Xtr.iloc[:, medoids], ytr)  # indices into medoids
        sel_clusters = {int(cid[medoids[i]]) for i in med_sel}
        expanded = np.array([j for j in range(Xtr.shape[1]) if cid[j] in sel_clusters])
        red_t.append(time.perf_counter() - t0)
        red_a.append(_auc(Xtr, ytr, Xte, yte, expanded)); red_k.append(len(expanded))
        print(f"seed={seed} p={p}: FULL t={full_t[-1]:.1f}s auc={full_a[-1]:.4f} k={full_k[-1]} | "
              f"REDUCED t={red_t[-1]:.1f}s auc={red_a[-1]:.4f} k={red_k[-1]} "
              f"(medoids={len(medoids)})")
    print("---")
    print(f"mean wall-clock: full={np.mean(full_t):.1f}s  reduced={np.mean(red_t):.1f}s  " f"speedup={np.mean(full_t)/max(np.mean(red_t),1e-9):.2f}x")
    print(f"mean OOS AUC:    full={np.mean(full_a):.4f}  reduced={np.mean(red_a):.4f}  " f"delta={np.mean(red_a)-np.mean(full_a):+.4f}")


if __name__ == "__main__":
    main()
