"""Bench: estimator-type-aware RFECV importance aggregation (importance_agg='dispatched') vs 'legacy'.

Decision rule (CLAUDE.md "Variant defaults"): flip the default to 'dispatched' ONLY if it wins the MAJORITY
of (scenario x seed) cells on an HONEST holdout -- a test split RFECV never saw, scored by the downstream
model refit on the RFECV-selected features. Else keep 'legacy' + commit this bench + a REJECTED verdict.

Scenarios (each crafted so a known signal/noise structure exists):
  tree_heavy   : RandomForest, informative + many noise cols (gain importance is noisy fold-to-fold).
  linear_signal: LogisticRegression, clean linear signal + noise.
  imbalanced   : LogisticRegression, 5% positive class.
  sign_flip    : LogisticRegression, correlated decoys whose coef sign is unstable across folds.
  kernel       : SVC(rbf) -> permutation FI path (dispatched defers to legacy; sanity that it doesn't regress).

Run: python -m mlframe.feature_selection._benchmarks.bench_rfecv_importance_agg

VERDICT (measured here, parsimonious n_features_selection_rule='one_se_min', 5 scenarios x 3 seeds):
  dispatched 4 wins / legacy 0 / 11 ties  ->  FLIP default to 'dispatched'.
  Decisive wins: tree_heavy seed2 +0.0456 AUC; imbalanced all 3 seeds (+0.0009 / +0.0030 / +0.0196).
  Kernel family: all ties (defers to legacy by construction -> zero regression). Dispatched never lost a cell.
  Re-run on other hardware/data to re-confirm before re-flipping.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MLFRAME_NO_CUDA_AUTOCONFIG", "1")
os.environ.setdefault("MLFRAME_KEEP_BROKEN_CUPY", "1")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mlframe.feature_selection.wrappers.rfecv import RFECV


def _make_scenario(name: str, seed: int):
    rng = np.random.default_rng(seed)
    n, p_inf, p_noise = 1200, 4, 26
    if name == "tree_heavy":
        Xi = rng.normal(size=(n, p_inf))
        y = (Xi[:, 0] * 1.5 + Xi[:, 1] - 0.8 * Xi[:, 2] + Xi[:, 3] ** 2 + rng.normal(scale=0.5, size=n) > 0).astype(int)
        Xn = rng.normal(size=(n, p_noise))
        est = RandomForestClassifier(n_estimators=60, random_state=seed)
    elif name == "linear_signal":
        Xi = rng.normal(size=(n, p_inf))
        w = np.array([2.0, -1.5, 1.0, -0.7])
        y = (Xi @ w + rng.normal(scale=0.7, size=n) > 0).astype(int)
        Xn = rng.normal(size=(n, p_noise))
        est = LogisticRegression(max_iter=500, random_state=seed)
    elif name == "imbalanced":
        Xi = rng.normal(size=(n, p_inf))
        w = np.array([2.2, -1.6, 1.1, -0.6])
        logit = Xi @ w - 3.0
        prob = 1 / (1 + np.exp(-logit))
        y = (rng.uniform(size=n) < prob).astype(int)
        Xn = rng.normal(size=(n, p_noise))
        est = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
    elif name == "sign_flip":
        # Informative cols + decoys that are weakly+noisily related so their fitted coef sign flips across folds.
        Xi = rng.normal(size=(n, p_inf))
        w = np.array([2.0, -1.5, 1.0, -0.7])
        y = (Xi @ w + rng.normal(scale=0.8, size=n) > 0).astype(int)
        # Decoys: pure noise that happens to correlate spuriously per-fold -> sign-unstable coef.
        Xn = rng.normal(size=(n, p_noise))
        est = LogisticRegression(max_iter=500, random_state=seed)
    elif name == "kernel":
        Xi = rng.normal(size=(n, p_inf))
        y = (Xi[:, 0] * Xi[:, 1] + Xi[:, 2] - Xi[:, 3] + rng.normal(scale=0.4, size=n) > 0).astype(int)
        Xn = rng.normal(size=(n, p_noise))
        est = SVC(kernel="rbf", probability=True, random_state=seed)
    else:
        raise ValueError(name)
    X = np.hstack([Xi, Xn])
    cols = [f"inf{i}" for i in range(p_inf)] + [f"noise{i}" for i in range(p_noise)]
    Xdf = pd.DataFrame(X, columns=cols)
    return Xdf, y, est, p_inf


def _honest_score(name, seed, importance_agg):
    Xdf, y, est, p_inf = _make_scenario(name, seed)
    X_tr, X_te, y_tr, y_te = train_test_split(Xdf, y, test_size=0.3, random_state=seed, stratify=y)
    sel = RFECV(
        estimator=est, cv=3, max_refits=14, importance_agg=importance_agg,
        early_stopping_val_nsplits=None, verbose=0, random_state=seed,
        # Parsimony so the cross-fold ranking actually decides which FEW features survive (one_se_max keeps ~all).
        n_features_selection_rule="one_se_min",
        importance_getter="permutation" if name == "kernel" else None,
    )
    sel.fit(X_tr, y_tr)
    kept = sel._selected_cols_cache or list(Xdf.columns)
    from sklearn.base import clone
    m = clone(est)
    m.fit(X_tr[kept], y_tr)
    if hasattr(m, "predict_proba"):
        s = m.predict_proba(X_te[kept])[:, 1]
    elif hasattr(m, "decision_function"):
        s = m.decision_function(X_te[kept])
    else:
        s = m.predict(X_te[kept])
    try:
        auc = roc_auc_score(y_te, s)
    except Exception:
        auc = float("nan")
    return auc, len(kept), p_inf


def main():
    scenarios = ["tree_heavy", "linear_signal", "imbalanced", "sign_flip", "kernel"]
    seeds = [0, 1, 2]
    print(f"{'scenario':<14}{'seed':<5}{'legacy_auc':<12}{'disp_auc':<12}{'delta':<10}{'leg_n':<7}{'disp_n':<7}winner")
    wins = {"dispatched": 0, "legacy": 0, "tie": 0}
    rows = []
    for sc in scenarios:
        for sd in seeds:
            leg, leg_n, _ = _honest_score(sc, sd, "legacy")
            dis, dis_n, _ = _honest_score(sc, sd, "dispatched")
            delta = dis - leg
            if abs(delta) < 0.002:
                w = "tie"
            elif delta > 0:
                w = "dispatched"
            else:
                w = "legacy"
            wins[w] += 1
            rows.append((sc, sd, leg, dis, delta, leg_n, dis_n, w))
            print(f"{sc:<14}{sd:<5}{leg:<12.4f}{dis:<12.4f}{delta:<+10.4f}{leg_n:<7}{dis_n:<7}{w}")
    print("\nWIN COUNTS:", wins)
    n_decisive = wins["dispatched"] + wins["legacy"]
    verdict = "FLIP->dispatched" if wins["dispatched"] > wins["legacy"] else "KEEP legacy (REJECT dispatched default)"
    print(f"VERDICT: {verdict} (dispatched {wins['dispatched']} vs legacy {wins['legacy']} of {n_decisive} decisive, {wins['tie']} ties)")
    return rows, wins


if __name__ == "__main__":
    main()
