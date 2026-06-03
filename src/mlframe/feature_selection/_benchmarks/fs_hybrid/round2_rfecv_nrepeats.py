"""Round-2 R2r-1 (now testable -- n_repeats surfaced): does the permutation n_repeats change the SELECTION?

R2r-1 (variance-aware n_repeats early-stop) is a SPEED lever that is correctness-AFFECTING (fewer repeats -> noisier
permutation importance -> possibly different ranking/selection). Discriminating test: fit RFECV(importance='permutation')
at n_repeats in {2,3,5} and compare selection Jaccard + downstream AUC vs n_repeats=5. If selection is stable across
repeat counts, an adaptive early-stop is SAFE (the importance ranking has already converged); if it drifts, the
early-stop is risky and must be gated. make_dataset, 3 seeds.
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth import make_dataset
from mlframe.feature_selection.wrappers import RFECV, FIConfig, SearchConfig


def fit_rfecv(X, y, n_repeats):
    r = RFECV(estimator=lgb.LGBMClassifier(n_estimators=150, num_leaves=31, learning_rate=0.06, n_jobs=-1, verbose=-1),
              cv=3, scoring=None, verbose=0, n_repeats=n_repeats,
              fi_config=FIConfig(importance_getter="permutation", n_features_selection_rule="one_se_min"),
              search_config=SearchConfig(max_refits=16, max_runtime_mins=3), random_state=0)
    r.fit(X, y)
    return [c for c in r.get_feature_names_out() if c in X.columns]


def auc(Xtr, Xte, ytr, yte, cols):
    if not cols:
        return float("nan")
    a = [roc_auc_score(yte, mk().fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1]) for mk in
         (lambda: lgb.LGBMClassifier(n_estimators=300, verbose=-1),
          lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
          lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)))]
    return round(float(np.mean(a)), 4)


def main():
    rows = []
    for sd in [0, 1, 2]:
        X, y, t = make_dataset(n_samples=5000, seed=sd)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=sd, stratify=y)
        ref = None
        for nr in (5, 3, 2):
            t0 = time.time(); s = fit_rfecv(Xtr, ytr, nr); dt = time.time() - t0
            if nr == 5:
                ref = set(s)
            jac = round(len(set(s) & ref) / max(1, len(set(s) | ref)), 3)
            rows.append(dict(seed=sd, n_repeats=nr, n=len(s), auc_mean=auc(Xtr, Xte, ytr, yte, s), jacc_vs_5=jac, fit_s=round(dt, 1)))
            print(f"sd{sd} n_repeats={nr} n={len(s):2d} auc={rows[-1]['auc_mean']} jacc_vs_5={jac} {dt:5.1f}s", flush=True)
    df = pd.DataFrame(rows)
    print("\n=== mean over seeds ===")
    print(df.groupby("n_repeats").agg(auc_mean=("auc_mean", "mean"), n=("n", "mean"), jacc_vs_5=("jacc_vs_5", "mean"), fit_s=("fit_s", "mean")).round(4).to_string())


if __name__ == "__main__":
    main()
