"""Round-2 R2r-7: does reducing the RFECV interior-CV fold count change the SELECTION / downstream AUC?

R2r-7 (early-stopped / reduced-fold interior CV) is a SPEED lever that "changes the MBH search trajectory (risk)".
Discriminating test: fit RFECV at cv in {2,3,5} and compare downstream AUC + selection Jaccard vs the cv=3 default.
If selection + AUC are stable across folds -> reduced folds is a SAFE speed lever; if they drift -> it is
correctness-affecting (the flagged risk) and must be gated. Impurity importance (fast; the question is fold count).
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

SEEDS = [0, 1, 2]


def fit_rfecv(X, y, cv):
    r = RFECV(estimator=lgb.LGBMClassifier(n_estimators=150, num_leaves=31, learning_rate=0.06, n_jobs=-1, verbose=-1),
              cv=cv, scoring=None, verbose=0,
              fi_config=FIConfig(importance_getter="feature_importances_", n_features_selection_rule="one_se_min"),
              search_config=SearchConfig(max_refits=14, max_runtime_mins=2), random_state=0)
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


def jacc(a, b):
    a, b = set(a), set(b)
    return round(len(a & b) / max(1, len(a | b)), 3)


def main():
    rows = []
    for sd in SEEDS:
        X, y, t = make_dataset(n_samples=5000, seed=sd)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=sd, stratify=y)
        sels = {}
        for cv in (2, 3, 5):
            t0 = time.time(); s = fit_rfecv(Xtr, ytr, cv); dt = time.time() - t0
            sels[cv] = s
            rows.append(dict(seed=sd, cv=cv, n=len(s), auc_mean=auc(Xtr, Xte, ytr, yte, s),
                             jacc_vs_cv3=jacc(s, sels.get(3, s)), fit_s=round(dt, 1)))
            print(f"sd{sd} cv={cv} n={len(s):2d} auc={rows[-1]['auc_mean']} jacc_vs_cv3={rows[-1]['jacc_vs_cv3']} {dt:5.1f}s", flush=True)
    df = pd.DataFrame(rows)
    print("\n=== mean over seeds ===")
    print(df.groupby("cv").agg(auc_mean=("auc_mean", "mean"), n=("n", "mean"), jacc_vs_cv3=("jacc_vs_cv3", "mean"), fit_s=("fit_s", "mean")).round(4).to_string())


if __name__ == "__main__":
    main()
