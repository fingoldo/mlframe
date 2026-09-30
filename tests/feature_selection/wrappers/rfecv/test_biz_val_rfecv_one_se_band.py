"""biz_value: the standard-error ``one_se_max`` band keeps materially fewer features than the legacy across-fold-std band at no OOS cost.

Multi-dataset evidence (13 datasets x 6 seeds x CatBoost/LightGBM/logistic-or-ridge, ``_benchmarks/bench_rfecv_one_se_rule.py``): paired vs the legacy band
dOOS = +0.0004 +- 0.0002 (W/T/L 47/151/36), 7.4 fewer features on average (-78 on p>>n CatBoost, -35 on many-noise LightGBM).  This is a fast single-regime
replay (400x200, 8 informative; over 5 seeds the SE band keeps 75 vs 105 features at +0.0025 AUC, measured 100 -> 75 on the two seeds used here, ratio 0.75): many noise columns, logistic model, selection rules replayed on ONE RFECV curve so the comparison is paired.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mlframe.feature_selection.wrappers import RFECV


def _replay(sel, rule):
    sel.n_features_selection_rule = rule
    cv = sel.cv_results_
    sel.select_optimal_nfeatures_(
        checked_nfeatures=cv["nfeatures"], cv_mean_perf=cv["cv_mean_perf"], cv_std_perf=cv["cv_std_perf"], feature_cost=sel.feature_cost, smooth_perf=sel.smooth_perf
    )
    return np.asarray(sel.support_, dtype=bool)


def test_biz_val_rfecv_one_se_band_se_keeps_fewer_features_at_equal_oos():
    X, y = make_classification(400, 200, n_informative=8, n_redundant=0, class_sep=1.5, random_state=0, shuffle=False)
    X = pd.DataFrame((X - X.mean(0)) / X.std(0), columns=[f"f{i}" for i in range(200)])
    n_legacy, n_se, auc_legacy, auc_se = [], [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in (2, 3):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=seed, stratify=y)
            sel = RFECV(estimator=LogisticRegression(C=0.3, max_iter=300), cv=3, random_state=seed, max_refits=8, verbose=0).fit(Xtr, ytr)
            for rule, n_out, auc_out in (("one_se_max_foldstd", n_legacy, auc_legacy), ("one_se_max", n_se, auc_se)):
                kept = [c for c, m in zip(X.columns, _replay(sel, rule)) if m]
                auc_out.append(roc_auc_score(yte, LogisticRegression(C=0.3, max_iter=300).fit(Xtr[kept], ytr).predict_proba(Xte[kept])[:, 1]))
                n_out.append(len(kept))
    assert np.mean(n_se) <= 0.85 * np.mean(n_legacy), (n_se, n_legacy)
    assert np.mean(auc_se) >= np.mean(auc_legacy) - 0.002, (auc_se, auc_legacy)
