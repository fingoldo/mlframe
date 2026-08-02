"""Shared downstream-AUC evaluators for the fs_hybrid bench family (round2_*/round3_* scripts): small
utilities independently duplicated across those scripts, consolidated here so a fix can't silently
drift out of sync across copies. Each benchmark script stays independently runnable from this same
``fs_hybrid/`` directory -- only the literal duplicated bodies move here.
"""
from __future__ import annotations

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def downstream_on_cols(Xtr, Xte, ytr, yte, cols):
    """Fit LGBM / logistic / kNN on the ``cols`` subset of ``Xtr``/``Xte`` and return each model's test AUC (NaN triple if ``cols`` is empty)."""
    if not cols:
        return {"lgbm": float("nan"), "logit": float("nan"), "knn": float("nan")}
    o = {}
    o["lgbm"] = roc_auc_score(yte, lgb.LGBMClassifier(n_estimators=300, verbose=-1).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    o["logit"] = roc_auc_score(yte, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    o["knn"] = roc_auc_score(yte, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    return {k: round(float(v), 4) for k, v in o.items()}


def downstream_on_matrix(Ztr, Zte, ytr, yte):
    """Fit LGBM / logistic / kNN directly on already-selected matrices ``Ztr``/``Zte`` and return each model's test AUC."""
    o = {}
    o["lgbm"] = roc_auc_score(yte, lgb.LGBMClassifier(n_estimators=300, verbose=-1).fit(Ztr, ytr).predict_proba(Zte)[:, 1])
    o["logit"] = roc_auc_score(yte, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Ztr, ytr).predict_proba(Zte)[:, 1])
    o["knn"] = roc_auc_score(yte, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)).fit(Ztr, ytr).predict_proba(Zte)[:, 1])
    return {k: round(float(v), 4) for k, v in o.items()}
