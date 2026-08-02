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


def checkpoint(msg: str) -> None:
    """Append ``msg`` to the shared fe_richops progress log and echo it to stdout -- the checkpoint
    helper duplicated across the fe_richops control/main bench pair."""
    with open(r"D:/Temp/fe_ops_progress.txt", "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


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


def make_ckpt_writer(path):
    """Build a ``ck(m)`` checkpoint-line appender: each call opens ``path`` in append mode and writes one line."""

    def ck(m):
        with open(path, "a") as f:
            f.write(m + "\n")

    return ck


def no_sweep_get_or_tune(self, kernel_name, *, dims, tuner, axes, fallback, **kw):
    """Monkeypatch target for ``KernelTuningCache.get_or_tune``, forcing the always-fallback (no-sweep) path: never
    tunes/sweeps, just calls ``fallback`` -- zero-arg first, falling back to dim-keyword then dim-positional so a
    fallback needing the dims never raises a TypeError."""
    if not callable(fallback):
        return fallback
    try:
        return fallback()
    except TypeError:
        try:
            return fallback(**dims)
        except TypeError:
            return fallback(*dims.values())


def mrmr_sel_transform(self, X):
    """Shared ``_Sel.transform`` body for the fs_hybrid MRMR-wrapper adapters: rename the fitted MRMR's
    output columns to the ``fit``-time-computed ``self.ren_`` mapping (raw cols kept, engineered cols
    made LightGBM-safe). Bound as a class attribute (``transform = mrmr_sel_transform``) on each local
    ``_Sel`` class, which relies on ``self.m_`` (the fitted MRMR instance) and ``self.ren_`` (the rename
    map) always being set in ``fit`` under those exact attribute names.
    """
    df = self.m_.transform(X).copy()
    df.columns = [self.ren_[c] for c in df.columns]
    return df
