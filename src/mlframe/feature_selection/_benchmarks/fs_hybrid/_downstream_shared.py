"""Shared downstream-AUC evaluators for the fs_hybrid bench family (round2_*/round3_* scripts): small
utilities independently duplicated across those scripts, consolidated here so a fix can't silently
drift out of sync across copies. Each benchmark script stays independently runnable from this same
``fs_hybrid/`` directory -- only the literal duplicated bodies move here.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def lgb_logit_knn_mean_auc(Xtr, Xte, ytr, yte, cols) -> float:
    """Mean held-out AUC across LGBM (300 trees) / StandardScaler+LogisticRegression /
    StandardScaler+kNN(25) on the ``cols`` subset -- the round2 RFECV bench pair's shared downstream metric.
    NaN if ``cols`` is empty."""
    if not cols:
        return float("nan")
    a = [
        roc_auc_score(yte, mk().fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
        for mk in (
            lambda: lgb.LGBMClassifier(n_estimators=300, verbose=-1),
            lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
        )
    ]
    return round(float(np.mean(a)), 4)


def boruta_single(X, y):
    """Fit a single BorutaShap run (RF, 80 trees, 60 trials, percentile=95, seed=0) and return the
    selected columns present in ``X`` -- the round2 boruta/hetero-skillweight bench pair's shared baseline call."""
    from sklearn.ensemble import RandomForestClassifier

    from mlframe.feature_selection.boruta_shap import BorutaShap

    b = BorutaShap(
        model=RandomForestClassifier(n_estimators=80, n_jobs=-1, random_state=0),
        importance_measure="gini",
        classification=True,
        n_trials=60,
        percentile=95,
        verbose=False,
        random_state=0,
    )
    b.fit(X, y)
    return [c for c in b.selected_features_ if c in X.columns]


def entropy(labels) -> float:
    """Discrete Shannon entropy (nats) of an integer-coded label array -- shared by the round4
    tentative_to_cmi / su_seeded_interactions bench pair."""
    _, cnt = np.unique(labels, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log(p)).sum())


def load_scene(n_rows):
    """Load the OpenML 'scene' dataset (binarised to the majority-class target), optionally subsampled
    to ``n_rows`` rows -- shared by the round4_scene_cprofile / round4_fs_campaign_profile bench pair."""
    import pandas as pd
    from sklearn.datasets import fetch_openml

    d = fetch_openml(name="scene", version=1, as_frame=True, parser="auto")
    X = d.data.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X.columns = [f"f{i}" for i in range(X.shape[1])]
    y = pd.Series(pd.factorize(d.target)[0])
    y = (y == y.value_counts().idxmax()).astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)
    if n_rows < len(X):
        idx = np.random.default_rng(0).choice(len(X), size=n_rows, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    return X, y


def build_products(X, pairs) -> dict:
    """Return ``{name: product array}`` for the given ``(a, b)`` pairs present in ``X`` -- shared by
    the round4 fe_accept/fe_accept_frugal bench pair."""
    prods = {}
    for i, (a, b) in enumerate(pairs):
        if a in X.columns and b in X.columns:
            prods[f"prod_{i}"] = (X[a].values * X[b].values).astype(np.float64)
    return prods


def checkpoint_at(progress_path: str, msg: str) -> None:
    """Append a ``HH:MM:SS``-timestamped ``msg`` to ``progress_path`` and echo it to stdout: the
    checkpoint helper shared by the round4 fe_accept/fe_accept_frugal bench pair (both use the same
    ``D:/Temp/fe_accept_progress.txt`` path)."""
    with open(progress_path, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(f"  [ckpt] {msg}", flush=True)


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
