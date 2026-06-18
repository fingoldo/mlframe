"""Cheap "do the raws already explain y linearly?" probe -- a necessary-condition gate for the
expensive nonlinear-FE passes (the discrete-structural operators and the pure-form retention).

Several FE passes hunt for NONLINEAR / interaction structure (regime-switch conditional gates, modular
/ gcd integer lattices, binned-numeric aggregates, pure pair interactions). They are expensive (MI-kernel
scans over many candidate combos) and run by default on every fit -- but on an ADDITIVE-LINEAR regression
target there is simply no such structure to find, so the scans burn time and recover nothing (measured:
~58% of an additive-regression fit went to the conditional-gate + binned-agg scans, and the pure-form
retention's pool build was ~88% before it was gated). A single plain linear fit on the raw columns is a
cheap necessary condition: if it already explains y well, no nonlinear pass can add anything, so skip them.

Regression-only by construction: an R^2 gate is meaningless for a classification target, so a
classification y always returns ``False`` (the operators must still run there -- they capture class
structure a linear regression R^2 cannot see). Best-effort: any failure returns ``False`` (run the
passes -- correctness over the optimisation).
"""
from __future__ import annotations

from typing import Any

import numpy as np


def raws_linearly_explain_y(
    X: Any, y: Any, *, thresh: float = 0.92, clf_thresh: float = 0.92,
    max_rows: int = 2000, seed: int = 0,
) -> bool:
    """True iff a plain (standardised) LINEAR model on the numeric raw columns of ``X`` already explains
    ``y`` well -- i.e. there is no nonlinear/interaction structure left for a downstream nonlinear-FE pass
    to recover, so it can be skipped. Row-subsamples to ``max_rows`` so the probe stays ~O(0.1s).

    REGRESSION (default): in-sample R^2 >= ``thresh`` (LinearRegression). Behaviour unchanged from the
    original regression-only contract.

    CLASSIFICATION (2026-06-18): a logistic model (StandardScaler+LogisticRegression) is fitted on the raw
    numeric columns and the gate fires on a held-in CLASSIFICATION metric instead of R^2 -- binary AUC (or
    multiclass accuracy) >= ``clf_thresh``. A polynomial / interaction classification target (the recovered
    pure form lifts AUC) leaves the raw-only logistic AUC well below ``clf_thresh`` -> the passes still
    fire; an already-linearly-separable target clears it -> they are skipped. Returns ``False`` on any
    error (run the passes -- correctness over the optimisation)."""
    try:
        import pandas as pd
        from ._fe_accuracy_gate import infer_classification

        yraw = np.asarray(y.to_numpy() if hasattr(y, "to_numpy") else y).ravel()
        if not isinstance(X, pd.DataFrame):
            return False
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c].dtype)]
        if not num_cols:
            return False
        n = yraw.shape[0]
        if n != len(X) or n < 8:
            return False

        is_clf = infer_classification(yraw)
        if not is_clf:
            # REGRESSION path -- byte-identical to the original contract.
            if yraw.dtype.kind not in "fiu":
                return False  # non-numeric target -> classification-like; keep the passes
            yc = yraw.astype(np.float64)
            if not np.isfinite(yc).all():
                return False
            if n > max_rows:
                rg = np.random.default_rng(int(seed) + 11)
                idx = np.sort(rg.choice(n, size=max_rows, replace=False))
                Xn = X.iloc[idx][num_cols].to_numpy(dtype=np.float64, copy=False)
                yy = yc[idx]
            else:
                Xn = X[num_cols].to_numpy(dtype=np.float64, copy=False)
                yy = yc
            Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline

            r2 = float(make_pipeline(StandardScaler(), LinearRegression()).fit(Xn, yy).score(Xn, yy))
            return r2 >= thresh

        # CLASSIFICATION path -- a logistic model + a classification metric gate.
        classes, y_enc = np.unique(yraw, return_inverse=True)
        if classes.size < 2:
            return False  # degenerate single-class -> keep the passes
        if n > max_rows:
            rg = np.random.default_rng(int(seed) + 11)
            idx = np.sort(rg.choice(n, size=max_rows, replace=False))
            Xn = X.iloc[idx][num_cols].to_numpy(dtype=np.float64, copy=False)
            yy = y_enc[idx]
        else:
            Xn = X[num_cols].to_numpy(dtype=np.float64, copy=False)
            yy = y_enc
        if np.unique(yy).size < 2:
            return False
        Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.metrics import roc_auc_score, accuracy_score

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)).fit(Xn, yy)
        if classes.size == 2:
            score = float(roc_auc_score(yy, clf.predict_proba(Xn)[:, 1]))
        else:
            score = float(accuracy_score(yy, clf.predict(Xn)))
        return score >= clf_thresh
    except Exception:
        return False


__all__ = ["raws_linearly_explain_y"]
