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

    REGRESSION (default): in-sample R^2 >= ``thresh`` (LinearRegression) -- a low residual variance means a
    nonlinear pass cannot add much, so it is skipped.

    CLASSIFICATION: always returns ``False`` (keep the passes). A high raw-only logistic AUC/accuracy does
    NOT imply the absence of exploitable nonlinear/discrete structure -- a linearly-SEPARABLE target can
    still hold a clean composite the operators recover (canonical: y=argmax(a,b,c) is high-accuracy from the
    raws yet argmax__a__b__c is a genuine selectable composite), so an in-sample classification score is not
    a licence to skip an enabled discrete-structural operator. ``clf_thresh`` is retained for back-compat
    only. Returns ``False`` on any error (run the passes -- correctness over the optimisation)."""
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
        if is_clf:
            # CLASSIFICATION: always keep the operators (return False). A high raw-only logistic
            # AUC/accuracy does NOT mean there is no exploitable nonlinear/discrete structure -- a
            # linearly-SEPARABLE target can still hold a clean composite the operators recover
            # (canonical: y=argmax(a,b,c) -- raw-only logistic accuracy is high, yet argmax__a__b__c
            # is a genuine selectable composite). The in-sample classification score is NOT a licence
            # to skip an enabled discrete-structural operator, so the R^2-style shortcut is regression-only.
            return False
        else:
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

            # FE_ORCH_BUDGET-1 fix: an IN-SAMPLE R^2 has no columns-vs-rows guard --
            # empirically confirmed on pure noise (n=2000, matching this function's own max_rows default)
            # that in-sample R^2 crosses the default 0.92 threshold purely from overfitting once p numeric
            # raw columns exceeds ~95% of n (p=1900,n=2000 -> R^2=0.936 with ZERO real linear signal). MRMR
            # is a wide-p feature-selection tool, so this shape is plausible, not exotic, and a false-
            # positive here silently disables the discrete-structural operators for the WHOLE fit. Use
            # held-out (K-fold CV) R^2 instead: pure noise gives a near-zero-or-negative CV R^2 regardless
            # of p (each fold overfits its own train split but generalizes poorly to the held-out fold), so
            # only a GENUINE linear relationship clears the threshold. Bounded cost: still O(0.1s)-class
            # (few folds x one LinearRegression fit each on the same subsampled Xn/yy).
            n_splits = min(5, max(2, Xn.shape[0] // 50))
            if n_splits < 2 or Xn.shape[0] < 2 * n_splits:
                return False  # too few rows to CV meaningfully -- keep the passes (correctness over speed)
            from sklearn.model_selection import KFold, cross_val_score

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
            pipe = make_pipeline(StandardScaler(), LinearRegression())
            cv_r2 = cross_val_score(pipe, Xn, yy, cv=kf, scoring="r2")
            r2 = float(np.mean(cv_r2))
            return r2 >= thresh
    except Exception:
        return False


__all__ = ["raws_linearly_explain_y"]
