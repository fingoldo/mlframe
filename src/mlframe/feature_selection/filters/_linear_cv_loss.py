"""Held-out CV loss of a scaled linear model, the one scorer shared by the pure-form side check and the compaction's linear-value check.

Log-loss of a scaled logistic model for a classification target, MAE of OLS otherwise: the scorers and the fold split the usability-aware retention commits with
(:func:`mlframe.feature_selection.filters._usability_aware_selection._usability_scorers`).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _standardise(x_tr: np.ndarray, x_va: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """``StandardScaler`` fitted on the train fold (population std, a constant column scaled by 1) applied to both folds."""
    mean = x_tr.mean(axis=0)
    scale = x_tr.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (x_tr - mean) / scale, (x_va - mean) / scale


def _direct_cv_loss(x: np.ndarray, y: np.ndarray, y_enc: Any, classification: bool, n_classes: Any, splits: list) -> "float | None":
    """The CV loss without the scikit-learn pipelines, or ``None`` when the case is not covered or a solve does not converge (the pipelines then decide).

    Binary classification: the scaled L2 logistic fit (inverse strength 1, intercept unpenalised) of ``LogisticRegression`` by the Newton solver of the accuracy gate, which converges to the
    optimum that lbfgs approaches to its 1e-4 tolerance (the losses agree to about 1e-8). Regression: ordinary least squares with an intercept (the scaling does not change it). A fit of a few
    columns on a few thousand rows took 20 ms in the pipeline and about 1 ms here."""
    losses = []
    if classification:
        if n_classes != 2:
            return None
        from ._fe_accuracy_gate_newton import fit_logistic_ridge_newton

        eps = np.finfo(np.float64).eps  # Intentional float64: the log-loss clip of the pipeline path
        for tr, va in splits:
            a_tr, a_va = _standardise(x[tr], x[va])
            fit = fit_logistic_ridge_newton(a_tr, np.asarray(y_enc)[tr], 1.0)
            if fit is None:
                return None
            w, b = fit
            p1 = np.clip(1.0 / (1.0 + np.exp(-(a_va @ w + b))), eps, 1.0 - eps)
            t = np.asarray(y_enc)[va]
            losses.append(float(-np.mean(np.where(t == 1, np.log(p1), np.log(1.0 - p1)))))
        return float(np.mean(losses))
    for tr, va in splits:
        a_tr = np.concatenate([x[tr], np.ones((len(tr), 1))], axis=1)
        coef = np.linalg.lstsq(a_tr, y[tr], rcond=None)[0]
        losses.append(float(np.mean(np.abs(y[va] - np.concatenate([x[va], np.ones((len(va), 1))], axis=1) @ coef))))
    return float(np.mean(losses))


def cv_loss(design: np.ndarray, y: np.ndarray, classification: bool, n_folds: int, seed: int) -> float:
    """Mean CV loss of a scaled linear model on the columns of ``design``; ``inf`` when it cannot be fit (non-finite design, one class)."""
    from sklearn.model_selection import KFold

    from ._usability_aware_selection import _scrub, _usability_scorers

    scorers = _usability_scorers(classification, y)
    if scorers is None:
        return float("inf")
    mk, logloss, y_enc, _n_classes = scorers
    x = _scrub(np.asarray(design, dtype=np.float64))  # Intentional float64: the retention gates score float64 columns
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if not np.isfinite(x).all() or float(np.std(x)) <= 1e-12:
        return float("inf")
    splits = list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(x))
    direct = _direct_cv_loss(x, y, y_enc, classification, _n_classes, splits)
    if direct is not None:
        return direct
    losses = []
    for tr, va in splits:
        model = mk()
        if classification:
            model.fit(x[tr], y_enc[tr])
            losses.append(logloss(y_enc[va], model.predict_proba(x[va])))
        else:
            model.fit(x[tr], y[tr])
            losses.append(float(np.mean(np.abs(y[va] - model.predict(x[va])))))
    return float(np.mean(losses))
