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


class _DeviceFitFallbackError(Exception):  # audit-exempt: unraised_exception_class -- raised as fallback_error() by _usability_greedy_clf_batched through a parameter; the scanner cannot follow it
    """Raised by the batched device fit when a Newton fit does not converge: the per-column scorer decides instead."""


def cv_loss_columns_device(columns: "list[np.ndarray]", y: np.ndarray, classification: bool, n_folds: int, seed: int) -> "list[float] | None":
    """:func:`cv_loss` of every single column of ``columns`` at once on the device (binary classification only), as a list in the order of ``columns`` (``inf`` where :func:`cv_loss` gives ``inf``);
    ``None`` when the batched route does not apply (regression, more than two classes, no usable device, a fold with one class, a fit that does not converge) and the columns are scored one by one.

    The fit is the scaled L2 logistic fit of :func:`_direct_cv_loss` (inverse strength 1, intercept unpenalised) run by ``logistic_cv_newton``: one block per column, the train standardization, the Newton
    iteration to the step tolerance 1e-9 and the validation log-loss all inside the kernel, one launch a fold (about 30 small fits of 100000 rows each took 0.2 s through the host pool)."""
    if not classification or not columns:
        return None
    try:
        from ._gpu_policy import cuda_available_for_run, gpu_globally_disabled

        if gpu_globally_disabled() or not cuda_available_for_run():
            return None
        import cupy as cp
        from sklearn.model_selection import KFold

        from ._usability_aware_selection import _scrub, _usability_scorers
        from ._usability_greedy_clf_batched import cv_candidates_binary

        scorers = _usability_scorers(True, y)
        if scorers is None:
            return [float("inf")] * len(columns)
        _mk, _logloss, y_enc, n_classes = scorers
        if n_classes != 2:
            return None
        y_enc = np.asarray(y_enc)
        n = int(y_enc.shape[0])
        out = [float("inf")] * len(columns)
        usable: list = []
        cols: list = []
        for i, c in enumerate(columns):
            x = _scrub(np.asarray(c, dtype=np.float64)).reshape(-1)  # Intentional float64: the retention gates score float64 columns
            if x.shape[0] != n or not np.isfinite(x).all() or float(np.std(x)) <= 1e-12:
                continue
            usable.append(i)
            cols.append(x)
        if not cols:
            return out
        design = np.ascontiguousarray(np.stack(cols, axis=1))
        folds = []
        for tr, va in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(design):
            if np.unique(y_enc[tr]).size < 2:
                return None
            folds.append((cp.asarray(tr.astype(np.int64)), cp.asarray(va.astype(np.int64)), True))
        loss = cv_candidates_binary(
            cp, cp.asarray(design), [], list(range(len(cols))), folds, cp.asarray(np.ascontiguousarray(y_enc, dtype=np.int64)), dtype=np.float64, newton_max_iter=200, tol=1e-9, lam=1.0,  # Intentional float64: the device log-loss runs in double, see the note on the call
            logloss_eps=float(np.finfo(np.float64).eps), fallback_error=_DeviceFitFallbackError,  # Intentional float64: the log-loss clip is the float64 epsilon
        )  # Intentional float64: the log-loss clip of the pipeline path
        for j, i in enumerate(usable):
            out[i] = float(np.mean(loss[j]))
        return out
    except _DeviceFitFallbackError:  # best-effort: the device fit's own fallback signal; the caller uses the per-column scorer, which gives the same verdicts
        return None
    except Exception as exc:  # best-effort: the per-column scorer gives the same verdicts
        import logging

        logging.getLogger(__name__).warning("batched device CV loss failed (%s: %s); the columns are scored one by one", type(exc).__name__, exc)
        return None


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
            from ._usability_greedy_steps import _full_class_proba

            losses.append(logloss(y_enc[va], _full_class_proba(model, model.predict_proba(x[va]), int(_n_classes))))
        else:
            model.fit(x[tr], y[tr])
            losses.append(float(np.mean(np.abs(y[va] - model.predict(x[va])))))
    return float(np.mean(losses))
