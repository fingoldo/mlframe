"""Closed-form ridge fit of the FE accuracy gate's regression probe.

The gate fits ``Ridge(alpha=1.0)`` on a few standardised columns once per fold and per candidate. The solution is the one linear system ``(Xc'Xc + alpha I) w = Xc'yc`` on the centred
columns with the intercept ``mean(y) - mean(X) w``, which is what scikit-learn's dense Cholesky solver computes; solving it here skips the estimator's input validation and parameter handling,
which dominate a fit this small. A non-finite result returns ``None`` and the caller keeps the scikit-learn fit.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg as _sla

#: From this many columns the normal equations are solved by Cholesky (they are symmetric positive definite: the penalty is on the diagonal): measured on the development host, ``cho_factor + cho_solve`` takes 0.06 ms
#: against 1.59 ms for ``np.linalg.solve`` at 100 columns, 2.1 against 7.4 ms at 400 (the two agree to 1e-18), and is slower below about 16 columns.
CHOLESKY_MIN_COLUMNS = 16


def fit_ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float) -> "tuple[np.ndarray, float] | None":
    """``(weights, intercept)`` of the ridge fit of ``y`` on ``X`` with penalty ``alpha`` (intercept unpenalised), or ``None`` when the system is singular or the result is not finite."""
    # Intentional float64: the normal equations square the conditioning of the columns, so they are formed and solved in double precision
    A = np.asarray(X, dtype=np.float64)
    # Intentional float64: normal equations square the conditioning of the columns
    t = np.asarray(y, dtype=np.float64).ravel()
    x_mean = A.mean(axis=0)
    y_mean = float(t.mean())
    Ac = A - x_mean
    gram = Ac.T @ Ac
    gram[np.diag_indices_from(gram)] += float(alpha)
    rhs = Ac.T @ (t - y_mean)
    w = None
    if gram.shape[0] >= CHOLESKY_MIN_COLUMNS:
        try:
            w = _sla.cho_solve(_sla.cho_factor(gram, lower=True, check_finite=False), rhs, check_finite=False)
        except (np.linalg.LinAlgError, _sla.LinAlgError):  # not positive definite (non-finite entries): the general solver decides
            w = None
    if w is None:
        try:
            w = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:  # best-effort: None is the singular signal; each caller then solves by least squares or ScaledRidge
            return None
    if not np.isfinite(w).all():
        return None
    return w, y_mean - float(x_mean @ w)


def fit_ridge_closed_form_multi(X: np.ndarray, Y: np.ndarray, alpha: float) -> "tuple[np.ndarray, np.ndarray] | None":
    """``(weights (columns, targets), intercepts (targets,))`` of the ridge fits of every column of ``Y`` on ``X``: the normal equations are formed and factorised once and solved for all targets together.

    Each column equals ``fit_ridge_closed_form(X, Y[:, j], alpha)`` (the same centred system); ``None`` when the system is singular or the result is not finite.
    """
    # Intentional float64: the normal equations square the conditioning of the columns, so they are formed and solved in double precision
    A = np.asarray(X, dtype=np.float64)
    # Intentional float64: normal equations square the conditioning of the columns
    T = np.asarray(Y, dtype=np.float64)
    if T.ndim == 1:
        T = T[:, None]
    x_mean = A.mean(axis=0)
    y_mean = T.mean(axis=0)
    Ac = A - x_mean
    gram = Ac.T @ Ac
    gram[np.diag_indices_from(gram)] += float(alpha)
    rhs = Ac.T @ (T - y_mean)
    w = None
    if gram.shape[0] >= CHOLESKY_MIN_COLUMNS:
        try:
            w = _sla.cho_solve(_sla.cho_factor(gram, lower=True, check_finite=False), rhs, check_finite=False)
        except (np.linalg.LinAlgError, _sla.LinAlgError):
            w = None
    if w is None:
        try:
            w = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:  # best-effort: a singular probe design means no ridge fit; the caller reads None as no candidate, the same outcome as a rejected fit
            return None
    if not np.isfinite(w).all():
        return None
    return w, y_mean - x_mean @ w
