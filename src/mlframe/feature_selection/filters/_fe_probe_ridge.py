"""Closed-form ridge fit of the FE accuracy gate's regression probe.

The gate fits ``Ridge(alpha=1.0)`` on a few standardised columns once per fold and per candidate. The solution is the one linear system ``(Xc'Xc + alpha I) w = Xc'yc`` on the centred
columns with the intercept ``mean(y) - mean(X) w``, which is what scikit-learn's dense Cholesky solver computes; solving it here skips the estimator's input validation and parameter handling,
which dominate a fit this small. A non-finite result returns ``None`` and the caller keeps the scikit-learn fit.
"""

from __future__ import annotations

import numpy as np


def fit_ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float) -> "tuple[np.ndarray, float] | None":
    """``(weights, intercept)`` of the ridge fit of ``y`` on ``X`` with penalty ``alpha`` (intercept unpenalised), or ``None`` when the system is singular or the result is not finite."""
    # Intentional float64: the normal equations square the conditioning of the columns, so they are formed and solved in double precision
    A = np.asarray(X, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64).ravel()
    x_mean = A.mean(axis=0)
    y_mean = float(t.mean())
    Ac = A - x_mean
    gram = Ac.T @ Ac
    gram[np.diag_indices_from(gram)] += float(alpha)
    try:
        w = np.linalg.solve(gram, Ac.T @ (t - y_mean))
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(w).all():
        return None
    return w, y_mean - float(x_mean @ w)
