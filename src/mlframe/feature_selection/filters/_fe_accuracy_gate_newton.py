"""Newton fit of the binary L2-penalised logistic probe used by the FE accuracy gate.

The gate fits ``LogisticRegression(C=1.0)`` on a few standardised columns once per fold and per candidate (about 14 ms per fit, almost all of it the scikit-learn and lbfgs call glue around a
four-column problem). The objective is strictly convex (a ridge term on the weights, free intercept), so its optimum is unique and a Newton iteration reaches it in a handful of steps at about a
twentieth of the cost. A fit that does not converge, or whose weights are not finite, returns ``None`` and the caller keeps the scikit-learn fit, so this path never decides what the exact one could not.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_MAX_ITER = 30
_GRAD_TOL = 1e-8


def _objective(z: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    """Penalised negative log-likelihood ``sum(log(1 + e^z) - y z) + lam / 2 * |w|^2`` (intercept not penalised)."""
    return float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * lam * float(np.dot(w[:-1], w[:-1])))


def fit_logistic_ridge_newton(X: np.ndarray, y01: np.ndarray, C: float = 1.0) -> "tuple[np.ndarray, float] | None":
    """``(weights, intercept)`` of the L2 logistic fit with inverse strength ``C`` on columns ``X`` and 0/1 labels, or ``None`` when Newton does not converge to a finite optimum."""
    # Intentional float64: the optimum is solved in double so the stop test (gradient 1e-8) is meaningful for float32 inputs
    A = np.concatenate([np.asarray(X, dtype=np.float64), np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    # Intentional float64: the Newton solve is in double precision (normal equations of a few columns)
    y = np.asarray(y01, dtype=np.float64)
    p = A.shape[1]
    lam = 1.0 / float(C)
    reg = np.full(p, lam)
    reg[-1] = 0.0
    w = np.zeros(p)
    z = A @ w
    f = _objective(z, y, w, lam)
    for _ in range(_MAX_ITER):
        mu = 1.0 / (1.0 + np.exp(-z))
        grad = A.T @ (mu - y) + reg * w
        if float(np.max(np.abs(grad))) < _GRAD_TOL * max(1.0, float(A.shape[0])) ** 0.5:
            return (w[:-1].copy(), float(w[-1])) if np.isfinite(w).all() else None
        s = mu * (1.0 - mu)
        hess = (A.T * s) @ A
        hess[np.diag_indices(p)] += reg + 1e-12
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            logger.debug("logistic Newton probe: singular Hessian, the scikit-learn fit decides")
            return None
        t = 1.0
        for _ in range(12):
            w_new = w - t * step
            z_new = A @ w_new
            f_new = _objective(z_new, y, w_new, lam)
            if np.isfinite(f_new) and f_new <= f:
                break
            t *= 0.5
        else:
            return None
        w, z, f = w_new, z_new, f_new
    return None
