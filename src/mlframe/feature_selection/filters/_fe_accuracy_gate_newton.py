"""Newton fit of the binary L2-penalised logistic probe used by the FE accuracy gate.

The gate fits ``LogisticRegression(C=1.0)`` on a few standardised columns once per fold and per candidate (about 14 ms per fit, almost all of it the scikit-learn and lbfgs call glue around a
four-column problem). The objective is strictly convex (a ridge term on the weights, free intercept), so its optimum is unique and a Newton iteration reaches it in a handful of steps at about a
twentieth of the cost. A fit that does not converge, or whose weights are not finite, returns ``None`` and the caller keeps the scikit-learn fit, so this path never decides what the exact one could not.
"""

from __future__ import annotations

import logging

import numpy as np
from numba import njit

logger = logging.getLogger(__name__)

_MAX_ITER = 30
_GRAD_TOL = 1e-8


def _objective(z: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    """Penalised negative log-likelihood ``sum(log(1 + e^z) - y z) + lam / 2 * |w|^2`` (intercept not penalised)."""
    return float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * lam * float(np.dot(w[:-1], w[:-1])))


@njit(cache=True, nogil=True)
def _newton_kernel(A: np.ndarray, y: np.ndarray, lam: float, max_iter: int, grad_tol: float, w_out: np.ndarray) -> bool:
    """The Newton iteration of :func:`fit_logistic_ridge_newton` in one GIL-free kernel; the optimum is written to ``w_out`` (weights, then the intercept); False when it does not converge."""
    n, p = A.shape
    w = np.zeros(p)
    z = np.zeros(n)
    f = n * np.log(2.0)  # sum of log(1 + e^0) - y * 0 at w = 0
    tol = grad_tol * max(1.0, float(n)) ** 0.5
    for _ in range(max_iter):
        grad = np.zeros(p)
        hess = np.zeros((p, p))
        for i in range(n):
            mu = 1.0 / (1.0 + np.exp(-z[i]))
            r = mu - y[i]
            s = mu * (1.0 - mu)
            for a in range(p):
                grad[a] += A[i, a] * r
                sa = s * A[i, a]
                for b in range(a + 1):
                    hess[a, b] += sa * A[i, b]
        for a in range(p - 1):
            grad[a] += lam * w[a]
        gmax = 0.0
        for a in range(p):
            if abs(grad[a]) > gmax:
                gmax = abs(grad[a])
        if gmax < tol:
            for a in range(p):
                w_out[a] = w[a]
            return True
        for a in range(p):
            for b in range(a):
                hess[b, a] = hess[a, b]
            hess[a, a] += 1e-12 + (lam if a < p - 1 else 0.0)
        step = np.linalg.solve(hess, grad)
        t = 1.0
        accepted = False
        for _ in range(12):
            w_new = w - t * step
            z_new = A @ w_new
            f_new = 0.0
            for i in range(n):
                zi = z_new[i]
                f_new += (max(zi, 0.0) + np.log1p(np.exp(-abs(zi)))) - y[i] * zi
            pen = 0.0
            for a in range(p - 1):
                pen += w_new[a] * w_new[a]
            f_new += 0.5 * lam * pen
            if np.isfinite(f_new) and f_new <= f + 1e-11 * abs(f):
                accepted = True
                break
            t *= 0.5
        if not accepted:
            return False
        w = w_new
        z = z_new
        f = f_new
    return False


def fit_logistic_ridge_newton(X: np.ndarray, y01: np.ndarray, C: float = 1.0) -> "tuple[np.ndarray, float] | None":
    """``(weights, intercept)`` of the L2 logistic fit with inverse strength ``C`` on columns ``X`` and 0/1 labels, or ``None`` when Newton does not converge to a finite optimum (the GIL-free kernel form of the NumPy twin)."""
    # Intentional float64: the optimum is solved in double so the stop test (gradient 1e-8) is meaningful for float32 inputs
    A = np.ascontiguousarray(np.concatenate([np.asarray(X, dtype=np.float64), np.ones((X.shape[0], 1), dtype=np.float64)], axis=1))
    y = np.ascontiguousarray(y01, dtype=np.float64)  # Intentional float64: the Newton solve is in double precision
    w = np.zeros(A.shape[1])
    try:
        converged = _newton_kernel(A, y, 1.0 / float(C), int(_MAX_ITER), float(_GRAD_TOL), w)
    except np.linalg.LinAlgError:
        logger.debug("logistic Newton probe: singular Hessian, the scikit-learn fit decides")
        return None
    if not converged or not np.isfinite(w).all():
        return None
    return w[:-1].copy(), float(w[-1])


def fit_logistic_ridge_newton_numpy(X: np.ndarray, y01: np.ndarray, C: float = 1.0) -> "tuple[np.ndarray, float] | None":
    """NumPy twin of :func:`fit_logistic_ridge_newton` (the reference the kernel is tested against): ``(weights, intercept)`` of the L2 logistic fit with inverse strength ``C`` on columns ``X`` and 0/1 labels, or ``None`` when Newton does not converge to a finite optimum."""
    # Intentional float64: the optimum is solved in double so the stop test (gradient 1e-8) is meaningful for float32 inputs
    A = np.concatenate([np.asarray(X, dtype=np.float64), np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    # Intentional float64: the Newton solve is in double precision (normal equations of a few columns)
    y = np.asarray(y01, dtype=np.float64)
    p = A.shape[1]
    lam = 1.0 / float(C)
    reg = np.full(p, lam)
    reg[-1] = 0.0
    w: np.ndarray = np.zeros(p)
    z: np.ndarray = A @ w
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
            z_new: np.ndarray = A @ w_new
            f_new = _objective(z_new, y, w_new, lam)
            if np.isfinite(f_new) and f_new <= f:
                break
            t *= 0.5
        else:
            return None
        w, z, f = w_new, z_new, f_new
    return None
