"""Cyclic coordinate descent for the elastic net and the lasso, in numba, in place of ``sklearn.linear_model.ElasticNet`` / ``Lasso``.

The objective is scikit-learn's, ``1/(2 n) ||y - X w - b||^2 + alpha l1_ratio ||w||_1 + alpha (1 - l1_ratio) / 2 ||w||^2`` with a free intercept ``b`` (the data are centred), and the update of a
coordinate is the same soft-threshold step, visited in the same cyclic order, so the iterates are scikit-learn's. The stopping rule is stricter: the loop ends when no coefficient moves by more than
``tol`` times the largest coefficient magnitude (at least 1), so the result is the optimum to that tolerance where scikit-learn stops at a duality gap of ``1e-4 * ||y||^2``.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _column_sq_sums(Xc):
    """Sum of squares of every column of ``Xc``."""
    n, p = Xc.shape
    sq = np.zeros(p)
    for j in range(p):
        s = 0.0
        for i in range(n):
            s += Xc[i, j] * Xc[i, j]
        sq[j] = s
    return sq


@njit(cache=True, nogil=True)
def _add_scaled_column(resid, Xc, j, scale):
    """``resid += scale * Xc[:, j]`` in place."""
    for i in range(resid.shape[0]):
        resid[i] += Xc[i, j] * scale


@njit(cache=True, nogil=True)
def _column_dot(Xc, j, resid):
    """``Xc[:, j] . resid``."""
    rho = 0.0
    for i in range(resid.shape[0]):
        rho += Xc[i, j] * resid[i]
    return rho


@njit(cache=True, nogil=True)
def _soft_threshold_step(rho, l1, denom):
    """The coordinate update ``S(rho, l1) / denom`` of the elastic net (``S`` the soft threshold)."""
    if rho > l1:
        return (rho - l1) / denom
    if rho < -l1:
        return (rho + l1) / denom
    return 0.0


@njit(cache=True, nogil=True)
def _cd_kernel(Xc, yc, alpha, l1_ratio, max_iter, tol, w):
    """Coordinate descent on the centred design ``Xc`` and target ``yc``; ``w`` holds the start and receives the solution. Returns the number of sweeps made and whether it converged."""
    n, p = Xc.shape
    sq = _column_sq_sums(Xc)
    resid = yc.copy()
    for j in range(p):
        if w[j] != 0.0:
            _add_scaled_column(resid, Xc, j, -w[j])
    l1 = n * alpha * l1_ratio
    l2 = n * alpha * (1.0 - l1_ratio)
    for sweep in range(max_iter):
        max_move = 0.0
        max_w = 0.0
        for j in range(p):
            if sq[j] == 0.0:
                continue
            old = w[j]
            if old != 0.0:
                _add_scaled_column(resid, Xc, j, old)
            new = _soft_threshold_step(_column_dot(Xc, j, resid), l1, sq[j] + l2)
            w[j] = new
            if new != 0.0:
                _add_scaled_column(resid, Xc, j, -new)
            max_move = max(max_move, abs(new - old))
            max_w = max(max_w, abs(new))
        if max_move <= tol * max(1.0, max_w):
            return sweep + 1, True
    return max_iter, False


@njit(cache=True, nogil=True)
def _cd_gram_kernel(G, q, alpha, l1_ratio, n, max_iter, tol, w):
    """The same coordinate descent on the Gram matrix ``G = Xc^T Xc`` and ``q = Xc^T yc``: a coordinate costs O(p) instead of O(n), the update, the cyclic order and the stopping rule are those of
    :func:`_cd_kernel`. ``w`` holds the start and receives the solution."""
    p = G.shape[0]
    grad = q.copy()  # q - G w
    for j in range(p):
        if w[j] != 0.0:
            for k in range(p):
                grad[k] -= G[j, k] * w[j]
    l1 = n * alpha * l1_ratio
    l2 = n * alpha * (1.0 - l1_ratio)
    for sweep in range(max_iter):
        max_move = 0.0
        max_w = 0.0
        for j in range(p):
            gjj = G[j, j]
            if gjj == 0.0:
                continue
            old = w[j]
            new = _soft_threshold_step(grad[j] + gjj * old, l1, gjj + l2)
            w[j] = new
            delta = new - old
            if delta != 0.0:
                for k in range(p):
                    grad[k] -= G[j, k] * delta
            move = abs(delta)
            if move > max_move:
                max_move = move
            if abs(new) > max_w:
                max_w = abs(new)
        if max_move <= tol * max(1.0, max_w):
            return sweep + 1, True
    return max_iter, False


_GRAM_MAX_COLUMNS = 4000
"""Widest design that takes the Gram form (a (p, p) float64 matrix: 128 MB at 4000); wider ones keep the residual form."""


def elastic_net_coef(X: np.ndarray, y: np.ndarray, alpha: float, l1_ratio: float = 1.0, *, max_iter: int = 5000, tol: float = 1e-9) -> "tuple[np.ndarray, bool]":
    """``(coefficients, converged)`` of the elastic net (``l1_ratio=1``: the lasso) of ``y`` on the columns of ``X``, with an unpenalised intercept; the columns are NOT rescaled here."""
    # Intentional float64: the coordinate updates accumulate sums over all rows, so the design is held in double precision
    A = np.asarray(X, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64).ravel()
    Xc = np.ascontiguousarray(A - A.mean(axis=0))
    yc = t - t.mean()
    w = np.zeros(A.shape[1])
    n, p = Xc.shape
    if n > p and p <= _GRAM_MAX_COLUMNS:
        # more rows than columns: one BLAS product gives every inner product the sweeps need (the choice scikit-learn's ``precompute='auto'`` makes), and a coordinate then costs O(p), not O(n)
        _, converged = _cd_gram_kernel(Xc.T @ Xc, Xc.T @ yc, float(alpha), float(l1_ratio), float(n), int(max_iter), float(tol), w)
    else:
        _, converged = _cd_kernel(Xc, yc, float(alpha), float(l1_ratio), int(max_iter), float(tol), w)
    return w, bool(converged)
