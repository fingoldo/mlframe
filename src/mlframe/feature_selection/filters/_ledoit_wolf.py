"""Ledoit-Wolf shrinkage covariance in numpy, in place of ``sklearn.covariance.LedoitWolf``.

The estimate is scikit-learn's: the sample covariance of the centred columns shrunk toward ``mu I`` (``mu`` the mean variance) by the Ledoit-Wolf coefficient ``beta / delta`` with ``beta`` clipped to
``delta``, and the precision matrix is the pseudo-inverse of the (symmetric) covariance, as ``get_precision`` computes it.
"""

from __future__ import annotations

import numpy as np


def ledoit_wolf_fit(X: np.ndarray) -> "tuple[np.ndarray, np.ndarray, np.ndarray, float]":
    """``(location, covariance, precision, shrinkage)`` of the rows of ``X`` (``(n, p)``): the column means, the shrunk covariance, its pseudo-inverse and the shrinkage coefficient in ``[0, 1]``."""
    # Intentional float64: the shrinkage sums fourth powers of the data, which overflow single precision for columns beyond ~1e10 and lose digits well before
    A = np.asarray(X, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"ledoit_wolf_fit: expected a 2-D array, got shape {A.shape}")
    n, p = A.shape
    location = A.mean(axis=0)
    C = A - location
    emp_cov = C.T @ C / n
    mu = float(np.trace(emp_cov)) / p
    if p == 1:
        shrinkage = 0.0
    else:
        sq = C**2
        emp_cov_trace = sq.sum(axis=0) / n
        beta_acc = float((sq.T @ sq).sum())
        delta_acc = float(((C.T @ C) ** 2).sum()) / n**2
        beta_raw = 1.0 / (p * n) * (beta_acc / n - delta_acc)
        delta = delta_acc - 2.0 * mu * float(emp_cov_trace.sum()) + p * mu**2
        delta /= p
        beta = min(beta_raw, delta)
        shrinkage = 0.0 if beta == 0.0 else beta / delta
    covariance = (1.0 - shrinkage) * emp_cov
    covariance.flat[:: p + 1] += shrinkage * mu
    precision = np.linalg.pinv(covariance, hermitian=True)
    return location, covariance, precision, float(shrinkage)
