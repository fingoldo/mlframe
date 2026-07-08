"""EXPERIMENTAL prototype: Bernstein/Bezier basis + Jacobi/Gegenbauer generalised orthogonal basis.

NOT wired into prod. Candidate universal-approximation families NOT already in ``bases.py`` (which has Fourier / RBF /
Sigmoid / Pade) or ``hermite_fe`` (Hermite / Legendre / Chebyshev / Laguerre) or the spline/wavelet extra-basis path.

* Bernstein -- ``B_{k,d}(t) = C(d,k) t^k (1-t)^(d-k)`` on ``t = (x - lo) / span`` in [0, 1]. Partition of unity, all-positive,
  variation-diminishing: a least-squares Bernstein fit cannot overshoot a bounded monotone/sigmoidal shape the way a global
  Chebyshev fit rings at the endpoints. The natural win is a bounded-domain saturating shape (CDF-like, dose-response plateau).

* Jacobi/Gegenbauer -- ``P_n^{(alpha,beta)}`` generalises Legendre (alpha=beta=0) and Chebyshev (alpha=beta=-1/2). The point of
  the prototype is the HONEST redundancy check: does a tuned alpha/beta recover anything Legendre/Chebyshev cannot? If not, it is
  REDUNDANT (the catalog already spans the same polynomial space -- different weighting, same span).

Each family fits via least-squares to y on the train fold (the prototype scores the fitted column's MI vs y, mirroring how the
existing extra-basis families are evaluated).
"""
from __future__ import annotations

import numpy as np
from scipy.special import comb, eval_jacobi, gegenbauer

__all__ = [
    "bernstein_design",
    "jacobi_design",
    "gegenbauer_design",
    "fit_basis_mi",
]


def _to_unit(x: np.ndarray) -> np.ndarray:
    c = np.asarray(x, dtype=np.float64)
    return np.asarray((c - c.min()) / (np.ptp(c) + 1e-12))


def _to_pm1(x: np.ndarray) -> np.ndarray:
    return 2.0 * _to_unit(x) - 1.0


def bernstein_design(x: np.ndarray, degree: int) -> np.ndarray:
    """(n, degree+1) Bernstein basis on x mapped to [0, 1]."""
    t = _to_unit(x)
    cols = [comb(degree, k) * t**k * (1.0 - t) ** (degree - k) for k in range(degree + 1)]
    return np.column_stack(cols)


def jacobi_design(x: np.ndarray, degree: int, alpha: float, beta: float) -> np.ndarray:
    """(n, degree+1) Jacobi basis P_n^{(alpha,beta)} on x mapped to [-1, 1]."""
    z = _to_pm1(x)
    return np.column_stack([eval_jacobi(n, alpha, beta, z) for n in range(degree + 1)])


def gegenbauer_design(x: np.ndarray, degree: int, lam: float) -> np.ndarray:
    """(n, degree+1) Gegenbauer (ultraspherical) basis C_n^{(lam)} on x mapped to [-1, 1]."""
    z = _to_pm1(x)
    return np.column_stack([gegenbauer(n, lam)(z) for n in range(degree + 1)])


def fit_basis_mi(design: np.ndarray, y: np.ndarray, nbins: int = 12) -> float:
    """Least-squares fit of the design matrix to y, return MI of the fitted column vs y."""
    from ._pairwise_modular_fe import _mi

    coef, *_ = np.linalg.lstsq(design, np.asarray(y, dtype=np.float64), rcond=None)
    return _mi(design @ coef, np.asarray(y).astype(np.int64), nbins=nbins)
