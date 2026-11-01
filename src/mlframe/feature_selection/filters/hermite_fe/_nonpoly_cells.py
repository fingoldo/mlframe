"""Single-function cells of the non-polynomial bases (fourier, rbf, sigmoid, pade) for the per-column basis FE.

The orthogonal-polynomial bases give one column per ``(basis, degree)``: the ``degree``-th basis function of the preprocessed axis ``z``. The four bases registered next to them have a fit
and an evaluator for a coefficient vector but no entry in ``polyeval_dispatch``, so every cell of theirs raised ``KeyError`` and the routers never saw them. The cell of degree ``d`` here is the
same one-hot choice the polynomial cells make, in each basis' own coefficient layout:

* ``fourier``: harmonic order ``ceil(d / 2)``, ``sin`` for odd ``d`` and ``cos`` for even ``d`` (the layout ``[a_1, b_1, a_2, b_2, ...]`` with the one-hot at position ``d - 1``);
* ``rbf``: the Gaussian bump at the ``d``-th of the nine quantile centres (``d`` up to 9);
* ``sigmoid``: the logistic step at the ``d``-th of the nine quantile thresholds (``d`` up to 9);
* ``pade``: ``1 / (1 + z**d)`` (numerator ``a_0 = 1``, denominator ``b_d = 1``), zero where the denominator is within 1e-3 of a pole.

``params`` is what the basis' ``fit`` returned (data-dependent centres, thresholds, bandwidth, ranges), so replay from a frozen recipe evaluates the same cell.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

#: Bases whose cells this module evaluates.
NONPOLY_BASES = ("fourier", "rbf", "sigmoid", "pade")
#: Highest degree with a distinct cell for the bases that have a fixed number of centres / thresholds.
MAX_CENTRE_DEGREE = 9


def is_nonpoly(basis: str) -> bool:
    """Whether ``basis`` is one of the non-polynomial bases."""
    return basis in NONPOLY_BASES


def _one_hot(length: int, position: int) -> np.ndarray:
    """Coefficient vector of ``length`` zeros with a one at ``position``."""
    c = np.zeros(length, dtype=np.float64)  # Intentional float64: the njit evaluators accumulate in double
    c[position] = 1.0
    return c


def nonpoly_cell(basis: str, z: np.ndarray, degree: int, params: "Mapping[str, Any] | None") -> np.ndarray:
    """The degree-``degree`` cell of the non-polynomial ``basis`` at the preprocessed axis ``z``; ``ValueError`` for a degree with no cell."""
    from ..bases import _fourier_eval_njit, _pade_eval_njit, _rbf_eval_kernel_njit, _sigmoid_eval_kernel_njit

    d = int(degree)
    if d < 1:
        raise ValueError(f"{basis} has no cell of degree {d}")
    zc = np.ascontiguousarray(z, dtype=np.float64)  # Intentional float64: fit/replay bit-identity contract (basis axis)
    if basis == "fourier":
        return np.asarray(_fourier_eval_njit(zc, _one_hot(2 * ((d + 1) // 2), d - 1)))
    if basis == "pade":
        c = _one_hot(2 * d + 1, 0)
        c[2 * d] = 1.0
        return np.asarray(_pade_eval_njit(zc, c))
    if basis in ("rbf", "sigmoid"):
        if d > MAX_CENTRE_DEGREE:
            raise ValueError(f"{basis} has {MAX_CENTRE_DEGREE} centres; no cell of degree {d}")
        if params is None:
            raise ValueError(f"{basis} needs the fitted centres / thresholds")
        c = _one_hot(d, d - 1)
        if basis == "rbf":
            # Intentional float64: the njit evaluators accumulate in double
            return np.asarray(_rbf_eval_kernel_njit(zc, c, np.ascontiguousarray(params["centres"], dtype=np.float64), float(params["bandwidth"])))
        # Intentional float64: the njit evaluators accumulate in double
        return np.asarray(_sigmoid_eval_kernel_njit(zc, c, np.ascontiguousarray(params["thresholds"], dtype=np.float64), float(params["slope"])))
    raise KeyError(basis)


def eval_basis_cell(basis: str, z: np.ndarray, degree: int, params: "Mapping[str, Any] | None" = None) -> np.ndarray:
    """The ``(basis, degree)`` column at the preprocessed axis ``z``: the one-hot polynomial for the orthogonal bases, :func:`nonpoly_cell` for the others."""
    if is_nonpoly(basis):
        return nonpoly_cell(basis, z, degree, params)
    from . import polyeval_dispatch

    return np.asarray(polyeval_dispatch(basis, np.ascontiguousarray(z, dtype=np.float64), _one_hot(int(degree) + 1, int(degree))))  # Intentional float64: fit/replay bit-identity contract
