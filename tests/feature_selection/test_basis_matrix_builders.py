"""Tests for the basis-matrix precompute path used by the GEMV
optimisation in ``hermite_fe._eval_coef_pair``.

The builders ``_build_basis_{hermite,legendre,chebyshev,laguerre}``
must reproduce the columnwise output of the matching numpy polynomial
``*val`` evaluator on every input row + every degree k <= max_degree.
The GEMV path computes ``B[:, :k] @ c`` so the column slice MUST match
``polyval(z, c)`` exactly (up to fp round-off in the recurrence) -- any
column-k drift would break Horner equivalence at runtime and only show
up as biased MI scores under multi_fidelity=False.
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.hermite_e import hermeval
from numpy.polynomial.laguerre import lagval
from numpy.polynomial.legendre import legval

from mlframe.feature_selection.filters.hermite_fe import (
    _build_basis_chebyshev,
    _build_basis_hermite,
    _build_basis_laguerre,
    _build_basis_legendre,
    build_basis_matrix,
)

_BUILDERS = {
    "hermite": (_build_basis_hermite, hermeval),
    "legendre": (_build_basis_legendre, legval),
    "chebyshev": (_build_basis_chebyshev, chebval),
    "laguerre": (_build_basis_laguerre, lagval),
}


class TestBasisBuildersAgreeWithNumpyPolyval:
    """For each basis, ``B[:, :k] @ c`` must equal ``polyval(z, c[:k])``."""

    @pytest.mark.parametrize("basis", sorted(_BUILDERS))
    @pytest.mark.parametrize("max_degree", [1, 3, 6])
    def test_basis_matrix_matches_polyval_for_random_coefs(
        self, basis: str, max_degree: int,
    ) -> None:
        builder, polyval_fn = _BUILDERS[basis]
        rng = np.random.default_rng(seed=11 + max_degree)
        # Each basis has a natural domain; pick a representative sample
        # inside it. Out-of-domain inputs cause numerical blow-up that
        # the GEMV path is NOT expected to handle (preprocess is the
        # caller's job).
        if basis == "hermite":
            z = rng.normal(size=300)  # std-normal support.
        elif basis in ("legendre", "chebyshev"):
            z = rng.uniform(-1.0, 1.0, size=300)
        else:  # laguerre on [0, +inf)
            z = rng.exponential(scale=2.0, size=300)
        z = np.ascontiguousarray(z, dtype=np.float64)
        B = builder(z, max_degree)
        assert B.shape == (z.shape[0], max_degree + 1)
        for k in range(1, max_degree + 2):
            coef = rng.uniform(-1.0, 1.0, size=k)
            gemv_out = B[:, :k] @ coef
            ref_out = polyval_fn(z, coef)
            np.testing.assert_allclose(
                gemv_out, ref_out, atol=1e-10, rtol=1e-10,
                err_msg=(
                    f"basis={basis} max_degree={max_degree} k={k}: "
                    f"GEMV path disagrees with numpy polyval"
                ),
            )

    @pytest.mark.parametrize("basis", sorted(_BUILDERS))
    def test_public_dispatcher_returns_same_as_private_builder(
        self, basis: str,
    ) -> None:
        builder, _ = _BUILDERS[basis]
        rng = np.random.default_rng(seed=11)
        if basis == "hermite":
            z = rng.normal(size=200)
        elif basis in ("legendre", "chebyshev"):
            z = rng.uniform(-1.0, 1.0, size=200)
        else:
            z = rng.exponential(scale=2.0, size=200)
        max_degree = 5
        priv = builder(np.ascontiguousarray(z, dtype=np.float64), max_degree)
        pub = build_basis_matrix(basis, z, max_degree)
        np.testing.assert_array_equal(pub, priv)

    def test_unknown_basis_raises_keyerror(self) -> None:
        with pytest.raises(KeyError, match="rbf"):
            build_basis_matrix("rbf", np.linspace(0, 1, 50), 3)


class TestBasisBuilderEdgeCases:
    """``max_degree=0`` returns the constant column, no recurrence runs."""

    @pytest.mark.parametrize("basis", sorted(_BUILDERS))
    def test_max_degree_zero_is_constant_column(self, basis: str) -> None:
        builder, _ = _BUILDERS[basis]
        rng = np.random.default_rng(seed=11)
        z = rng.normal(size=100)
        z = np.ascontiguousarray(z, dtype=np.float64)
        B = builder(z, 0)
        assert B.shape == (100, 1)
        # All bases have P_0(z) = 1.0 (or L_0(z) = 1.0 for Laguerre).
        np.testing.assert_array_equal(B[:, 0], np.ones(100))

    @pytest.mark.parametrize("basis", sorted(_BUILDERS))
    def test_max_degree_one_matches_first_degree_polynomial(
        self, basis: str,
    ) -> None:
        """P_1(z) is z for hermite/legendre/chebyshev, (1-z) for laguerre."""
        builder, polyval_fn = _BUILDERS[basis]
        rng = np.random.default_rng(seed=11)
        if basis == "hermite":
            z = rng.normal(size=100)
        elif basis in ("legendre", "chebyshev"):
            z = rng.uniform(-1.0, 1.0, size=100)
        else:
            z = rng.exponential(scale=2.0, size=100)
        z = np.ascontiguousarray(z, dtype=np.float64)
        B = builder(z, 1)
        assert B.shape == (100, 2)
        # Compare against polyval([0, 1]) which returns just P_1(z).
        ref = polyval_fn(z, np.array([0.0, 1.0]))
        np.testing.assert_allclose(B[:, 1], ref, atol=1e-12)


class TestBasisGEMVNumericalEquivalence:
    """For each basis the GEMV path ``B @ c`` must agree with the
    Horner backend ``polyeval_dispatch`` to fp round-off. This is the
    invariant that justifies the basis-matrix optimisation kept in the
    discovery codepath -- if it ever drifts, the optimisation produces
    biased fitness scores and CMA-ES converges to a different optimum.
    """

    @pytest.mark.parametrize("basis", sorted(_BUILDERS))
    @pytest.mark.parametrize("n", [100, 5_000, 50_000])
    def test_gemv_matches_horner_dispatcher(
        self, basis: str, n: int,
    ) -> None:
        from mlframe.feature_selection.filters.hermite_fe import polyeval_dispatch

        builder, _ = _BUILDERS[basis]
        rng = np.random.default_rng(seed=11 + n)
        if basis == "hermite":
            z = rng.normal(size=n)
        elif basis in ("legendre", "chebyshev"):
            z = rng.uniform(-1.0, 1.0, size=n)
        else:
            z = rng.exponential(scale=2.0, size=n)
        z = np.ascontiguousarray(z, dtype=np.float64)
        max_degree = 5
        coef = rng.uniform(-1.0, 1.0, size=max_degree + 1)
        B = builder(z, max_degree)
        horner_out = polyeval_dispatch(basis, z, coef)
        gemv_out = B @ coef
        np.testing.assert_allclose(
            gemv_out, horner_out, atol=1e-10, rtol=1e-10,
            err_msg=(
                f"basis={basis} n={n}: GEMV path disagrees with the "
                f"polyeval_dispatch Horner backend"
            ),
        )
