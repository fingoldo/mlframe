"""Unit + biz_value coverage for the matrix-based Rényi alpha-entropy MI estimator (``_renyi_alpha.py``).

Yu, Giraldo, Jenssen, Príncipe 2020 (IEEE TPAMI) -- opt-in via
``estimator='renyi_alpha'`` in ``_mi_dispatch.py``'s ``score_pair_mi``. Contract tests
mirror the routing-seam coverage pattern used for the other dispatcher-level
estimators (mixed_ksg/fastmi/genie): correctness on known cases here, dispatch
routing in ``test_mi_dispatch_contract.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._renyi_alpha import (
    renyi_alpha_mi,
    renyi_alpha_cmi,
    _renyi_entropy_from_gram,
    _rbf_gram,
)

SEEDS = (0, 1, 7, 42)


class TestEntropyPrimitive:
    """The core alpha-entropy-from-Gram-matrix building block."""

    def test_identity_gram_is_zero_entropy(self):
        """A perfectly concentrated Gram matrix (all points identical -> rank-1 K) has near-zero Rényi entropy."""
        K = np.ones((20, 20))
        s = _renyi_entropy_from_gram(K, alpha=1.01)
        assert s < 1e-6, f"rank-1 Gram matrix should carry ~0 entropy, got {s}"

    def test_diagonal_gram_has_max_entropy_for_its_rank(self):
        """A Gram matrix with n orthogonal points (identity matrix) has the maximal entropy log2(n)."""
        n = 16
        K = np.eye(n)
        s = _renyi_entropy_from_gram(K, alpha=1.01)
        assert abs(s - np.log2(n)) < 0.05, f"identity Gram matrix (n={n}) should have entropy ~log2(n)={np.log2(n):.3f}, got {s:.3f}"

    def test_entropy_nonnegative_and_finite(self):
        """The alpha-entropy of a real RBF Gram matrix is always finite and non-negative."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=(100, 1))
        K = _rbf_gram(x)
        s = _renyi_entropy_from_gram(K)
        assert np.isfinite(s) and s >= 0.0, f"entropy must be finite and >= 0, got {s}"


class TestMutualInformationCorrectness:
    """MI behavior on known-signal / known-noise fixtures."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_independent_gaussians_low_mi(self, seed):
        """Two independent standard normals should show low MI relative to a strongly dependent pair."""
        rng = np.random.default_rng(seed)
        n = 600
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        y_dep = x + 0.02 * rng.normal(size=n)
        mi_indep = renyi_alpha_mi(x, y)
        mi_dep = renyi_alpha_mi(x, y_dep)
        assert mi_indep >= 0.0 and np.isfinite(mi_indep)
        assert (
            mi_dep > mi_indep + 0.5
        ), f"seed={seed}: near-identical pair MI ({mi_dep:.3f}) should exceed independent-pair MI ({mi_indep:.3f}) by a wide margin"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_nonlinear_zero_correlation_signal_detected(self, seed):
        """y = x^2 has ~zero Pearson correlation with x but nonzero MI -- the whole point of an MI-based (not correlation-based) estimator."""
        rng = np.random.default_rng(seed)
        n = 1500
        x = rng.normal(size=n)
        y = x**2
        corr = float(np.corrcoef(x, y)[0, 1])
        mi = renyi_alpha_mi(x, y, max_n=1500)
        # x^2 is even, x is odd -> the population correlation is exactly 0; finite-sample noise at n=1500 stays well under 0.15.
        assert abs(corr) < 0.15, f"seed={seed}: fixture invariant broken, pearson corr={corr} not near 0"
        assert mi > 0.3, f"seed={seed}: renyi_alpha_mi must detect the nonlinear x->x^2 dependency despite ~0 correlation, got {mi:.3f}"

    def test_mi_symmetric(self):
        """I(X;Y) == I(Y;X) (the estimator has no directional asymmetry)."""
        rng = np.random.default_rng(3)
        x = rng.normal(size=400)
        y = x + 0.3 * rng.normal(size=400)
        assert abs(renyi_alpha_mi(x, y) - renyi_alpha_mi(y, x)) < 1e-9

    def test_subsampling_keeps_result_finite_above_max_n(self):
        """n > max_n triggers subsampling; the result is still finite and non-negative, not a crash or NaN."""
        rng = np.random.default_rng(5)
        n = 3000
        x = rng.normal(size=n)
        y = x + 0.1 * rng.normal(size=n)
        mi = renyi_alpha_mi(x, y, max_n=500)
        assert np.isfinite(mi) and mi >= 0.0


class TestConditionalMutualInformation:
    """I(X; Y | Z) chain-rule behavior."""

    def test_conditioning_on_y_itself_collapses_to_independence_floor(self):
        """I(X; Y | Z=Y) should collapse to roughly the independence floor -- Z fully absorbs Y's information."""
        rng = np.random.default_rng(11)
        n = 500
        x = rng.normal(size=n)
        y = x + 0.05 * rng.normal(size=n)
        mi_unconditional = renyi_alpha_mi(x, y)
        cmi = renyi_alpha_cmi(x, y, z=y)
        assert cmi < mi_unconditional * 0.5, f"I(X;Y|Z=Y) ({cmi:.3f}) should be well below the unconditional I(X;Y) ({mi_unconditional:.3f})"

    def test_conditioning_on_independent_z_preserves_mi(self):
        """I(X; Y | Z) for a Z independent of both X and Y should stay close to the unconditional I(X;Y)."""
        rng = np.random.default_rng(12)
        n = 500
        x = rng.normal(size=n)
        y = x + 0.1 * rng.normal(size=n)
        z = rng.normal(size=n)
        mi_unconditional = renyi_alpha_mi(x, y)
        cmi = renyi_alpha_cmi(x, y, z=z)
        assert (
            abs(cmi - mi_unconditional) < 0.4 * mi_unconditional + 0.2
        ), f"I(X;Y|Z=independent) ({cmi:.3f}) should stay close to unconditional I(X;Y) ({mi_unconditional:.3f})"

    def test_cmi_nonnegative_finite(self):
        """A generic three-variable conditional MI call returns a finite, non-negative value."""
        rng = np.random.default_rng(13)
        n = 400
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        z = rng.normal(size=n)
        cmi = renyi_alpha_cmi(x, y, z=z)
        assert np.isfinite(cmi) and cmi >= 0.0


class TestDispatcherRouting:
    """``estimator='renyi_alpha'`` is reachable through the public score_pair_mi seam."""

    def test_routed_through_score_pair_mi(self):
        """``score_pair_mi(..., estimator='renyi_alpha')`` returns a finite, positive MI on a dependent pair."""
        from mlframe.feature_selection.filters._mi_dispatch import score_pair_mi

        rng = np.random.default_rng(9)
        n = 400
        x = rng.normal(size=n)
        y = x + 0.05 * rng.normal(size=n)
        mi = score_pair_mi(x, y, estimator="renyi_alpha")
        assert np.isfinite(mi) and mi > 0.0
