"""Round-trip + property tests for Pack J (unary y-transforms) + Pack K (chain composer).

Each transform must satisfy ``inverse(forward(y)) == y`` to machine epsilon on a representative sample. The chain test verifies ``inverse(forward(y, base)) == y`` when a unary is stacked on top of ``linear_residual`` (the production composite shape).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite import (
    _linear_residual_fit,
    _linear_residual_forward,
    _linear_residual_inverse,
)
from mlframe.training.composite.transforms.unary import (
    cbrt_y_fit,
    cbrt_y_forward,
    cbrt_y_inverse,
    cbrt_y_domain,
    log_y_fit,
    log_y_forward,
    log_y_inverse,
    log_y_domain,
    yeo_johnson_y_fit,
    yeo_johnson_y_forward,
    yeo_johnson_y_inverse,
    quantile_normal_y_fit,
    quantile_normal_y_forward,
    quantile_normal_y_inverse,
    chain_bivariate_then_unary_fit,
    chain_bivariate_then_unary_forward,
    chain_bivariate_then_unary_inverse,
)

# ----------------------------------------------------------------------
# Pack J: unary round-trip
# ----------------------------------------------------------------------


class TestCbrtY:
    """Groups tests covering cbrt y."""
    @pytest.mark.parametrize(
        "y",
        [
            np.array([0.0, 1.0, -1.0, 8.0, -8.0, 1e6, -1e6]),
            np.linspace(-1000.0, 1000.0, 200),
        ],
    )
    def test_round_trip(self, y: np.ndarray) -> None:
        """Round trip."""
        params = cbrt_y_fit(y)
        t = cbrt_y_forward(y, params)
        y_back = cbrt_y_inverse(t, params)
        np.testing.assert_allclose(y_back, y, rtol=1e-10, atol=1e-9)

    def test_compresses_heavy_tails(self) -> None:
        """A heavy-tailed Laplace becomes ~Gaussian under cbrt: variance ratio collapses."""
        rng = np.random.default_rng(0)
        y = rng.laplace(0.0, 1.0, size=5000)
        t = cbrt_y_forward(y, {})
        # Cube-root of Laplace has smaller tail mass; compare the ratio of 99th to 50th percentile.
        ratio_y = float(np.percentile(np.abs(y), 99) / max(np.percentile(np.abs(y), 50), 1e-9))
        ratio_t = float(np.percentile(np.abs(t), 99) / max(np.percentile(np.abs(t), 50), 1e-9))
        assert ratio_t < ratio_y, f"cbrt did not compress tails: ratio_y={ratio_y:.2f}, ratio_t={ratio_t:.2f}"


class TestLogY:
    """Groups tests covering log y."""
    def test_round_trip_positive(self) -> None:
        """Round trip positive."""
        y = np.linspace(0.5, 100.0, 200)
        params = log_y_fit(y)
        t = log_y_forward(y, params)
        y_back = log_y_inverse(t, params)
        np.testing.assert_allclose(y_back, y, rtol=1e-10, atol=1e-9)

    def test_round_trip_with_negative(self) -> None:
        """Offset is fitted so y_min + offset > 0; round-trip still must hold after the offset."""
        rng = np.random.default_rng(1)
        y = rng.normal(0.0, 50.0, 200)
        params = log_y_fit(y)
        t = log_y_forward(y, params)
        y_back = log_y_inverse(t, params)
        np.testing.assert_allclose(y_back, y, rtol=1e-9, atol=1e-7)

    def test_offset_keeps_log_finite(self) -> None:
        """Offset keeps log finite."""
        y = np.array([-5.0, 0.0, 10.0])
        params = log_y_fit(y)
        t = log_y_forward(y, params)
        assert np.all(np.isfinite(t))


class TestYeoJohnsonY:
    """Groups tests covering yeo johnson y."""
    def test_round_trip_positive_skew(self) -> None:
        """Heavy right-skewed y; YJ fits lambda < 1 and inverse must recover y."""
        rng = np.random.default_rng(2)
        y = rng.exponential(scale=2.0, size=500)
        params = yeo_johnson_y_fit(y)
        t = yeo_johnson_y_forward(y, params)
        y_back = yeo_johnson_y_inverse(t, params)
        # YJ is numerically delicate near lambda boundaries; allow ~1e-6 rel.
        np.testing.assert_allclose(y_back, y, rtol=1e-6, atol=1e-6)

    def test_round_trip_mixed_sign(self) -> None:
        """Round trip mixed sign."""
        rng = np.random.default_rng(3)
        y = rng.standard_t(df=5, size=500)
        params = yeo_johnson_y_fit(y)
        t = yeo_johnson_y_forward(y, params)
        y_back = yeo_johnson_y_inverse(t, params)
        np.testing.assert_allclose(y_back, y, rtol=1e-6, atol=1e-6)

    def test_lambda_in_valid_range(self) -> None:
        """Lambda in valid range."""
        rng = np.random.default_rng(4)
        y = rng.exponential(scale=2.0, size=500)
        params = yeo_johnson_y_fit(y)
        lam = float(params["lambda"])
        assert -2.0 <= lam <= 4.0, f"lambda={lam} outside the clipped range"


class TestQuantileNormalY:
    """Groups tests covering quantile normal y."""
    def test_round_trip_to_train_distribution(self) -> None:
        """Round-trip is exact ONLY for y values within the train support; on extreme tails ``norm.ppf`` is clipped and the inverse cannot recover them. Test on train rows only."""
        rng = np.random.default_rng(5)
        y = rng.exponential(scale=3.0, size=2000)
        params = quantile_normal_y_fit(y)
        t = quantile_normal_y_forward(y, params)
        y_back = quantile_normal_y_inverse(t, params)
        # Tolerance: knots resolution + interp error -- relax to ~5% rel on the bulk; tail samples may drift more.
        # Compare ordered statistics to make the check distribution-free.
        np.testing.assert_allclose(
            np.sort(y_back),
            np.sort(y),
            rtol=0.05,
            atol=0.5,
        )

    def test_forward_is_approximately_standard_normal(self) -> None:
        """Forward is approximately standard normal."""
        rng = np.random.default_rng(6)
        y = rng.exponential(scale=3.0, size=5000)
        params = quantile_normal_y_fit(y)
        t = quantile_normal_y_forward(y, params)
        # Std should be close to 1 and mean close to 0 under the empirical -> Normal map.
        assert abs(float(t.mean())) < 0.2
        assert abs(float(t.std()) - 1.0) < 0.2


# ----------------------------------------------------------------------
# Pack K: chained round-trip (linear_residual -> cbrt_y)
# ----------------------------------------------------------------------


class TestChainBivariateUnary:
    """The production motivator: stack ``cbrt_y`` on top of ``linear_residual``. Verifies the chain composer's math (forward / inverse) on a controlled dataset."""

    def test_linres_then_cbrt_round_trip(self) -> None:
        """Linres then cbrt round trip."""
        rng = np.random.default_rng(7)
        n = 500
        base = rng.normal(11500.0, 600.0, n)
        T = rng.laplace(0.0, 5.0, n)  # heavy-tailed residual mimicking production
        y = base + 5.0 + T  # alpha=1, beta=5 by construction

        unary = (cbrt_y_fit, cbrt_y_forward, cbrt_y_inverse)
        params = chain_bivariate_then_unary_fit(
            y=y,
            base=base,
            bivariate_fit=_linear_residual_fit,
            bivariate_forward=_linear_residual_forward,
            unary=unary,
        )
        t2 = chain_bivariate_then_unary_forward(
            y=y,
            base=base,
            params=params,
            bivariate_forward=_linear_residual_forward,
            unary=unary,
        )
        y_back = chain_bivariate_then_unary_inverse(
            t2=t2,
            base=base,
            params=params,
            bivariate_inverse=_linear_residual_inverse,
            unary=unary,
        )
        np.testing.assert_allclose(y_back, y, rtol=1e-9, atol=1e-7)

    def test_linres_then_yj_round_trip(self) -> None:
        """Linres then yj round trip."""
        rng = np.random.default_rng(8)
        n = 500
        base = rng.normal(11500.0, 600.0, n)
        T = rng.standard_t(df=4, size=n) * 5.0  # heavy-tailed residual
        y = base + 5.0 + T

        unary = (yeo_johnson_y_fit, yeo_johnson_y_forward, yeo_johnson_y_inverse)
        params = chain_bivariate_then_unary_fit(
            y=y,
            base=base,
            bivariate_fit=_linear_residual_fit,
            bivariate_forward=_linear_residual_forward,
            unary=unary,
        )
        t2 = chain_bivariate_then_unary_forward(
            y=y,
            base=base,
            params=params,
            bivariate_forward=_linear_residual_forward,
            unary=unary,
        )
        y_back = chain_bivariate_then_unary_inverse(
            t2=t2,
            base=base,
            params=params,
            bivariate_inverse=_linear_residual_inverse,
            unary=unary,
        )
        np.testing.assert_allclose(y_back, y, rtol=1e-5, atol=1e-5)


class TestDomainChecks:
    """Domain checks must return all-True on finite inputs (unary transforms have no domain restriction at predict-time except finiteness)."""

    def test_cbrt_y_domain_all_finite(self) -> None:
        """Cbrt y domain all finite."""
        y = np.array([0.0, 1.0, -1.0, 100.0, np.nan, np.inf])
        mask = cbrt_y_domain(y)
        assert mask.tolist() == [True, True, True, True, False, False]

    def test_log_y_domain_respects_offset(self) -> None:
        """Log y domain respects offset."""
        y = np.array([-5.0, 0.0, 10.0])
        params = log_y_fit(y)
        mask = log_y_domain(y, params)
        assert mask.all(), "fitted offset must make all train rows valid"
