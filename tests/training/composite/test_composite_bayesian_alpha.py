"""Tests for ``bayesian_alpha_fit_bootstrap`` (the legacy bootstrap posterior
that pre-dates the conjugate Normal-Inverse-Gamma implementation now exposed
as :func:`bayesian_alpha_fit`).

Bootstrap posterior for the (alpha, beta) of ``linear_residual``. Lock:
- Mean recovers the OLS point estimate within sampling error.
- CI shrinks with n (large n => tight posterior; small n => wide).
- Reproducible under random_state.
- Subsample mode bounds compute on large datasets.
- Degenerate input (n < 4) returns a point estimate without crashing.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite import (
    _linear_residual_fit,
    bayesian_alpha_fit_bootstrap as bayesian_alpha_fit,
)


class TestPosteriorMean:
    def test_posterior_mean_close_to_ols_on_clean_data(self) -> None:
        rng = np.random.default_rng(0)
        n = 1000
        base = rng.normal(loc=10.0, scale=2.0, size=n)
        true_alpha, true_beta = 0.85, 3.14
        y = true_alpha * base + true_beta + rng.normal(scale=0.05, size=n)
        ols = _linear_residual_fit(y, base)
        post = bayesian_alpha_fit(y, base, n_bootstrap=200, random_state=42)
        assert abs(post["alpha_mean"] - ols["alpha"]) < 0.01
        assert abs(post["beta_mean"] - ols["beta"]) < 0.05

    def test_credible_interval_tightens_with_n(self) -> None:
        rng = np.random.default_rng(1)
        true_alpha = 0.85

        def _ci_width(n: int) -> float:
            base = rng.normal(loc=10.0, scale=2.0, size=n)
            y = true_alpha * base + rng.normal(scale=0.5, size=n)
            post = bayesian_alpha_fit(y, base, n_bootstrap=200, random_state=42)
            return post["alpha_ci_high"] - post["alpha_ci_low"]

        wide = _ci_width(100)
        tight = _ci_width(10000)
        assert tight < wide * 0.5, f"CI should tighten with n; wide(n=100)={wide:.4f}, tight(n=10000)={tight:.4f}"


class TestReproducibility:
    def test_same_seed_same_samples(self) -> None:
        rng = np.random.default_rng(0)
        base = rng.normal(size=500)
        y = base + rng.normal(scale=0.1, size=500)
        p1 = bayesian_alpha_fit(y, base, n_bootstrap=50, random_state=42)
        p2 = bayesian_alpha_fit(y, base, n_bootstrap=50, random_state=42)
        np.testing.assert_array_equal(p1["alpha_samples"], p2["alpha_samples"])
        np.testing.assert_array_equal(p1["beta_samples"], p2["beta_samples"])

    def test_different_seed_different_samples(self) -> None:
        rng = np.random.default_rng(0)
        base = rng.normal(size=500)
        y = base + rng.normal(scale=0.1, size=500)
        p1 = bayesian_alpha_fit(y, base, n_bootstrap=50, random_state=42)
        p2 = bayesian_alpha_fit(y, base, n_bootstrap=50, random_state=43)
        assert not np.array_equal(p1["alpha_samples"], p2["alpha_samples"])


class TestSubsampleMode:
    def test_subsample_smaller_than_n(self) -> None:
        """``subsample_n`` < n bounds bootstrap-draw size; posterior moments still finite."""
        rng = np.random.default_rng(2)
        n = 5000
        base = rng.normal(size=n)
        y = base + rng.normal(scale=0.1, size=n)
        post = bayesian_alpha_fit(y, base, n_bootstrap=30, subsample_n=500, random_state=0)
        assert np.isfinite(post["alpha_mean"])
        assert np.isfinite(post["alpha_std"])
        assert post["alpha_samples"].size == 30

    def test_subsample_larger_than_n_clamps(self) -> None:
        """``subsample_n`` > n clamps to n (no out-of-range indexing)."""
        rng = np.random.default_rng(3)
        n = 100
        base = rng.normal(size=n)
        y = base + rng.normal(scale=0.1, size=n)
        # Should not raise.
        post = bayesian_alpha_fit(y, base, n_bootstrap=20, subsample_n=10_000, random_state=0)
        assert np.isfinite(post["alpha_mean"])


class TestDegenerate:
    def test_n_lt_4_returns_point_estimate(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        base = np.array([1.0, 2.0, 3.0])
        post = bayesian_alpha_fit(y, base, n_bootstrap=100)
        assert post["n_bootstrap"] == 0
        assert np.isnan(post["alpha_std"])
        # Point estimate still finite.
        assert np.isfinite(post["alpha_mean"])

    def test_ci_level_records(self) -> None:
        rng = np.random.default_rng(4)
        base = rng.normal(size=200)
        y = base + rng.normal(scale=0.1, size=200)
        post = bayesian_alpha_fit(y, base, n_bootstrap=50, ci_level=0.90)
        assert post["ci_level"] == 0.90

    def test_ci_level_widens_interval(self) -> None:
        """CI at 99% wider than CI at 80%."""
        rng = np.random.default_rng(5)
        base = rng.normal(size=300)
        y = base + rng.normal(scale=0.3, size=300)
        p80 = bayesian_alpha_fit(y, base, n_bootstrap=200, ci_level=0.80, random_state=0)
        p99 = bayesian_alpha_fit(y, base, n_bootstrap=200, ci_level=0.99, random_state=0)
        w80 = p80["alpha_ci_high"] - p80["alpha_ci_low"]
        w99 = p99["alpha_ci_high"] - p99["alpha_ci_low"]
        assert w99 > w80


class TestBizValue:
    def test_high_noise_widens_posterior(self) -> None:
        """Increasing noise should widen the posterior for alpha. Reproducible lock that the bootstrap is doing real work, not just returning a constant point estimate."""
        rng = np.random.default_rng(0)
        n = 500
        base = rng.normal(loc=10.0, scale=2.0, size=n)

        def _ci_width(noise_scale: float) -> float:
            y = 0.85 * base + rng.normal(scale=noise_scale, size=n)
            post = bayesian_alpha_fit(y, base, n_bootstrap=150, random_state=42)
            return post["alpha_ci_high"] - post["alpha_ci_low"]

        low = _ci_width(0.05)
        high = _ci_width(2.0)
        assert high > low * 5, f"high-noise CI should be much wider; low={low:.4f}, high={high:.4f}"
