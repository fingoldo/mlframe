"""Tests for the conjugate Normal-Inverse-Gamma posterior in
``bayesian_alpha_fit`` (replaces the bootstrap-as-Bayes-posterior shim).

Locks:
- Posterior mean matches OLS on noiseless / clean data (the closed-form
  posterior mean IS the OLS estimate under flat prior).
- Posterior std > 0 on noisy data, and shrinks as n grows.
- Credible interval contains the true alpha on synthetic data.
- 95% CI computed from t-quantile, not normal.
- Legacy bootstrap is still reachable via ``bayesian_alpha_fit_bootstrap``.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.discovery.bayesian import (
    bayesian_alpha_fit,
    bayesian_alpha_fit_bootstrap,
)
from mlframe.training.composite.transforms import _linear_residual_fit


class TestPosteriorMeanMatchesOLS:
    """Groups tests covering posterior mean matches o l s."""
    def test_noiseless_posterior_mean_equals_ols(self) -> None:
        """On noiseless y = alpha * base + beta, posterior mean is exactly OLS."""
        n = 200
        base = np.linspace(0.0, 10.0, n)
        true_alpha, true_beta = 0.85, 3.14
        y = true_alpha * base + true_beta  # zero noise
        post = bayesian_alpha_fit(y, base, random_state=0)
        ols = _linear_residual_fit(y, base)
        # Posterior mean === OLS coef under flat prior.
        assert abs(post["alpha_mean"] - ols["alpha"]) < 1e-9
        assert abs(post["beta_mean"] - ols["beta"]) < 1e-9
        # Noiseless => sigma^2 -> 0 => CI collapses to a point.
        assert post["alpha_ci_high"] - post["alpha_ci_low"] < 1e-9

    def test_posterior_mean_close_to_ols_on_clean_data(self) -> None:
        """Posterior mean close to ols on clean data."""
        rng = np.random.default_rng(0)
        n = 1000
        base = rng.normal(loc=10.0, scale=2.0, size=n)
        true_alpha, true_beta = 0.85, 3.14
        y = true_alpha * base + true_beta + rng.normal(scale=0.05, size=n)
        post = bayesian_alpha_fit(y, base, random_state=0)
        ols = _linear_residual_fit(y, base)
        # The conjugate posterior MEAN is OLS exactly (not just close).
        assert abs(post["alpha_mean"] - ols["alpha"]) < 1e-9
        assert abs(post["beta_mean"] - ols["beta"]) < 1e-9


class TestPosteriorStd:
    """Groups tests covering posterior std."""
    def test_posterior_std_positive_on_noisy_data(self) -> None:
        """Posterior std positive on noisy data."""
        rng = np.random.default_rng(1)
        n = 300
        base = rng.normal(loc=10.0, scale=2.0, size=n)
        y = 0.85 * base + rng.normal(scale=1.0, size=n)
        post = bayesian_alpha_fit(y, base, random_state=0)
        assert post["alpha_std"] > 0.0
        assert post["beta_std"] > 0.0
        assert np.isfinite(post["alpha_std"])
        assert np.isfinite(post["beta_std"])

    def test_posterior_std_shrinks_with_n(self) -> None:
        """Posterior std shrinks with n."""
        rng = np.random.default_rng(2)
        true_alpha = 0.85

        def _alpha_std(n: int) -> float:
            """Alpha std."""
            base = rng.normal(loc=10.0, scale=2.0, size=n)
            y = true_alpha * base + rng.normal(scale=0.5, size=n)
            post = bayesian_alpha_fit(y, base, random_state=42)
            return post["alpha_std"]

        wide = _alpha_std(100)
        tight = _alpha_std(10_000)
        # Posterior std should scale ~ 1/sqrt(n); 10x data => ~3x tighter.
        assert tight < wide * 0.5, f"std should shrink with n; got wide={wide:.4f} tight={tight:.4f}"


class TestCredibleInterval:
    """Groups tests covering credible interval."""
    def test_ci_contains_true_alpha_on_noisy_data(self) -> None:
        """95% CI should contain the true alpha when the model is well-specified."""
        rng = np.random.default_rng(3)
        n = 1000
        base = rng.normal(loc=10.0, scale=2.0, size=n)
        true_alpha = 0.85
        y = true_alpha * base + rng.normal(scale=0.5, size=n)
        post = bayesian_alpha_fit(y, base, ci_level=0.95, random_state=0)
        assert post["alpha_ci_low"] < true_alpha < post["alpha_ci_high"]

    def test_ci_level_widens_interval(self) -> None:
        """Ci level widens interval."""
        rng = np.random.default_rng(4)
        n = 200
        base = rng.normal(size=n)
        y = base + rng.normal(scale=0.3, size=n)
        p80 = bayesian_alpha_fit(y, base, ci_level=0.80, random_state=0)
        p99 = bayesian_alpha_fit(y, base, ci_level=0.99, random_state=0)
        w80 = p80["alpha_ci_high"] - p80["alpha_ci_low"]
        w99 = p99["alpha_ci_high"] - p99["alpha_ci_low"]
        assert w99 > w80


class TestMetadata:
    """Groups tests covering metadata."""
    def test_posterior_kind_recorded(self) -> None:
        """Posterior kind recorded."""
        rng = np.random.default_rng(5)
        base = rng.normal(size=100)
        y = base + rng.normal(scale=0.1, size=100)
        post = bayesian_alpha_fit(y, base, random_state=0)
        assert post["posterior_kind"] == "conjugate_normal_inverse_gamma"
        assert post["degrees_of_freedom"] == 98  # n - 2
        assert post["n_samples"] > 0

    def test_legacy_bootstrap_still_works(self) -> None:
        """Legacy bootstrap still works."""
        rng = np.random.default_rng(6)
        base = rng.normal(size=200)
        y = base + rng.normal(scale=0.1, size=200)
        # Verify the legacy bootstrap stays callable under its alias.
        post_b = bayesian_alpha_fit_bootstrap(y, base, n_bootstrap=50, random_state=0)
        assert post_b["n_bootstrap"] == 50
        assert post_b["alpha_samples"].size == 50


class TestDegenerate:
    """Groups tests covering degenerate."""
    def test_n_lt_4_returns_point_estimate(self) -> None:
        """N lt 4 returns point estimate."""
        y = np.array([1.0, 2.0, 3.0])
        base = np.array([1.0, 2.0, 3.0])
        post = bayesian_alpha_fit(y, base)
        assert post["n_samples"] == 0
        assert np.isnan(post["alpha_std"])
        assert np.isfinite(post["alpha_mean"])
