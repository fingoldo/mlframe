"""Regression: ``linear_residual_robust`` stamps ``is_redundant_with_linres=True`` when its
MAD-trim pass finds no outliers (so the second-pass OLS is identical to plain ``linear_residual``).
TVT-2026-05-21 prod log: TVT-linres-Y and TVT-linresR-Y produced IDENTICAL RMSE=21.5433 because
clean-residual targets trip this exact case and the discovery loop re-evaluated the duplicate."""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.transforms import _linear_residual_robust_fit


class TestLinresRobustRedundancyFlag:
    """Groups tests covering linres robust redundancy flag."""
    def test_uniform_residuals_all_within_3_MAD_marks_redundant(self) -> None:
        # Uniform[-c, +c] residuals: MAD ~ c/2, sigma_MAD ~ 0.74c, 3*sigma_MAD ~ 2.22c.
        # All residuals are within [-c, +c] which is strictly inside [-2.22c, +2.22c]
        # so the keep mask covers every row -- the n_kept==n redundancy branch fires.
        """Uniform residuals all within 3 m a d marks redundant."""
        rng = np.random.default_rng(0)
        n = 500
        base = rng.normal(size=n).astype(np.float64)
        resid = rng.uniform(-1.0, 1.0, size=n)
        y = 1.7 * base + 0.3 + resid
        result = _linear_residual_robust_fit(y, base)
        assert result.get("is_redundant_with_linres") is True

    def test_real_outliers_do_NOT_mark_redundant(self) -> None:
        """Real outliers do n o t mark redundant."""
        rng = np.random.default_rng(1)
        n = 1000
        base = rng.normal(size=n).astype(np.float64)
        y = 1.7 * base + 0.3 + rng.normal(scale=0.05, size=n)
        # Inject ~5% Cauchy-flavoured outliers to trigger the trim pass.
        outlier_idx = rng.choice(n, size=n // 20, replace=False)
        y[outlier_idx] += rng.standard_cauchy(size=len(outlier_idx)) * 50.0
        result = _linear_residual_robust_fit(y, base)
        assert result.get("is_redundant_with_linres", False) is False, (
            "With genuine outliers the second-pass OLS differs from plain linres; should NOT be flagged redundant."
        )

    def test_degenerate_constant_residual_marks_redundant(self) -> None:
        """Degenerate constant residual marks redundant."""
        rng = np.random.default_rng(2)
        base = rng.normal(size=300).astype(np.float64)
        y = 2.0 * base + 1.0  # exact linear -> resid identically 0 -> sigma_MAD=0 -> redundant
        result = _linear_residual_robust_fit(y, base)
        assert result.get("is_redundant_with_linres") is True

    def test_too_few_samples_marks_redundant(self) -> None:
        """Too few samples marks redundant."""
        result = _linear_residual_robust_fit(
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
        )
        assert result.get("is_redundant_with_linres") is True
