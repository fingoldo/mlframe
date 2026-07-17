"""Tests for ``streaming_alpha_check_and_refit`` (R10c brainstorm extension #8).

Concept-drift guard for the linear_residual alpha: Chow-style stability check on a rolling buffer of recent observations. When the buffer's alpha drifts significantly from the deployed alpha (|z| > threshold), the function returns the refit alpha for the caller to apply; otherwise no-op.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite import (
    streaming_alpha_check_and_refit,
)


class TestNoDriftPath:
    """Groups tests covering no drift path."""
    def test_no_drift_returns_current_params(self) -> None:
        """When the buffer's alpha matches the deployed alpha (within noise), the function returns the unchanged ``(current_alpha, current_beta)``."""
        rng = np.random.default_rng(0)
        n = 500
        base = rng.normal(loc=10.0, scale=2.0, size=n)
        true_alpha, true_beta = 0.85, 3.14
        y = true_alpha * base + true_beta + rng.normal(scale=0.5, size=n)
        new_alpha, new_beta, info = streaming_alpha_check_and_refit(
            y,
            base,
            current_alpha=true_alpha,
            current_beta=true_beta,
        )
        assert info["refit"] is False
        assert new_alpha == pytest.approx(true_alpha, abs=1e-12)
        assert new_beta == pytest.approx(true_beta, abs=1e-12)
        assert info["reason"] == "no_drift"

    def test_z_score_reported_even_without_refit(self) -> None:
        """Z score reported even without refit."""
        rng = np.random.default_rng(1)
        n = 500
        base = rng.normal(size=n)
        y = base + rng.normal(scale=0.1, size=n)
        _, _, info = streaming_alpha_check_and_refit(
            y,
            base,
            current_alpha=1.0,
            current_beta=0.0,
        )
        assert np.isfinite(info["z_score"])


class TestDriftDetected:
    """Groups tests covering drift detected."""
    def test_clear_drift_triggers_refit(self) -> None:
        """Deployed alpha is 0.5; buffer's true alpha is 2.0 -- large drift, z >> threshold, refit fires."""
        rng = np.random.default_rng(2)
        n = 500
        base = rng.normal(loc=10.0, scale=2.0, size=n)
        true_alpha_new = 2.0
        y = true_alpha_new * base + 1.0 + rng.normal(scale=0.5, size=n)
        deployed_alpha = 0.5
        new_alpha, _new_beta, info = streaming_alpha_check_and_refit(
            y,
            base,
            current_alpha=deployed_alpha,
            current_beta=0.0,
            z_threshold=3.0,
        )
        assert info["refit"] is True
        assert info["reason"] == "drift_detected"
        # New alpha close to the buffer's true alpha.
        assert abs(new_alpha - true_alpha_new) < 0.05
        # z_score way above threshold.
        assert info["z_score"] > 3.0

    def test_threshold_controls_sensitivity(self) -> None:
        """Same buffer with the same deployed alpha: high threshold => no refit; low threshold => refit."""
        rng = np.random.default_rng(3)
        n = 500
        base = rng.normal(loc=10.0, scale=2.0, size=n)
        y = 1.0 * base + rng.normal(scale=0.5, size=n)
        deployed = 0.95  # slight drift from 1.0
        # High threshold: no refit.
        _, _, info_high = streaming_alpha_check_and_refit(
            y,
            base,
            current_alpha=deployed,
            current_beta=0.0,
            z_threshold=100.0,
        )
        # Low threshold: refit.
        _, _, info_low = streaming_alpha_check_and_refit(
            y,
            base,
            current_alpha=deployed,
            current_beta=0.0,
            z_threshold=0.1,
        )
        assert info_high["refit"] is False
        assert info_low["refit"] is True


class TestEdgeCases:
    def test_buffer_too_small_skips(self) -> None:
        """Buffer too small skips."""
        rng = np.random.default_rng(4)
        y = rng.normal(size=50)
        base = rng.normal(size=50)
        new_alpha, _new_beta, info = streaming_alpha_check_and_refit(
            y,
            base,
            current_alpha=0.5,
            current_beta=0.0,
            min_buffer_n=200,
        )
        assert info["refit"] is False
        assert info["reason"] == "buffer_too_small"
        assert new_alpha == 0.5

    def test_constant_base_degenerate(self) -> None:
        """Constant base degenerate."""
        n = 300
        base = np.array([7.0] * n)
        y = np.linspace(0.0, 10.0, n)
        new_alpha, _new_beta, info = streaming_alpha_check_and_refit(
            y,
            base,
            current_alpha=0.5,
            current_beta=0.0,
            min_buffer_n=200,
        )
        assert info["refit"] is False
        assert info["reason"] == "degenerate_buffer"
        # current params preserved.
        assert new_alpha == 0.5

    def test_non_finite_filtered(self) -> None:
        """Non finite filtered."""
        rng = np.random.default_rng(5)
        n = 500
        base = rng.normal(loc=10.0, scale=2.0, size=n)
        y = 1.0 * base + rng.normal(scale=0.5, size=n)
        # Inject some NaN; finite count still well above min_buffer_n.
        y[:50] = np.nan
        _new_alpha, _new_beta, info = streaming_alpha_check_and_refit(
            y,
            base,
            current_alpha=1.0,
            current_beta=0.0,
            min_buffer_n=200,
        )
        # Should not crash, info has finite z_score from the 450 surviving rows.
        assert np.isfinite(info["z_score"])
