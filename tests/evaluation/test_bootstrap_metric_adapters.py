"""Tests for ``mlframe.evaluation._bootstrap_metric_adapters`` -- shared Brier/log-loss adapters
consolidated specifically to prevent two independent copies (training.honest_diagnostics and its bench
harness) from silently drifting apart. Previously had zero direct test coverage.

The per-row variants (``ll_per_row``/``brier_per_row``) exist to feed a closed-form O(n) BCa jackknife --
their defining contract is that their MEAN equals the aggregate ``log_loss``/``brier`` these same adapters
compute, since the jackknife derivation depends on that equivalence.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss as sk_log_loss

from mlframe.evaluation._bootstrap_metric_adapters import brier, brier_per_row, ll_per_row, log_loss


def _make_probs(n=500, seed=0):
    """Seeded binary labels + predicted probabilities (clipped away from 0/1) for reference comparisons."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(np.float64)
    p = np.clip(rng.random(n), 1e-6, 1 - 1e-6)
    return y, p


class TestBrierMatchesReference:
    """Groups tests pinning brier() against sklearn's brier_score_loss."""

    def test_matches_sklearn_brier_score_loss(self):
        """Matches sklearn brier score loss."""
        y, p = _make_probs()
        got = brier(y, p)
        expected = brier_score_loss(y, p)
        assert np.isclose(got, expected, rtol=1e-9)

    def test_perfect_predictions_give_zero(self):
        """Perfect predictions give zero."""
        y = np.array([0.0, 1.0, 0.0, 1.0])
        p = np.array([0.0, 1.0, 0.0, 1.0])
        assert brier(y, p) == 0.0

    def test_returns_a_python_float(self):
        """Returns a python float."""
        y, p = _make_probs(n=10)
        assert isinstance(brier(y, p), float)


class TestLogLossMatchesReference:
    """Groups tests pinning log_loss() against sklearn's log_loss."""

    def test_matches_sklearn_log_loss(self):
        """Matches sklearn log loss."""
        y, p = _make_probs()
        got = log_loss(y, p)
        expected = sk_log_loss(y, p, labels=[0.0, 1.0])
        assert np.isclose(got, expected, rtol=1e-6, atol=1e-9)

    def test_returns_a_python_float(self):
        """Returns a python float."""
        y, p = _make_probs(n=10)
        assert isinstance(log_loss(y, p), float)


class TestPerRowMeanEqualsAggregate:
    """The defining contract feeding the closed-form jackknife: mean(per_row) == aggregate metric."""

    def test_ll_per_row_mean_equals_log_loss(self):
        """Ll per row mean equals log loss."""
        y, p = _make_probs(n=800, seed=1)
        per_row = ll_per_row(y, p)
        assert per_row.shape == y.shape
        assert np.isclose(float(np.mean(per_row)), log_loss(y, p), rtol=1e-9)

    def test_brier_per_row_mean_equals_brier(self):
        """Brier per row mean equals brier."""
        y, p = _make_probs(n=800, seed=2)
        per_row = brier_per_row(y, p)
        assert per_row.shape == y.shape
        assert np.isclose(float(np.mean(per_row)), brier(y, p), rtol=1e-9)


class TestPerRowEdgeCases:
    """Groups tests covering the eps-clipping / degenerate-probability edge cases."""

    def test_ll_per_row_handles_probability_exactly_zero_or_one(self):
        """log-loss's per-row formula must clip away from 0/1 (via eps) rather than producing inf/nan --
        the jackknife it feeds cannot tolerate a non-finite leave-one-out contribution."""
        y = np.array([1.0, 0.0])
        p = np.array([0.0, 1.0])  # maximally wrong AND at the exact clipping boundary
        out = ll_per_row(y, p)
        assert np.isfinite(out).all()
        assert (out > 0).all()

    def test_brier_per_row_is_always_non_negative(self):
        """Brier per row is always non negative."""
        y, p = _make_probs(n=100, seed=3)
        assert (brier_per_row(y, p) >= 0).all()

    def test_ll_per_row_correct_class_gets_lower_loss_than_incorrect(self):
        """Ll per row correct class gets lower loss than incorrect."""
        y = np.array([1.0, 1.0])
        p_correct = np.array([0.9, 0.9])
        p_incorrect = np.array([0.1, 0.1])
        assert ll_per_row(y, p_correct)[0] < ll_per_row(y, p_incorrect)[0]
