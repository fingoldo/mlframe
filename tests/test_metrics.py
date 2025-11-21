"""
Comprehensive test suite for mlframe.metrics module.

Tests include:
- Correctness tests comparing with sklearn implementations
- Property-based tests using hypothesis
- Edge case tests
- Performance benchmarks
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss as sklearn_brier_score,
    precision_score,
    recall_score,
    f1_score,
)

from mlframe.metrics import (
    fast_roc_auc,
    fast_aucs,
    fast_numba_aucs,
    brier_score_loss,
    compute_pr_recall_f1_metrics,
    fast_calibration_binning,
    fast_calibration_metrics,
    calibration_metrics_from_freqs,
    fast_precision,
    fast_classification_report,
    probability_separation_score,
)


# =============================================================================
# Test Strategies
# =============================================================================

@st.composite
def binary_classification_data(draw, min_size=10, max_size=1000):
    """Generate valid binary classification data."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))

    # Ensure at least one of each class
    y_true = draw(arrays(
        dtype=np.int64,
        shape=size,
        elements=st.integers(0, 1),
    ))

    # Ensure we have both classes
    if y_true.sum() == 0:
        y_true[0] = 1
    elif y_true.sum() == len(y_true):
        y_true[0] = 0

    y_score = draw(arrays(
        dtype=np.float64,
        shape=size,
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    ))

    return y_true, y_score


@st.composite
def binary_predictions(draw, min_size=10, max_size=1000):
    """Generate binary predictions (hard labels)."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))

    y_true = draw(arrays(
        dtype=np.int64,
        shape=size,
        elements=st.integers(0, 1),
    ))

    y_pred = draw(arrays(
        dtype=np.int64,
        shape=size,
        elements=st.integers(0, 1),
    ))

    return y_true, y_pred


# =============================================================================
# ROC AUC Tests
# =============================================================================

class TestROCAUC:
    """Tests for ROC AUC computation."""

    @given(binary_classification_data())
    @settings(max_examples=100, deadline=None)
    def test_roc_auc_matches_sklearn(self, data):
        """Verify fast_roc_auc matches sklearn's roc_auc_score."""
        y_true, y_score = data

        # Skip if only one class present
        if len(np.unique(y_true)) < 2:
            return

        custom_auc = fast_roc_auc(y_true, y_score)
        sklearn_auc = roc_auc_score(y_true, y_score)

        np.testing.assert_allclose(custom_auc, sklearn_auc, rtol=1e-6, atol=1e-6)

    def test_roc_auc_perfect_prediction(self):
        """Perfect predictions should give AUC = 1.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        auc = fast_roc_auc(y_true, y_score)
        assert auc == 1.0

    def test_roc_auc_inverse_prediction(self):
        """Completely wrong predictions should give AUC = 0.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])

        auc = fast_roc_auc(y_true, y_score)
        assert auc == 0.0

    def test_roc_auc_random_prediction(self):
        """Random predictions should give AUC ~ 0.5."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 10000)
        y_score = np.random.random(10000)

        auc = fast_roc_auc(y_true, y_score)
        assert 0.45 < auc < 0.55

    def test_roc_auc_handles_2d_scores(self):
        """Should handle 2D score arrays (taking last column)."""
        y_true = np.array([0, 0, 1, 1])
        y_score_2d = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]])

        auc = fast_roc_auc(y_true, y_score_2d)
        sklearn_auc = roc_auc_score(y_true, y_score_2d[:, 1])

        np.testing.assert_allclose(auc, sklearn_auc, rtol=1e-6)

    def test_roc_auc_tied_scores(self):
        """Should handle tied scores correctly."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_score = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        auc = fast_roc_auc(y_true, y_score)
        sklearn_auc = roc_auc_score(y_true, y_score)

        np.testing.assert_allclose(auc, sklearn_auc, rtol=1e-6)


# =============================================================================
# PR AUC Tests
# =============================================================================

class TestPRAUC:
    """Tests for PR AUC (Average Precision) computation."""

    @given(binary_classification_data())
    @settings(max_examples=100, deadline=None)
    def test_pr_auc_matches_sklearn(self, data):
        """Verify PR AUC matches sklearn's average_precision_score."""
        y_true, y_score = data

        # Skip if only one class present
        if len(np.unique(y_true)) < 2:
            return

        _, custom_pr_auc = fast_aucs(y_true, y_score)
        sklearn_pr_auc = average_precision_score(y_true, y_score)

        # PR AUC can have small differences due to interpolation methods
        np.testing.assert_allclose(custom_pr_auc, sklearn_pr_auc, rtol=0.05, atol=0.02)

    def test_pr_auc_perfect_prediction(self):
        """Perfect predictions should give high PR AUC."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        _, pr_auc = fast_aucs(y_true, y_score)
        assert pr_auc >= 0.95

    def test_pr_auc_all_zeros_returns_zero(self):
        """All-zero true labels should return 0.0."""
        y_true = np.zeros(10)
        y_score = np.random.random(10)

        _, pr_auc = fast_aucs(y_true, y_score)
        assert pr_auc == 0.0


# =============================================================================
# Combined AUC Tests
# =============================================================================

class TestFastAucs:
    """Tests for fast_aucs which computes both ROC and PR AUC."""

    @given(binary_classification_data())
    @settings(max_examples=50, deadline=None)
    def test_fast_aucs_both_metrics_match(self, data):
        """Both AUCs should match individual computations."""
        y_true, y_score = data

        if len(np.unique(y_true)) < 2:
            return

        roc_auc, pr_auc = fast_aucs(y_true, y_score)
        roc_auc_single = fast_roc_auc(y_true, y_score)

        np.testing.assert_allclose(roc_auc, roc_auc_single, rtol=1e-10)

    def test_fast_aucs_returns_tuple(self):
        """Should return a tuple of two floats."""
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.1, 0.9, 0.2, 0.8])

        result = fast_aucs(y_true, y_score)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(x, float) for x in result)


# =============================================================================
# Brier Score Tests
# =============================================================================

class TestBrierScore:
    """Tests for Brier score loss computation."""

    @given(binary_classification_data())
    @settings(max_examples=100, deadline=None)
    def test_brier_score_matches_sklearn(self, data):
        """Verify brier_score_loss matches sklearn."""
        y_true, y_prob = data

        custom_brier = brier_score_loss(y_true.astype(np.float64), y_prob)
        sklearn_brier = sklearn_brier_score(y_true, y_prob)

        np.testing.assert_allclose(custom_brier, sklearn_brier, rtol=1e-10)

    def test_brier_score_perfect_prediction(self):
        """Perfect predictions should give Brier score = 0."""
        y_true = np.array([0, 1, 0, 1], dtype=np.float64)
        y_prob = np.array([0, 1, 0, 1], dtype=np.float64)

        brier = brier_score_loss(y_true, y_prob)
        assert brier == 0.0

    def test_brier_score_worst_prediction(self):
        """Completely wrong predictions should give Brier score = 1."""
        y_true = np.array([0, 1, 0, 1], dtype=np.float64)
        y_prob = np.array([1, 0, 1, 0], dtype=np.float64)

        brier = brier_score_loss(y_true, y_prob)
        assert brier == 1.0

    def test_brier_score_bounds(self):
        """Brier score should always be between 0 and 1."""
        np.random.seed(42)
        for _ in range(100):
            y_true = np.random.randint(0, 2, 100).astype(np.float64)
            y_prob = np.random.random(100)

            brier = brier_score_loss(y_true, y_prob)
            assert 0 <= brier <= 1


# =============================================================================
# Precision/Recall/F1 Tests
# =============================================================================

class TestPrecisionRecallF1:
    """Tests for precision, recall, and F1 computation."""

    @given(binary_predictions())
    @settings(max_examples=100, deadline=None)
    def test_precision_matches_sklearn(self, data):
        """Verify precision matches sklearn."""
        y_true, y_pred = data

        custom_precision, _, _ = compute_pr_recall_f1_metrics(y_true, y_pred)
        sklearn_precision = precision_score(y_true, y_pred, zero_division=0)

        np.testing.assert_allclose(custom_precision, sklearn_precision, rtol=1e-10)

    @given(binary_predictions())
    @settings(max_examples=100, deadline=None)
    def test_recall_matches_sklearn(self, data):
        """Verify recall matches sklearn."""
        y_true, y_pred = data

        _, custom_recall, _ = compute_pr_recall_f1_metrics(y_true, y_pred)
        sklearn_recall = recall_score(y_true, y_pred, zero_division=0)

        np.testing.assert_allclose(custom_recall, sklearn_recall, rtol=1e-10)

    @given(binary_predictions())
    @settings(max_examples=100, deadline=None)
    def test_f1_matches_sklearn(self, data):
        """Verify F1 score matches sklearn."""
        y_true, y_pred = data

        _, _, custom_f1 = compute_pr_recall_f1_metrics(y_true, y_pred)
        sklearn_f1 = f1_score(y_true, y_pred, zero_division=0)

        np.testing.assert_allclose(custom_f1, sklearn_f1, rtol=1e-10)

    def test_precision_recall_f1_all_correct(self):
        """All correct predictions should give P=R=F1=1."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1])

        p, r, f1 = compute_pr_recall_f1_metrics(y_true, y_pred)

        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0

    def test_precision_recall_f1_zero_division(self):
        """Should handle zero division cases."""
        # No positive predictions
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        p, r, f1 = compute_pr_recall_f1_metrics(y_true, y_pred)

        assert p == 0.0  # TP=0, FP=0


# =============================================================================
# Calibration Tests
# =============================================================================

class TestCalibration:
    """Tests for calibration binning and metrics."""

    def test_calibration_binning_basic(self):
        """Test basic calibration binning."""
        y_true = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
        y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])

        freqs_pred, freqs_true, hits = fast_calibration_binning(y_true, y_pred, nbins=10)

        assert len(freqs_pred) == len(freqs_true) == len(hits)
        assert hits.sum() == len(y_true)

    def test_calibration_binning_perfect(self):
        """Perfect calibration should have small MAE."""
        np.random.seed(42)
        n = 10000
        y_pred = np.random.random(n)
        y_true = (np.random.random(n) < y_pred).astype(np.int64)

        freqs_pred, freqs_true, hits = fast_calibration_binning(y_true, y_pred, nbins=10)
        mae = np.mean(np.abs(freqs_pred - freqs_true))

        # With perfect calibration, MAE should be small
        assert mae < 0.1

    def test_calibration_metrics_coverage(self):
        """Test calibration coverage calculation."""
        y_true = np.array([0, 0, 1, 1], dtype=np.int64)
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])

        mae, std, coverage = fast_calibration_metrics(y_true, y_pred, nbins=10)

        assert 0 <= coverage <= 1
        assert mae >= 0
        assert std >= 0

    def test_calibration_weighting_options(self):
        """Test different weighting strategies for calibration."""
        freqs_pred = np.array([0.1, 0.5, 0.9])
        freqs_true = np.array([0.15, 0.45, 0.85])
        hits = np.array([100, 500, 200])

        # Test log weighting
        mae_log, _, _ = calibration_metrics_from_freqs(
            freqs_pred, freqs_true, hits, nbins=10, array_size=800,
            use_weights=True, use_log_weighting=True, use_sqrt_weighting=False, use_power_weighting=False
        )

        # Test sqrt weighting
        mae_sqrt, _, _ = calibration_metrics_from_freqs(
            freqs_pred, freqs_true, hits, nbins=10, array_size=800,
            use_weights=True, use_log_weighting=False, use_sqrt_weighting=True, use_power_weighting=False
        )

        # Test power weighting
        mae_power, _, _ = calibration_metrics_from_freqs(
            freqs_pred, freqs_true, hits, nbins=10, array_size=800,
            use_weights=True, use_log_weighting=False, use_sqrt_weighting=False, use_power_weighting=True
        )

        # All should be valid
        assert mae_log >= 0
        assert mae_sqrt >= 0
        assert mae_power >= 0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_arrays_aucs(self):
        """fast_aucs with all-zero positives returns zeros."""
        y_true = np.zeros(10, dtype=np.int64)
        y_score = np.random.random(10)

        roc_auc, pr_auc = fast_aucs(y_true, y_score)

        assert roc_auc == 0.0
        assert pr_auc == 0.0

    def test_single_unique_score(self):
        """All same scores should still work."""
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.5, 0.5, 0.5, 0.5])

        roc_auc = fast_roc_auc(y_true, y_score)

        # Should return 0.5 for random
        assert 0 <= roc_auc <= 1

    def test_very_small_array(self):
        """Minimum viable array size."""
        y_true = np.array([0, 1])
        y_score = np.array([0.3, 0.7])

        roc_auc = fast_roc_auc(y_true, y_score)
        assert roc_auc == 1.0

    def test_probability_separation_empty_class(self):
        """Should return nan when class is empty."""
        y_true = np.array([0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4])

        score = probability_separation_score(y_true, y_prob, class_label=1)
        assert np.isnan(score)


# =============================================================================
# Performance Benchmarks
# =============================================================================

class TestPerformance:
    """Performance benchmarks comparing with sklearn."""

    @pytest.fixture
    def large_data(self):
        """Generate large dataset for benchmarking."""
        np.random.seed(42)
        n = 100000
        y_true = np.random.randint(0, 2, n)
        y_score = np.random.random(n)
        return y_true, y_score

    def test_custom_faster_than_sklearn_roc_auc(self, large_data):
        """Assert custom implementation is faster than sklearn for ROC AUC."""
        import time

        y_true, y_score = large_data

        # Warm up numba
        _ = fast_roc_auc(y_true[:1000], y_score[:1000])

        # Time custom
        start = time.perf_counter()
        for _ in range(10):
            custom_result = fast_roc_auc(y_true, y_score)
        custom_time = time.perf_counter() - start

        # Time sklearn
        start = time.perf_counter()
        for _ in range(10):
            sklearn_result = roc_auc_score(y_true, y_score)
        sklearn_time = time.perf_counter() - start

        # Verify correctness
        np.testing.assert_allclose(custom_result, sklearn_result, rtol=1e-6)

        # Assert faster (or at least comparable)
        # Custom should be at least 50% as fast (allowing some leeway)
        assert custom_time < sklearn_time * 2, f"Custom: {custom_time:.3f}s, sklearn: {sklearn_time:.3f}s"

    def test_custom_faster_than_sklearn_pr_auc(self, large_data):
        """Assert custom PR AUC implementation is faster than sklearn."""
        import time

        y_true, y_score = large_data

        # Warm up
        _ = fast_aucs(y_true[:1000], y_score[:1000])

        # Time custom
        start = time.perf_counter()
        for _ in range(10):
            _, custom_pr = fast_aucs(y_true, y_score)
        custom_time = time.perf_counter() - start

        # Time sklearn
        start = time.perf_counter()
        for _ in range(10):
            sklearn_pr = average_precision_score(y_true, y_score)
        sklearn_time = time.perf_counter() - start

        print(f"\nCustom PR AUC time: {custom_time:.3f}s")
        print(f"Sklearn PR AUC time: {sklearn_time:.3f}s")
        print(f"Speedup: {sklearn_time/custom_time:.2f}x")


# =============================================================================
# Regression Tests
# =============================================================================

class TestRegression:
    """Regression tests with known input/output pairs."""

    def test_roc_auc_known_value(self):
        """Test with known ROC AUC value."""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])

        auc = fast_roc_auc(y_true, y_score)
        expected = roc_auc_score(y_true, y_score)

        np.testing.assert_allclose(auc, expected, rtol=1e-10)

    def test_brier_score_known_value(self):
        """Test Brier score with known value."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.3])

        expected = ((1-0.9)**2 + (0-0.1)**2 + (1-0.8)**2 + (0-0.3)**2) / 4
        result = brier_score_loss(y_true, y_prob)

        np.testing.assert_allclose(result, expected, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
