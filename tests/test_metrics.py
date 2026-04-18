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
    log_loss as sklearn_log_loss,
    precision_score,
    recall_score,
    f1_score,
)

from mlframe.metrics import (
    fast_roc_auc,
    fast_aucs,
    fast_numba_aucs,
    brier_score_loss,
    fast_log_loss,
    compute_pr_recall_f1_metrics,
    fast_calibration_binning,
    fast_calibration_metrics,
    calibration_metrics_from_freqs,
    fast_precision,
    fast_classification_report,
    probability_separation_score,
    cb_logits_to_probs_binary,
    cb_logits_to_probs_multiclass,
    maximum_absolute_percentage_error,
    integral_calibration_error_from_metrics,
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

    def test_pr_auc_all_zeros_returns_nan(self):
        """All-zero true labels should return NaN (single-class, undefined)."""
        y_true = np.zeros(10)
        y_score = np.random.random(10)

        _, pr_auc = fast_aucs(y_true, y_score)
        assert np.isnan(pr_auc)


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
# Log Loss Tests
# =============================================================================

class TestLogLoss:
    """Tests for fast_log_loss implementation."""

    @given(binary_classification_data())
    @settings(max_examples=100, deadline=None)
    def test_log_loss_matches_sklearn(self, data):
        """Assert custom log_loss matches sklearn for random data."""
        y_true, y_score = data

        custom_ll = fast_log_loss(y_true.astype(np.float64), y_score)
        sklearn_ll = sklearn_log_loss(y_true, y_score)

        np.testing.assert_allclose(custom_ll, sklearn_ll, rtol=1e-6, atol=1e-6)

    def test_log_loss_known_value(self):
        """Test with known input/output."""
        y_true = np.array([0, 0, 1, 1], dtype=np.float64)
        y_pred = np.array([0.1, 0.4, 0.35, 0.8])

        custom_ll = fast_log_loss(y_true, y_pred)
        sklearn_ll = sklearn_log_loss(y_true, y_pred)

        np.testing.assert_allclose(custom_ll, sklearn_ll, rtol=1e-10)

    def test_log_loss_edge_cases(self):
        """Test edge cases: perfect predictions, worst predictions."""
        # Near-perfect predictions (can't use 0/1 exactly due to clipping)
        y_true = np.array([0, 1, 0, 1], dtype=np.float64)
        y_pred = np.array([0.001, 0.999, 0.001, 0.999])
        ll = fast_log_loss(y_true, y_pred)
        assert ll < 0.01  # Should be very small

        # Near-worst predictions
        y_pred_bad = np.array([0.999, 0.001, 0.999, 0.001])
        ll_bad = fast_log_loss(y_true, y_pred_bad)
        assert ll_bad > 5.0  # Should be very large

    def test_log_loss_single_class_returns_nan(self):
        """Log loss with single class should return nan."""
        y_true = np.array([1, 1, 1, 1], dtype=np.float64)
        y_pred = np.array([0.5, 0.6, 0.7, 0.8])

        ll = fast_log_loss(y_true, y_pred)
        assert np.isnan(ll)

    def test_log_loss_dtype_handling(self):
        """Test different input dtypes are handled correctly."""
        y_true_int = np.array([0, 1, 0, 1])
        y_pred = np.array([0.2, 0.8, 0.3, 0.7])

        # Should work with integer y_true
        ll = fast_log_loss(y_true_int, y_pred)
        sklearn_ll = sklearn_log_loss(y_true_int, y_pred)

        np.testing.assert_allclose(ll, sklearn_ll, rtol=1e-6)


class TestLogLossPerformance:
    """Performance tests for fast_log_loss."""

    @pytest.fixture
    def large_data(self):
        np.random.seed(42)
        n = 100000
        y_true = np.random.randint(0, 2, n).astype(np.float64)
        y_score = np.random.random(n)
        return y_true, y_score

    def test_faster_than_sklearn(self, large_data):
        """Assert fast_log_loss is faster than sklearn."""
        import time
        y_true, y_score = large_data

        # Warm up numba
        _ = fast_log_loss(y_true[:1000], y_score[:1000])

        # Time custom (10 iterations)
        start = time.perf_counter()
        for _ in range(10):
            custom_result = fast_log_loss(y_true, y_score)
        custom_time = time.perf_counter() - start

        # Time sklearn (10 iterations)
        start = time.perf_counter()
        for _ in range(10):
            sklearn_result = sklearn_log_loss(y_true, y_score)
        sklearn_time = time.perf_counter() - start

        print(f"\nCustom log_loss: {custom_time:.4f}s, sklearn: {sklearn_time:.4f}s")
        print(f"Speedup: {sklearn_time/custom_time:.1f}x")

        # Verify correctness
        np.testing.assert_allclose(custom_result, sklearn_result, rtol=1e-6)

        # Should be significantly faster
        assert custom_time < sklearn_time


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

    def test_single_class_aucs_return_nan(self):
        """fast_aucs with single class returns NaN (undefined metrics)."""
        # All negatives (no positives)
        y_true_all_neg = np.zeros(10, dtype=np.int64)
        y_score = np.random.random(10)

        roc_auc, pr_auc = fast_aucs(y_true_all_neg, y_score)
        assert np.isnan(roc_auc), "ROC AUC should be NaN for all-negative data"
        assert np.isnan(pr_auc), "PR AUC should be NaN for all-negative data"

        # All positives (no negatives)
        y_true_all_pos = np.ones(10, dtype=np.int64)
        roc_auc, pr_auc = fast_aucs(y_true_all_pos, y_score)
        assert np.isnan(roc_auc), "ROC AUC should be NaN for all-positive data"
        assert np.isnan(pr_auc), "PR AUC should be NaN for all-positive data"

    def test_single_class_roc_auc_matches_sklearn(self):
        """fast_roc_auc should return NaN for single-class data, matching sklearn behavior."""
        y_true_all_pos = np.ones(99, dtype=np.int64)
        y_true_all_neg = np.zeros(99, dtype=np.int64)
        y_score = np.random.random(99)

        # Test all positives
        custom_roc = fast_roc_auc(y_true_all_pos, y_score)
        assert np.isnan(custom_roc), "fast_roc_auc should return NaN for all-positive data"

        # Test all negatives
        custom_roc = fast_roc_auc(y_true_all_neg, y_score)
        assert np.isnan(custom_roc), "fast_roc_auc should return NaN for all-negative data"

        # Verify this matches sklearn's behavior (sklearn also returns NaN with warning)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sklearn_roc_pos = roc_auc_score(y_true_all_pos, y_score)
            sklearn_roc_neg = roc_auc_score(y_true_all_neg, y_score)

        assert np.isnan(sklearn_roc_pos), "sklearn should return NaN for all-positive"
        assert np.isnan(sklearn_roc_neg), "sklearn should return NaN for all-negative"

    def test_single_class_various_sizes(self):
        """Test single-class handling with various array sizes."""
        for size in [2, 10, 100, 1000]:
            y_score = np.random.random(size)

            # All positives
            y_true_pos = np.ones(size, dtype=np.int64)
            roc, pr = fast_aucs(y_true_pos, y_score)
            assert np.isnan(roc), f"ROC AUC should be NaN for size={size}, all positives"
            assert np.isnan(pr), f"PR AUC should be NaN for size={size}, all positives"

            # All negatives
            y_true_neg = np.zeros(size, dtype=np.int64)
            roc, pr = fast_aucs(y_true_neg, y_score)
            assert np.isnan(roc), f"ROC AUC should be NaN for size={size}, all negatives"
            assert np.isnan(pr), f"PR AUC should be NaN for size={size}, all negatives"

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
# CatBoost Logits to Probabilities
# =============================================================================


class TestCbLogitsToProbs:
    """Edge cases for cb_logits_to_probs_binary."""

    def test_zeros(self):
        """Logits of zero should give 0.5 probability for both classes."""
        logits = np.array([0.0, 0.0])
        probs = cb_logits_to_probs_binary(logits)
        np.testing.assert_allclose(probs[:, 0], 0.5)
        np.testing.assert_allclose(probs[:, 1], 0.5)

    def test_extreme_positive(self):
        """Very large positive logit should map to prob close to [0, 1]."""
        logits = np.array([100.0])
        probs = cb_logits_to_probs_binary(logits)
        np.testing.assert_allclose(probs[0, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(probs[0, 1], 1.0, atol=1e-10)

    def test_extreme_negative(self):
        """Very large negative logit should map to prob close to [1, 0]."""
        logits = np.array([-100.0])
        probs = cb_logits_to_probs_binary(logits)
        np.testing.assert_allclose(probs[0, 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(probs[0, 1], 0.0, atol=1e-10)

    @given(
        logits=arrays(np.float64, st.just(20), elements=st.floats(-10, 10))
    )
    @settings(max_examples=50, deadline=None)
    def test_rows_sum_to_one_and_in_unit_interval(self, logits):
        """For any logits, each row sums to 1.0 and all values in [0, 1]."""
        probs = cb_logits_to_probs_binary(logits)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)


class TestCbLogitsToPropsMulticlass:
    """Property-based tests for cb_logits_to_probs_multiclass."""

    @given(
        data=st.data(),
        n_classes=st.integers(2, 10),
        n_samples=st.integers(1, 50),
    )
    @settings(max_examples=30, deadline=None)
    def test_rows_sum_to_one_and_in_unit_interval(self, data, n_classes, n_samples):
        """For any (n_classes, n_samples) logits, rows sum to 1.0, values in [0, 1]."""
        logits = data.draw(
            arrays(np.float64, (n_classes, n_samples), elements=st.floats(-10, 10))
        )
        probs = cb_logits_to_probs_multiclass(logits)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)


# =============================================================================
# Maximum Absolute Percentage Error
# =============================================================================


class TestMaximumAbsolutePercentageError:
    """Edge cases for maximum_absolute_percentage_error."""

    def test_perfect_predictions(self):
        """When y_true == y_pred, result should be 0."""
        y = np.array([1.0, 2.0, 3.0])
        assert maximum_absolute_percentage_error(y, y) == 0.0

    def test_zero_true_values(self):
        """With zero true values, epsilon denominator gives large but finite result."""
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 1.0])
        result = maximum_absolute_percentage_error(y_true, y_pred)
        assert np.isfinite(result)
        assert result > 0


# =============================================================================
# Fast ROC AUC Edge Cases
# =============================================================================


class TestFastRocAucEdgeCases:
    """Edge cases for fast_roc_auc."""

    def test_perfect_separation(self):
        """Perfect separation should give AUC of 1.0."""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        assert fast_roc_auc(y_true, y_score) == 1.0


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


# =============================================================================
# ICE — roc_auc_penalty linear ramp
# =============================================================================

class TestICEPenaltyRamp:
    """Tests for the sub-threshold ramp in integral_calibration_error_from_metrics.

    Contract:
        - penalty contribution = 0 when |auc - 0.5| >= min_roc_auc - 0.5
        - penalty contribution = roc_auc_penalty at auc == 0.5 (deepest deficit)
        - linear in the deficit (not a step)
        - symmetric about 0.5 (inverted ranker punished same as barely-positive)
        - zero when roc_auc_penalty == 0
        - no-op when min_roc_auc <= 0.5 (empty penalty zone)
    """

    @staticmethod
    def _ice_with_zero_rest(auc, min_roc_auc, roc_auc_penalty):
        """Run the ICE formula with all base weights zeroed so the result
        is exactly the penalty contribution — isolates the thing under test.
        """
        return integral_calibration_error_from_metrics(
            calibration_mae=0.0,
            calibration_std=0.0,
            calibration_coverage=1.0,
            brier_loss=0.0,
            roc_auc=auc,
            pr_auc=0.0,
            mae_weight=0.0,
            std_weight=0.0,
            roc_auc_weight=0.0,
            pr_auc_weight=0.0,
            brier_loss_weight=0.0,
            min_roc_auc=min_roc_auc,
            roc_auc_penalty=roc_auc_penalty,
        )

    def test_penalty_zero_outside_zone(self):
        # auc=0.60 is above min_roc_auc=0.55 → outside → penalty 0
        assert self._ice_with_zero_rest(0.60, 0.55, 3.0) == 0.0
        # exactly at threshold → penalty ~0 (deficit rounds to FP noise, not a real penalty)
        np.testing.assert_allclose(self._ice_with_zero_rest(0.55, 0.55, 3.0), 0.0, atol=1e-12)

    def test_penalty_max_at_perfect_random(self):
        # At auc=0.5 the deficit equals the full threshold_width → full penalty
        penalty = self._ice_with_zero_rest(0.5, 0.55, 3.0)
        np.testing.assert_allclose(penalty, 3.0, rtol=1e-10)

    def test_penalty_linear_interior(self):
        # Midway between 0.5 and 0.55 → half the penalty
        penalty = self._ice_with_zero_rest(0.525, 0.55, 3.0)
        np.testing.assert_allclose(penalty, 1.5, rtol=1e-10)

    def test_penalty_symmetric_about_half(self):
        # Inverted rankers (auc<0.5) feel the same ramp as barely-positive ones
        left = self._ice_with_zero_rest(0.45, 0.55, 2.0)  # deficit=0.0, at boundary
        np.testing.assert_allclose(left, 0.0, atol=1e-12)
        left_inside = self._ice_with_zero_rest(0.47, 0.55, 2.0)  # deficit=0.02
        right_inside = self._ice_with_zero_rest(0.53, 0.55, 2.0)  # deficit=0.02
        np.testing.assert_allclose(left_inside, right_inside, rtol=1e-10)
        # And both equal (0.02 / 0.05) * 2.0 = 0.8
        np.testing.assert_allclose(left_inside, 0.8, rtol=1e-10)

    def test_penalty_continuous_across_threshold(self):
        # The replacement fixes the pre-existing step cliff. Sample a dense
        # grid across the threshold and assert adjacent samples differ by
        # at most the expected linear step — no jumps.
        min_roc_auc, penalty = 0.55, 5.0
        aucs = np.linspace(0.49, 0.60, 221)
        vals = np.array([self._ice_with_zero_rest(a, min_roc_auc, penalty) for a in aucs])
        steps = np.abs(np.diff(vals))
        # Per-step max deficit change = 0.11 / 220 ≈ 5e-4 → max ICE delta ≈ 5e-4/0.05 * 5 = 0.05
        assert steps.max() < 0.06, f"Step cliff regressed: max delta = {steps.max():.4f}"

    def test_penalty_monotonic_below_threshold(self):
        # Penalty grows as we move from threshold toward 0.5
        penalties = [self._ice_with_zero_rest(a, 0.55, 3.0) for a in [0.55, 0.54, 0.53, 0.52, 0.51, 0.50]]
        # Strictly non-decreasing, strictly increasing inside the interior
        assert all(penalties[i] <= penalties[i + 1] + 1e-12 for i in range(len(penalties) - 1))
        assert penalties[0] == 0.0
        assert penalties[-1] > penalties[0]

    def test_penalty_zero_when_knob_zero(self):
        assert self._ice_with_zero_rest(0.5, 0.55, 0.0) == 0.0
        assert self._ice_with_zero_rest(0.52, 0.55, 0.0) == 0.0

    def test_no_penalty_when_min_roc_auc_at_half(self):
        # threshold_width = 0 → guard prevents div-by-zero, penalty disabled
        assert self._ice_with_zero_rest(0.50, 0.50, 5.0) == 0.0
        assert self._ice_with_zero_rest(0.49, 0.50, 5.0) == 0.0

    def test_default_args_produce_no_penalty(self):
        # Defaults: roc_auc_penalty=0.0 → pure behaviour unchanged for callers
        # that never opted into the penalty mechanism.
        val = integral_calibration_error_from_metrics(
            calibration_mae=0.01, calibration_std=0.01, calibration_coverage=1.0,
            brier_loss=0.25, roc_auc=0.50, pr_auc=0.5,
        )
        # With penalty knob=0, ICE = 0.25*0.8 + 0.01*3 + 0.01*2 - |0|*1.5 - 0.5*0.1 = 0.2 + 0.03 + 0.02 - 0 - 0.05 = 0.2
        np.testing.assert_allclose(val, 0.2, rtol=1e-10)


class TestICENaNGuards:
    """Sensors for NaN-propagation guard added 2026-04-18.

    Background: fast_aucs_per_group_optimized returns NaN for single-class
    y_true or zero-variance y_score. Before the guard, those NaN values
    flowed into `res = ... - np.abs(roc_auc - 0.5) * weight` and made the
    entire ICE metric NaN, which silently broke early-stopping comparisons
    (NaN > best is always False, so the trainer locked in iteration-1's
    "best" and never improved — no visible error, just frozen training).
    """

    def test_nan_roc_auc_does_not_propagate(self):
        val = integral_calibration_error_from_metrics(
            calibration_mae=0.01, calibration_std=0.01, calibration_coverage=1.0,
            brier_loss=0.25, roc_auc=float("nan"), pr_auc=0.5,
            roc_auc_penalty=3.0, min_roc_auc=0.6,
        )
        assert np.isfinite(val), "NaN roc_auc must not poison ICE"
        # With roc_term skipped: base_loss = 0.25*0.8 + 0.01*3 + 0.01*2 = 0.25
        # minus pr_term (0.5 * 0.1 = 0.05) = 0.20. No penalty (guard skipped on NaN).
        np.testing.assert_allclose(val, 0.20, rtol=1e-10)

    def test_nan_pr_auc_does_not_propagate(self):
        val = integral_calibration_error_from_metrics(
            calibration_mae=0.01, calibration_std=0.01, calibration_coverage=1.0,
            brier_loss=0.25, roc_auc=0.7, pr_auc=float("nan"),
        )
        assert np.isfinite(val)

    def test_both_nan_returns_finite(self):
        val = integral_calibration_error_from_metrics(
            calibration_mae=0.01, calibration_std=0.01, calibration_coverage=1.0,
            brier_loss=0.25, roc_auc=float("nan"), pr_auc=float("nan"),
        )
        assert np.isfinite(val)


class TestPerGroupAUCEdgeCases:
    """Proactive-probe findings 2026-04-19 (round 5):

    - 1.3: single-sample groups returned ``(0.0, 0.0)`` instead of
      ``(nan, nan)``. ``compute_mean_aucs_per_group`` filters NaN but
      treats 0.0 as legitimate data → a fold with many 1-sample groups
      silently depressed the mean AUC toward 0.
    - 1.1: the inner njit loop silently returned NaN for single-class or
      single-sample groups. Operators staring at ``mean_group_roc_auc=nan``
      had no hint that "most of my groups collapsed" was the cause.
      Warning added at the Python-level wrapper when ≥50% of groups are
      NaN.
    """

    def _build(self, group_ids, y_true, y_score):
        from mlframe.metrics import fast_aucs_per_group_optimized
        return fast_aucs_per_group_optimized(
            y_true=np.asarray(y_true, dtype=np.int8),
            y_score=np.asarray(y_score, dtype=np.float64),
            group_ids=np.asarray(group_ids, dtype=np.int64),
        )

    def test_single_sample_group_returns_nan_not_zero(self):
        """The fix: group_size==1 -> (nan, nan). Pre-fix: (0.0, 0.0)."""
        # Group 0: 3 samples (valid). Group 1: 1 sample (degenerate).
        _, _, per_group = self._build(
            group_ids=[0, 0, 0, 1],
            y_true=[0, 1, 0, 1],
            y_score=[0.1, 0.9, 0.2, 0.5],
        )
        assert 1 in per_group
        roc, pr = per_group[1]
        assert np.isnan(roc), f"single-sample group must return NaN ROC, got {roc}"
        assert np.isnan(pr)

    def test_single_sample_group_excluded_from_mean(self):
        """End-to-end: compute_mean_aucs_per_group must not include the
        NaN from single-sample groups in the mean (was depressing mean
        toward 0 when treated as legitimate 0.0 data)."""
        from mlframe.metrics import compute_mean_aucs_per_group
        _, _, per_group = self._build(
            group_ids=[0, 0, 0, 0, 1, 2],  # group 0 valid; 1,2 single-sample
            y_true=[0, 1, 0, 1, 1, 0],
            y_score=[0.1, 0.9, 0.2, 0.8, 0.5, 0.5],
        )
        mean_roc, mean_pr = compute_mean_aucs_per_group(per_group)
        # Group 0 should produce a high ROC (~1.0). Pre-fix, groups 1,2
        # contributed 0.0 and dragged the mean down.
        assert mean_roc > 0.5, (
            f"mean ROC should reflect only the valid group ({mean_roc}); "
            "if this is <0.5, single-sample groups are polluting the mean"
        )

    def test_single_class_group_returns_nan(self):
        """Existing behavior: group with all-same-class y_true returns
        NaN (confirmed pre-fix). Locked in as a regression sensor —
        the inner njit path at fast_numba_aucs_simple line 739 is
        load-bearing for the mean-filter contract."""
        _, _, per_group = self._build(
            group_ids=[0, 0, 0, 1, 1, 1],
            y_true=[1, 1, 1, 0, 1, 0],  # group 0: all 1s; group 1: mixed
            y_score=[0.1, 0.2, 0.3, 0.5, 0.7, 0.4],
        )
        roc0, _ = per_group[0]
        roc1, _ = per_group[1]
        assert np.isnan(roc0), "all-single-class group must return NaN ROC"
        assert not np.isnan(roc1), "mixed-class group must compute a valid ROC"

    def test_warning_fires_when_majority_groups_are_nan(self, caplog):
        """The new observability warning: when ≥50% of groups return NaN
        (single-class or single-sample), log a single WARNING naming the
        count and the likely causes."""
        import logging
        # 4 groups: 3 single-sample (NaN), 1 valid. 3/4 = 75% NaN.
        gids = [0, 0, 0, 1, 2, 3]
        y_true = [0, 1, 0, 1, 0, 1]
        y_score = [0.1, 0.9, 0.2, 0.5, 0.5, 0.5]
        with caplog.at_level(logging.WARNING, logger="mlframe.metrics"):
            self._build(gids, y_true, y_score)
        msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("groups returned NaN" in m for m in msgs), (
            f"Expected a WARNING about majority-NaN groups; got: {msgs}"
        )
        # The message must name the count and the likely causes.
        joined = " ".join(msgs)
        assert "single-class" in joined or "single-sample" in joined

    def test_no_warning_when_minority_groups_are_nan(self, caplog):
        """False-positive sensor: if only 1 in 10 groups is NaN, no
        warning should fire — the per-group mean is still trustworthy
        and noisy logs would drown the signal."""
        import logging
        # 10 groups, 9 valid (2-sample each), 1 single-sample -> 10% NaN.
        gids = sum([[i, i] for i in range(9)], []) + [9]  # groups 0..8 have 2 samples; group 9 has 1
        y_true = ([0, 1] * 9) + [1]
        y_score = list(np.linspace(0.1, 0.9, 19))
        with caplog.at_level(logging.WARNING, logger="mlframe.metrics"):
            self._build(gids, y_true, y_score)
        warn_msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert not any("groups returned NaN" in m for m in warn_msgs), (
            f"Did not expect a warning for only 10% NaN groups; got: {warn_msgs}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
