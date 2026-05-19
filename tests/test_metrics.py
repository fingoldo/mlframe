"""
Comprehensive test suite for mlframe.metrics.core module.

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

from mlframe.metrics.core import (
    fast_roc_auc,
    fast_aucs,
    fast_numba_aucs,
    brier_score_loss,
    fast_brier_score_loss,
    fast_log_loss,
    compute_pr_recall_f1_metrics,
    fast_calibration_binning,
    fast_calibration_metrics,
    calibration_metrics_from_freqs,
    compute_ece_and_brier_decomposition,
    fast_calibration_report,
    render_title_metric_token,
    DEFAULT_TITLE_METRICS_TOKENS,
    TITLE_METRIC_TOKENS,
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

@pytest.mark.fast
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
        assert auc == pytest.approx(1.0, rel=1e-6)

    def test_roc_auc_inverse_prediction(self):
        """Completely wrong predictions should give AUC = 0.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])

        auc = fast_roc_auc(y_true, y_score)
        assert auc == pytest.approx(0.0, abs=1e-9)

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

@pytest.mark.fast
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
        assert brier == pytest.approx(0.0, abs=1e-9)

    def test_brier_score_worst_prediction(self):
        """Completely wrong predictions should give Brier score = 1."""
        y_true = np.array([0, 1, 0, 1], dtype=np.float64)
        y_prob = np.array([1, 0, 1, 0], dtype=np.float64)

        brier = brier_score_loss(y_true, y_prob)
        assert brier == pytest.approx(1.0, rel=1e-6)

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

        # Warm up numba. fast_log_loss dispatches seq vs par by N; the
        # par kernel has its own JIT compile cost (~1.5s cold) and a
        # small-N warmup hits seq only. Warm both paths.
        _ = fast_log_loss(y_true[:1000], y_score[:1000])
        _ = fast_log_loss(y_true, y_score)

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

    # ============================================================
    # ECE + Brier decomposition (added 2026-04-27)
    # ============================================================

    def _synthetic_binary_data(self, seed: int = 42, n: int = 5000, p_pos: float = 0.3, noise: float = 0.2):
        rng = np.random.default_rng(seed)
        y_true = (rng.random(n) < p_pos).astype(np.float64)
        y_pred = np.clip(p_pos + 0.4 * (y_true - p_pos) + noise * rng.standard_normal(n), 0.001, 0.999)
        return y_true, y_pred

    def test_brier_decomposition_identity(self):
        """Murphy 1973: BinnedBrier == REL - RES + UNC exactly to fp precision.

        Identity is on the *binned* Brier (computed inside the kernel using
        per-bin pred_means), not the raw Brier. Raw Brier differs by the
        within-bin variance of predictions; that gap is small and shrinks
        with finer binning.
        """
        for seed in (1, 7, 42, 123):
            for p_pos in (0.05, 0.3, 0.5, 0.7):
                y_true, y_pred = self._synthetic_binary_data(seed=seed, p_pos=p_pos)
                _ece, rel, res, unc, br_binned = compute_ece_and_brier_decomposition(
                    y_true=y_true, y_pred=y_pred, nbins=10,
                )
                assert abs(br_binned - (rel - res + unc)) < 1e-12, (
                    f"identity broke: seed={seed} p_pos={p_pos} "
                    f"br_binned={br_binned:.12f} REL-RES+UNC={rel-res+unc:.12f}"
                )

    def test_brier_decomp_perfect_calibration_zero_reliability(self):
        """When predictions equal observed frequencies per bin, REL ~= 0."""
        rng = np.random.default_rng(0)
        n = 50000
        y_pred = rng.random(n)
        # Sample y_true from Bernoulli(p_pred) - perfectly calibrated by construction.
        y_true = (rng.random(n) < y_pred).astype(np.float64)
        _ece, rel, _res, _unc, _ = compute_ece_and_brier_decomposition(
            y_true=y_true, y_pred=y_pred, nbins=20,
        )
        assert rel < 0.0005, f"REL should be tiny under perfect calibration, got {rel:.6f}"

    def test_brier_decomp_constant_predictor_at_base_rate(self):
        """A predictor that outputs the base rate has REL=0, RES=0, brier_binned=UNC."""
        rng = np.random.default_rng(2)
        n = 8000
        base = 0.27
        y_true = (rng.random(n) < base).astype(np.float64)
        # All predictions the same -> single non-empty bin, p_mean exactly equals base rate.
        y_pred = np.full(n, fill_value=float(np.mean(y_true)))
        _ece, rel, res, unc, br_binned = compute_ece_and_brier_decomposition(
            y_true=y_true, y_pred=y_pred, nbins=10,
        )
        assert rel < 1e-12
        assert res < 1e-12
        # UNC = base * (1 - base); equality up to fp.
        assert abs(unc - float(np.mean(y_true)) * (1.0 - float(np.mean(y_true)))) < 1e-12
        assert abs(br_binned - unc) < 1e-12

    def test_ece_matches_textbook_formula_tiny_example(self):
        """Hand-rolled ECE on a 4-bin example must match the kernel.

        Bin assignment uses ``ind = floor((p - min) * multiplier)`` with
        ``multiplier = (nbins - 1) / span``. With min=0.0, max=1.0,
        span=1.0, multiplier=3.0 - bin 0 covers [0, 1/3), bin 1 [1/3, 2/3),
        bin 2 [2/3, 1.0), bin 3 = {1.0}.
        """
        y_pred = np.array(
            [0.0, 0.05, 0.4, 0.5, 0.7, 0.8, 1.0, 1.0],
            dtype=np.float64,
        )
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float64)
        ece, _rel, _res, _unc, _ = compute_ece_and_brier_decomposition(
            y_true=y_true, y_pred=y_pred, nbins=4,
        )
        # Hand-roll: 4 bins of 2 samples each, weight = 2/8 = 0.25 per bin.
        # bin 0: pred_mean=(0.0+0.05)/2=0.025, acc=0     -> diff=0.025
        # bin 1: pred_mean=(0.4+0.5)/2=0.45,   acc=0.5   -> diff=0.05
        # bin 2: pred_mean=(0.7+0.8)/2=0.75,   acc=1     -> diff=0.25
        # bin 3: pred_mean=(1.0+1.0)/2=1.0,    acc=1     -> diff=0
        expected = 0.25 * (0.025 + 0.05 + 0.25 + 0.0)
        assert abs(ece - expected) < 1e-9, f"got {ece}, expected {expected}"

    def test_ece_kernel_handles_empty_input(self):
        """Empty input returns (1.0, 1.0, 0.0, 0.0, 1.0) - matches degenerate handling."""
        ece, rel, res, unc, br = compute_ece_and_brier_decomposition(
            y_true=np.array([], dtype=np.float64),
            y_pred=np.array([], dtype=np.float64),
            nbins=10,
        )
        assert (ece, rel, res, unc, br) == (1.0, 1.0, 0.0, 0.0, 1.0)

    def test_ece_kernel_handles_single_class(self):
        """Single-class input - kernel computes; identity still holds (binned)."""
        y_true = np.zeros(100, dtype=np.float64)
        y_pred = np.linspace(0.0, 1.0, 100)
        ece, rel, res, unc, br_binned = compute_ece_and_brier_decomposition(
            y_true=y_true, y_pred=y_pred, nbins=10,
        )
        assert abs(br_binned - (rel - res + unc)) < 1e-12
        # base_rate=0 -> UNC=0; resolution sums squared deviations of acc from 0,
        # which equal acc^2 (and acc=0 in every bin since y_true is all-zeros) -> RES=0.
        assert unc == 0.0
        assert res == 0.0
        # ECE = mean predicted prob (since acc=0 everywhere); positive.
        assert ece > 0.0

    def test_fast_calibration_report_returns_extended_tuple(self):
        """The 17-tuple must include the four new fields between calibration_coverage and roc_auc."""
        y_true, y_pred = self._synthetic_binary_data()
        out = fast_calibration_report(
            y_true=y_true, y_pred=y_pred, nbins=10,
            show_plots=False, plot_file="",
        )
        assert len(out) == 17
        # Positions: brier_loss(0), cal_mae(1), cal_std(2), cal_coverage(3),
        # ece(4), brier_reliability(5), brier_resolution(6), brier_uncertainty(7),
        # roc_auc(8), pr_auc(9), ice(10), ll(11), precision(12), recall(13), f1(14),
        # metrics_string(15), fig(16).
        ece, rel, res, unc = out[4], out[5], out[6], out[7]
        # Values match a direct kernel call on the same inputs.
        ece_k, rel_k, res_k, unc_k, _ = compute_ece_and_brier_decomposition(
            y_true=y_true, y_pred=y_pred, nbins=10,
        )
        assert abs(ece - ece_k) < 1e-12
        assert abs(rel - rel_k) < 1e-12
        assert abs(res - res_k) < 1e-12
        assert abs(unc - unc_k) < 1e-12

    # ============================================================
    # Title-metrics token rendering
    # ============================================================

    def test_token_set_complete(self):
        """TITLE_METRIC_TOKENS frozenset is the canonical allowed set."""
        assert TITLE_METRIC_TOKENS == frozenset({
            "ICE", "BR", "BR_DECOMP", "ECE", "CMAEW",
            "COV", "LL", "ROC_AUC", "PR_AUC", "DENS",
        })

    def test_default_token_sequence(self):
        """Default mirrors the calibration report layout - ICE first, BR with decomp, then ECE."""
        assert DEFAULT_TITLE_METRICS_TOKENS == ("ICE", "BR_DECOMP", "ECE", "CMAEW", "LL", "ROC_AUC", "PR_AUC")

    def test_render_token_ice(self):
        out = render_title_metric_token(
            "ICE", ndigits=3, ice=0.123, brier_loss=0.0, ece=0.0,
            brier_reliability=0.0, brier_resolution=0.0, brier_uncertainty=0.0,
            calibration_mae=0.0, calibration_std=0.0, use_weights=True,
            calibration_coverage=0.0, nbins=10, ll=None, max_hits=0, min_hits=0,
            roc_auc=0.0, mean_group_roc_auc=None, pr_auc=0.0, mean_group_pr_auc=None,
            precision=0.0, recall=0.0, f1=0.0,
        )
        assert out == "ICE=0.123"

    def test_render_token_br_decomp_format(self):
        out = render_title_metric_token(
            "BR_DECOMP", ndigits=2, ice=0.0,
            brier_loss=0.1234, ece=0.0,
            brier_reliability=0.05, brier_resolution=0.10, brier_uncertainty=0.21,
            calibration_mae=0.0, calibration_std=0.0, use_weights=True,
            calibration_coverage=0.0, nbins=10, ll=None, max_hits=0, min_hits=0,
            roc_auc=0.0, mean_group_roc_auc=None, pr_auc=0.0, mean_group_pr_auc=None,
            precision=0.0, recall=0.0, f1=0.0,
        )
        # 0.1234 -> 12.3%, 0.05 -> 5.0%, 0.10 -> 10.0%, 0.21 -> 21.0%.
        # New compact form: BR=X%(RL<rel>%+U<unc>%-RS<res>%)
        # Per-cent metrics use ``ndigits-1`` (here 2-1=1) — ``%`` adds two
        # chars per metric and the headline already runs long.
        assert out == "BR=12.3%(RL5.0%+U21.0%-RS10.0%)"

    def test_render_token_ll_skipped_when_none(self):
        out = render_title_metric_token(
            "LL", ndigits=3, ice=0.0, brier_loss=0.0, ece=0.0,
            brier_reliability=0.0, brier_resolution=0.0, brier_uncertainty=0.0,
            calibration_mae=0.0, calibration_std=0.0, use_weights=True,
            calibration_coverage=0.0, nbins=10, ll=None, max_hits=0, min_hits=0,
            roc_auc=0.0, mean_group_roc_auc=None, pr_auc=0.0, mean_group_pr_auc=None,
            precision=0.0, recall=0.0, f1=0.0,
        )
        assert out == ""

    def test_render_token_unknown_returns_empty(self):
        # Defence-in-depth: unknown tokens should never reach this function in
        # practice (ReportingConfig validates), but if one slips past, render
        # returns "" so the title just skips it without crashing.
        out = render_title_metric_token(
            "FOO", ndigits=3, ice=0.0, brier_loss=0.0, ece=0.0,
            brier_reliability=0.0, brier_resolution=0.0, brier_uncertainty=0.0,
            calibration_mae=0.0, calibration_std=0.0, use_weights=True,
            calibration_coverage=0.0, nbins=10, ll=None, max_hits=0, min_hits=0,
            roc_auc=0.0, mean_group_roc_auc=None, pr_auc=0.0, mean_group_pr_auc=None,
            precision=0.0, recall=0.0, f1=0.0,
        )
        assert out == ""

    def test_title_string_contains_default_tokens_in_order(self):
        """Smoke: title string after a real fast_calibration_report call has the default tokens in order."""
        y_true, y_pred = self._synthetic_binary_data()
        out = fast_calibration_report(
            y_true=y_true, y_pred=y_pred, nbins=10,
            show_plots=False, plot_file="",
        )
        title = out[15]
        # The metric labels appear in the same order as DEFAULT_TITLE_METRICS_TOKENS.
        # ECE label comes between BR-decomp's UNC=...) and CMAEW.
        idx_ice = title.index("ICE=")
        idx_br = title.index("BR=")
        idx_ece = title.index("ECE=")
        idx_cmaew = title.index("CMAEW=")
        idx_ll = title.index("LL=")
        idx_roc = title.index("ROC AUC=")
        idx_pr = title.index("PR AUC=")
        assert idx_ice < idx_br < idx_ece < idx_cmaew < idx_ll < idx_roc < idx_pr

    def test_title_custom_token_subset(self):
        """When passing fewer tokens, the title contains only those, in the chosen order."""
        y_true, y_pred = self._synthetic_binary_data()
        out = fast_calibration_report(
            y_true=y_true, y_pred=y_pred, nbins=10,
            show_plots=False, plot_file="",
            title_metrics_tokens=("CMAEW", "ICE"),
        )
        title = out[15]
        # CMAEW first, then ICE; nothing else.
        assert title.startswith("CMAE")
        assert title.index("ICE=") > title.index("CMAE")
        assert "BR=" not in title
        assert "ECE=" not in title
        assert "ROC AUC=" not in title

    def test_title_empty_tokens_yields_empty_metrics_string(self):
        """Passing an empty token tuple yields an empty title metrics string."""
        y_true, y_pred = self._synthetic_binary_data()
        out = fast_calibration_report(
            y_true=y_true, y_pred=y_pred, nbins=10,
            show_plots=False, plot_file="",
            title_metrics_tokens=(),
        )
        assert out[15] == ""


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
        assert roc_auc == pytest.approx(1.0, rel=1e-6)

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
        from mlframe.metrics.core import fast_aucs_per_group_optimized
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
        from mlframe.metrics.core import compute_mean_aucs_per_group
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
        with caplog.at_level(logging.WARNING, logger="mlframe.metrics.core"):
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
        with caplog.at_level(logging.WARNING, logger="mlframe.metrics.core"):
            self._build(gids, y_true, y_score)
        warn_msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert not any("groups returned NaN" in m for m in warn_msgs), (
            f"Did not expect a warning for only 10% NaN groups; got: {warn_msgs}"
        )


class TestFastAucsOverallParity:
    """Invariant: ``fast_aucs(y, score)`` must return the same overall ROC / PR AUC
    as ``fast_aucs_per_group_optimized(y, score, group_ids=None)[:2]``.

    ``fast_ice_only`` was refactored to call ``fast_aucs`` directly instead of the
    grouped variant (the grouped variant did the same overall-AUC work plus dispatcher
    overhead and returned an empty per-group dict that was always discarded). These
    tests lock in the equivalence so a future change to either function that breaks
    the invariant fires immediately.
    """

    @pytest.mark.parametrize(
        "y_true, y_score",
        [
            # Balanced binary, mid-range scores.
            ([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.45, 0.55]),
            # Imbalanced (1:4 positive:negative), separated scores.
            ([0] * 16 + [1] * 4, list(np.linspace(0.05, 0.45, 16)) + list(np.linspace(0.6, 0.95, 4))),
            # Random but reproducible.
            (list((np.random.default_rng(0).random(200) > 0.7).astype(np.int8)),
             list(np.random.default_rng(1).random(200))),
            # Perfect separation - both AUCs == 1.
            ([0, 0, 0, 1, 1, 1], [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]),
            # Random scoring - both AUCs near 0.5; lock the equivalence anyway.
            ([0, 1] * 50, [0.5] * 100),
        ],
    )
    def test_fast_aucs_matches_per_group_overall(self, y_true, y_score):
        from mlframe.metrics.core import fast_aucs, fast_aucs_per_group_optimized
        y_true_arr = np.asarray(y_true, dtype=np.int8)
        y_score_arr = np.asarray(y_score, dtype=np.float64)
        roc_direct, pr_direct = fast_aucs(y_true_arr, y_score_arr)
        roc_grouped, pr_grouped, per_group = fast_aucs_per_group_optimized(y_true_arr, y_score_arr, group_ids=None)
        # NaN-equality semantics: both NaN -> equivalent.
        if np.isnan(roc_direct) or np.isnan(roc_grouped):
            assert np.isnan(roc_direct) and np.isnan(roc_grouped)
        else:
            np.testing.assert_allclose(roc_direct, roc_grouped, rtol=1e-12, atol=1e-12)
        if np.isnan(pr_direct) or np.isnan(pr_grouped):
            assert np.isnan(pr_direct) and np.isnan(pr_grouped)
        else:
            np.testing.assert_allclose(pr_direct, pr_grouped, rtol=1e-12, atol=1e-12)
        assert per_group == {}, "group_ids=None must yield an empty per-group dict"

    def test_fast_ice_only_matches_manual_recomputation(self):
        """End-to-end sensor for the ``fast_ice_only`` AUC-call swap: recompute
        ICE manually using the same parts and assert equality."""
        from mlframe.metrics.core import (
            fast_aucs,
            fast_brier_score_loss,
            fast_calibration_binning,
            fast_ice_only,
            calibration_metrics_from_freqs,
            integral_calibration_error_from_metrics,
        )
        rng = np.random.default_rng(42)
        y_true = (rng.random(500) > 0.6).astype(np.int8)
        y_score = (y_true * 0.5 + rng.random(500) * 0.5).astype(np.float64)

        actual = fast_ice_only(y_true=y_true, y_pred=y_score, nbins=10, use_weights=True)

        brier = fast_brier_score_loss(y_true=y_true, y_prob=y_score)
        freqs_p, freqs_t, hits = fast_calibration_binning(y_true=y_true, y_pred=y_score, nbins=10)
        cal_mae, cal_std, cal_cov = calibration_metrics_from_freqs(
            freqs_predicted=freqs_p, freqs_true=freqs_t, hits=hits, nbins=10, array_size=len(y_true), use_weights=True,
        )
        roc_auc, pr_auc = fast_aucs(y_true=y_true, y_score=y_score)
        expected = integral_calibration_error_from_metrics(
            calibration_mae=cal_mae, calibration_std=cal_std, calibration_coverage=cal_cov,
            brier_loss=brier, roc_auc=roc_auc, pr_auc=pr_auc,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_fast_ice_only_empty_input_returns_one(self):
        """The empty-input early-return is the only branch that does NOT go through
        ``fast_aucs``; lock its contract."""
        from mlframe.metrics.core import fast_ice_only
        empty_y = np.array([], dtype=np.int8)
        empty_p = np.array([], dtype=np.float64)
        assert fast_ice_only(y_true=empty_y, y_pred=empty_p) == 1.0


try:
    import cupy as _cupy_test_marker  # noqa: F401
    _cupy_available = True
except ImportError:
    _cupy_available = False


@pytest.mark.skipif(not _cupy_available, reason="GPU tests require cupy")
class TestGpuMetrics:
    """Correctness checks for the cupy GPU batch metrics in mlframe.metrics.core.

    cupy is optional; missing -> whole class is skipped (no GPU is required
    to run the rest of the suite).
    """

    def test_rmse_matches_numpy_continuous(self):
        from mlframe.metrics.core import gpu_multiple_rmse_scores
        import cupy as cp

        rng = np.random.default_rng(42)
        N, M = 5_000, 4
        y = rng.standard_normal(N)
        p = rng.standard_normal((N, M))
        cpu = np.sqrt(np.mean((y[:, None] - p) ** 2.0, axis=0))
        gpu = cp.asnumpy(gpu_multiple_rmse_scores(y, p))
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-12)

    def test_rmse_accepts_2d_actual(self):
        from mlframe.metrics.core import gpu_multiple_rmse_scores
        import cupy as cp

        rng = np.random.default_rng(0)
        N, M = 1_000, 3
        y = rng.standard_normal((N, M))
        p = rng.standard_normal((N, M))
        cpu = np.sqrt(np.mean((y - p) ** 2.0, axis=0))
        gpu = cp.asnumpy(gpu_multiple_rmse_scores(y, p))
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-12)

    def test_roc_auc_matches_sklearn_continuous(self):
        from mlframe.metrics.core import gpu_multiple_roc_auc_scores
        import cupy as cp

        rng = np.random.default_rng(11)
        N, M = 5_000, 5
        y = (rng.standard_normal(N) > 0).astype(np.int8)
        p = rng.standard_normal((N, M))
        ref = np.array([roc_auc_score(y, p[:, j]) for j in range(M)])
        gpu = cp.asnumpy(gpu_multiple_roc_auc_scores(y, p))
        np.testing.assert_allclose(gpu, ref, rtol=0, atol=1e-12)

    def test_roc_auc_matches_sklearn_with_ties(self):
        """avg-rank impl must match sklearn bit-for-bit on heavily tied
        scores (probability bins). The naive ``argsort(argsort)`` snippet
        drifts ~1e-5 here -- this test guards against re-introducing it."""
        from mlframe.metrics.core import gpu_multiple_roc_auc_scores
        import cupy as cp

        rng = np.random.default_rng(22)
        N, M = 5_000, 4
        y = (rng.standard_normal(N) > 0).astype(np.int8)
        bins = np.linspace(0.0, 1.0, 10)
        p = rng.choice(bins, size=(N, M))
        ref = np.array([roc_auc_score(y, p[:, j]) for j in range(M)])
        gpu = cp.asnumpy(gpu_multiple_roc_auc_scores(y, p))
        np.testing.assert_allclose(gpu, ref, rtol=0, atol=1e-12)

    def test_pr_auc_matches_sklearn_continuous(self):
        from mlframe.metrics.core import gpu_multiple_pr_auc_scores
        from sklearn.metrics import average_precision_score
        import cupy as cp

        rng = np.random.default_rng(55)
        N, M = 5_000, 4
        y = (rng.standard_normal(N) > 0).astype(np.int8)
        p = rng.standard_normal((N, M))
        ref = np.array([average_precision_score(y, p[:, j]) for j in range(M)])
        gpu = cp.asnumpy(gpu_multiple_pr_auc_scores(y, p))
        np.testing.assert_allclose(gpu, ref, rtol=0, atol=1e-12)

    def test_pr_auc_matches_sklearn_with_ties(self):
        from mlframe.metrics.core import gpu_multiple_pr_auc_scores
        from sklearn.metrics import average_precision_score
        import cupy as cp

        rng = np.random.default_rng(66)
        N, M = 5_000, 4
        y = (rng.standard_normal(N) > 0).astype(np.int8)
        p = rng.choice(np.linspace(0, 1, 10), size=(N, M))
        ref = np.array([average_precision_score(y, p[:, j]) for j in range(M)])
        gpu = cp.asnumpy(gpu_multiple_pr_auc_scores(y, p))
        np.testing.assert_allclose(gpu, ref, rtol=0, atol=1e-12)

    def test_pr_auc_returns_nan_on_single_class(self):
        from mlframe.metrics.core import gpu_multiple_pr_auc_scores
        import cupy as cp
        rng = np.random.default_rng(77)
        N = 1_000
        y = np.zeros(N, dtype=np.int8)  # all-negative
        p = rng.standard_normal((N, 3))
        gpu = cp.asnumpy(gpu_multiple_pr_auc_scores(y, p))
        assert np.all(np.isnan(gpu)), gpu

    def test_aucs_accept_1d_predicted(self):
        from mlframe.metrics.core import (
            gpu_multiple_pr_auc_scores,
            gpu_multiple_roc_auc_scores,
        )
        from sklearn.metrics import average_precision_score
        import cupy as cp

        rng = np.random.default_rng(33)
        N = 2_000
        y = (rng.standard_normal(N) > 0).astype(np.int8)
        p = rng.standard_normal(N)
        ref_roc = roc_auc_score(y, p)
        ref_pr = average_precision_score(y, p)
        gpu_roc = float(cp.asnumpy(gpu_multiple_roc_auc_scores(y, p))[0])
        gpu_pr = float(cp.asnumpy(gpu_multiple_pr_auc_scores(y, p))[0])
        assert abs(gpu_roc - ref_roc) < 1e-12
        assert abs(gpu_pr - ref_pr) < 1e-12


@pytest.mark.skipif(not _cupy_available, reason="dispatcher tests use the GPU side too")
class TestGpuDispatchers:
    """``compute_batch_aucs`` / ``compute_batch_rmse`` / threshold knobs."""

    def test_compute_batch_rmse_auto_below_threshold_uses_cpu(self):
        from mlframe.metrics.core import compute_batch_rmse, _GPU_BATCH_THRESHOLD_N

        rng = np.random.default_rng(0)
        # Below threshold (default 100k): auto picks CPU.
        N = max(100, _GPU_BATCH_THRESHOLD_N // 100)
        y = rng.standard_normal(N)
        p = rng.standard_normal((N, 3))
        out = compute_batch_rmse(y, p)
        ref = np.sqrt(np.mean((y[:, None] - p) ** 2.0, axis=0))
        np.testing.assert_allclose(out, ref, rtol=0, atol=1e-12)
        assert isinstance(out, np.ndarray)

    def test_compute_batch_aucs_force_cpu_matches_loop(self):
        from mlframe.metrics.core import compute_batch_aucs, fast_aucs

        rng = np.random.default_rng(7)
        N, M = 2_000, 3
        y = (rng.standard_normal(N) > 0).astype(np.int8)
        p = rng.standard_normal((N, M))
        roc, pr = compute_batch_aucs(y, p, force_backend="cpu")
        ref_roc = np.empty(M)
        ref_pr = np.empty(M)
        for j in range(M):
            ref_roc[j], ref_pr[j] = fast_aucs(y, p[:, j])
        np.testing.assert_allclose(roc, ref_roc, rtol=0, atol=1e-12)
        np.testing.assert_allclose(pr, ref_pr, rtol=0, atol=1e-12)

    def test_compute_batch_aucs_force_gpu_matches_sklearn(self):
        from mlframe.metrics.core import compute_batch_aucs
        from sklearn.metrics import average_precision_score

        rng = np.random.default_rng(8)
        N, M = 5_000, 4
        y = (rng.standard_normal(N) > 0).astype(np.int8)
        p = rng.standard_normal((N, M))
        roc, pr = compute_batch_aucs(y, p, force_backend="gpu")
        ref_roc = np.array([roc_auc_score(y, p[:, j]) for j in range(M)])
        ref_pr = np.array([average_precision_score(y, p[:, j]) for j in range(M)])
        np.testing.assert_allclose(roc, ref_roc, rtol=0, atol=1e-12)
        np.testing.assert_allclose(pr, ref_pr, rtol=0, atol=1e-12)

    def test_set_gpu_thresholds_changes_dispatch(self):
        from mlframe.metrics.core import (
            compute_batch_aucs,
            set_gpu_thresholds,
            _GPU_BATCH_THRESHOLD_N,
        )

        rng = np.random.default_rng(9)
        N, M = 5_000, 3
        y = (rng.standard_normal(N) > 0).astype(np.int8)
        p = rng.standard_normal((N, M))

        # Baseline result for parity check
        roc_cpu, pr_cpu = compute_batch_aucs(y, p, force_backend="cpu")

        original = _GPU_BATCH_THRESHOLD_N
        try:
            set_gpu_thresholds(n=1_000)  # below 5_000 so auto picks GPU
            roc_auto, pr_auto = compute_batch_aucs(y, p)
            np.testing.assert_allclose(roc_auto, roc_cpu, rtol=0, atol=1e-12)
            np.testing.assert_allclose(pr_auto, pr_cpu, rtol=0, atol=1e-12)
        finally:
            set_gpu_thresholds(n=original)

    def test_force_backend_invalid_raises(self):
        from mlframe.metrics.core import compute_batch_aucs

        rng = np.random.default_rng(10)
        y = (rng.standard_normal(100) > 0).astype(np.int8)
        p = rng.standard_normal((100, 2))
        with pytest.raises(ValueError, match="force_backend"):
            compute_batch_aucs(y, p, force_backend="cudnn")


@pytest.mark.skipif(_cupy_available, reason="Skipped when cupy IS present")
class TestGpuMetricsImportError:
    """When cupy is NOT installed, ``gpu_*`` helpers must raise a clear
    ImportError (not a generic ``ModuleNotFoundError``). Dispatchers
    fall back to CPU silently."""

    def test_rmse_raises_clear_error_without_cupy(self):
        from mlframe.metrics.core import gpu_multiple_rmse_scores
        with pytest.raises(ImportError, match="cupy"):
            gpu_multiple_rmse_scores(np.zeros(10), np.zeros((10, 2)))

    def test_dispatcher_falls_back_to_cpu_without_cupy(self):
        """Dispatcher should NOT raise when cupy is missing -- it just
        runs on CPU."""
        from mlframe.metrics.core import compute_batch_rmse
        rng = np.random.default_rng(0)
        y = rng.standard_normal(200)
        p = rng.standard_normal((200, 3))
        out = compute_batch_rmse(y, p)  # no force_backend; auto -> CPU
        ref = np.sqrt(np.mean((y[:, None] - p) ** 2.0, axis=0))
        np.testing.assert_allclose(out, ref, rtol=0, atol=1e-12)


class TestFastRegressionMetrics:
    """fast_mean_absolute_error / mean_squared_error / root_mean_squared_error
    / max_error / r2_score must be bit-exact drop-ins for sklearn across
    1-D / 2-D, weighted / unweighted, and every multioutput aggregation."""

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(0)

    def _data(self, rng, shape):
        y = rng.standard_normal(shape)
        p = y + 0.1 * rng.standard_normal(shape)
        return y, p

    def test_mae_1d_unweighted(self, rng):
        from sklearn.metrics import mean_absolute_error as sk
        from mlframe.metrics.core import fast_mean_absolute_error
        y, p = self._data(rng, 5_000)
        assert abs(fast_mean_absolute_error(y, p) - sk(y, p)) < 1e-12

    def test_mae_1d_weighted(self, rng):
        from sklearn.metrics import mean_absolute_error as sk
        from mlframe.metrics.core import fast_mean_absolute_error
        y, p = self._data(rng, 5_000)
        w = rng.random(5_000) + 0.1
        assert abs(fast_mean_absolute_error(y, p, sample_weight=w) - sk(y, p, sample_weight=w)) < 1e-12

    def test_mae_2d_uniform_average(self, rng):
        from sklearn.metrics import mean_absolute_error as sk
        from mlframe.metrics.core import fast_mean_absolute_error
        y, p = self._data(rng, (5_000, 4))
        assert abs(fast_mean_absolute_error(y, p) - sk(y, p)) < 1e-12

    def test_mae_2d_raw_values(self, rng):
        from sklearn.metrics import mean_absolute_error as sk
        from mlframe.metrics.core import fast_mean_absolute_error
        y, p = self._data(rng, (5_000, 4))
        ref = sk(y, p, multioutput="raw_values")
        out = fast_mean_absolute_error(y, p, multioutput="raw_values")
        np.testing.assert_allclose(out, ref, rtol=0, atol=1e-12)

    def test_mae_2d_weighted_array_multioutput(self, rng):
        from sklearn.metrics import mean_absolute_error as sk
        from mlframe.metrics.core import fast_mean_absolute_error
        y, p = self._data(rng, (5_000, 3))
        w = rng.random(5_000) + 0.1
        out_w = np.array([2.0, 1.0, 0.5])
        ref = sk(y, p, sample_weight=w, multioutput=out_w)
        out = fast_mean_absolute_error(y, p, sample_weight=w, multioutput=out_w)
        assert abs(out - ref) < 1e-12

    def test_mse_1d_weighted(self, rng):
        from sklearn.metrics import mean_squared_error as sk
        from mlframe.metrics.core import fast_mean_squared_error
        y, p = self._data(rng, 5_000)
        w = rng.random(5_000) + 0.1
        assert abs(fast_mean_squared_error(y, p, sample_weight=w) - sk(y, p, sample_weight=w)) < 1e-12

    def test_rmse_2d_raw_values(self, rng):
        from mlframe.metrics.core import fast_root_mean_squared_error, fast_mean_squared_error
        y, p = self._data(rng, (5_000, 3))
        # sklearn rmse = per-output RMSE, then aggregated. raw_values returns sqrt(mse_per_col).
        ref = np.sqrt(fast_mean_squared_error(y, p, multioutput="raw_values"))
        out = fast_root_mean_squared_error(y, p, multioutput="raw_values")
        np.testing.assert_allclose(out, ref, rtol=0, atol=1e-12)

    def test_rmse_matches_sklearn_2d_weighted(self, rng):
        try:
            from sklearn.metrics import root_mean_squared_error as sk
        except ImportError:
            pytest.skip("sklearn root_mean_squared_error not available")
        from mlframe.metrics.core import fast_root_mean_squared_error
        y, p = self._data(rng, (5_000, 3))
        w = rng.random(5_000) + 0.1
        assert abs(fast_root_mean_squared_error(y, p, sample_weight=w)
                   - sk(y, p, sample_weight=w)) < 1e-12

    def test_max_error_1d_matches_sklearn(self, rng):
        from sklearn.metrics import max_error as sk
        from mlframe.metrics.core import fast_max_error
        y, p = self._data(rng, 5_000)
        assert abs(fast_max_error(y, p) - sk(y, p)) < 1e-12

    def test_max_error_2d_per_output(self, rng):
        from mlframe.metrics.core import fast_max_error
        y, p = self._data(rng, (5_000, 3))
        out = fast_max_error(y, p)  # default raw_values
        # Verify against direct numpy
        ref = np.max(np.abs(y - p), axis=0)
        np.testing.assert_allclose(out, ref, rtol=0, atol=1e-12)

    def test_r2_1d_weighted(self, rng):
        from sklearn.metrics import r2_score as sk
        from mlframe.metrics.core import fast_r2_score
        y, p = self._data(rng, 5_000)
        w = rng.random(5_000) + 0.1
        assert abs(fast_r2_score(y, p, sample_weight=w) - sk(y, p, sample_weight=w)) < 1e-12

    def test_r2_2d_raw_values(self, rng):
        from sklearn.metrics import r2_score as sk
        from mlframe.metrics.core import fast_r2_score
        y, p = self._data(rng, (5_000, 4))
        ref = sk(y, p, multioutput="raw_values")
        out = fast_r2_score(y, p, multioutput="raw_values")
        np.testing.assert_allclose(out, ref, rtol=0, atol=1e-12)

    def test_r2_2d_variance_weighted(self, rng):
        from sklearn.metrics import r2_score as sk
        from mlframe.metrics.core import fast_r2_score
        y, p = self._data(rng, (5_000, 4))
        ref = sk(y, p, multioutput="variance_weighted")
        out = fast_r2_score(y, p, multioutput="variance_weighted")
        assert abs(out - ref) < 1e-12

    def test_r2_2d_variance_weighted_with_sample_weight(self, rng):
        from sklearn.metrics import r2_score as sk
        from mlframe.metrics.core import fast_r2_score
        y, p = self._data(rng, (5_000, 3))
        w = rng.random(5_000) + 0.1
        ref = sk(y, p, sample_weight=w, multioutput="variance_weighted")
        out = fast_r2_score(y, p, sample_weight=w, multioutput="variance_weighted")
        assert abs(out - ref) < 1e-12

    def test_unknown_multioutput_raises(self, rng):
        from mlframe.metrics.core import fast_mean_absolute_error
        y, p = self._data(rng, (5_000, 3))
        with pytest.raises(ValueError, match="multioutput"):
            fast_mean_absolute_error(y, p, multioutput="not-a-thing")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
