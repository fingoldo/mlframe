"""Regression tests for the statistical defects the reporting audit confirmed live.

Each test reproduces the exact regime in which a chart stated something false about the data, and would fail on
the pre-fix code with the reported wrong value. They are grouped by the bug CLASS rather than by module, because
several of these classes recur across builders and this file is where the next instance should land.
"""

import numpy as np
import pytest

from mlframe.reporting.charts.decision_curve import build_decision_curve_spec, effective_binary_n
from mlframe.reporting.charts.drift import residual_vs_time
from mlframe.reporting.charts.error_analysis import _resolve_feature_matrix, _target_drift_verdict
from mlframe.reporting.charts.model_comparison import _spearman_corr_matrix
from mlframe.reporting.charts.multiclass import _top_k_acc_panel
from mlframe.reporting.spec import AnnotationPanelSpec


class TestSizeScaledThresholdsUseEffectiveN:
    """A sample-size-scaled bar must be fed the rows actually used, not the rows supplied."""

    def test_decision_curve_margin_uses_rows_that_survive_filtering(self):
        """The usefulness bar is sized from scorable rows, not the length of the input array."""
        rng = np.random.default_rng(0)
        n = 200_000
        y = rng.integers(0, 2, n)
        score = np.full(n, np.nan)
        score[:150] = rng.random(150)  # only 150 rows are actually scorable
        assert effective_binary_n(y, score) == 150
        # Pre-fix the margin came from n=200000 (0.0022) and random noise on 150 rows cleared it.
        assert build_decision_curve_spec(y, score).useful is False

    def test_decision_curve_reports_dropped_rows_in_caption(self):
        """The caption states how many rows were used and how many were dropped."""
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 4000)
        score = rng.random(4000)
        score[:1000] = np.nan
        caption = build_decision_curve_spec(y, score).figure.caption
        assert "3,000 usable rows" in caption
        assert "1,000 of 4,000 dropped" in caption

    def test_decision_curve_with_no_binary_rows_explains_instead_of_drawing_zeros(self):
        """No usable rows yields an explanation, not three flat zero curves."""
        rng = np.random.default_rng(2)
        res = build_decision_curve_spec(np.full(100, 7), rng.random(100))
        panel = res.figure.panels[0][0]
        assert isinstance(panel, AnnotationPanelSpec)
        assert "binary" in panel.text
        assert res.useful is False


class TestTargetDriftScalesWithSplitSize:
    """A rate shift is graded against its own sampling error, not a constant."""

    def test_quadrupled_rare_event_rate_is_flagged(self):
        """A 0.5% -> 2.0% base-rate move is a real shift and must be flagged."""
        train = np.zeros(1000)
        train[:5] = 1.0  # 0.5%
        test = np.zeros(1000)
        test[:20] = 1.0  # 2.0% -- four times as many positives
        # Pre-fix this needed a 0.25 ABSOLUTE move and reported "No material drift from train".
        assert "WARNING" in _target_drift_verdict({"train": train, "test": test}, train_key="train", task="classification")

    def test_tiny_split_of_the_same_distribution_is_not_flagged(self):
        """A 30-row split drawn from the train distribution must not raise a drift warning."""
        rng = np.random.default_rng(3)
        train = rng.integers(0, 2, 200_000).astype(float)
        test = rng.integers(0, 2, 30).astype(float)
        assert "WARNING" not in _target_drift_verdict({"train": train, "test": test}, train_key="train", task="classification")


class TestVarianceIsComputedWithoutCancellation:
    """E[x^2]-E[x]^2 collapses when the residuals sit far from zero relative to their spread."""

    @pytest.mark.parametrize(("centre", "scale"), [(1e5, 1e-4), (1e6, 1e-3), (1e8, 1e-2)])
    def test_residual_band_survives_a_large_offset(self, centre, scale):
        """Per-bucket residual std stays correct when residuals sit far from zero."""
        rng = np.random.default_rng(4)
        n = 50_000
        y_pred = np.full(n, centre)
        y_true = centre + rng.normal(0.0, scale, n)
        panel = residual_vs_time(y_true, y_pred, np.arange(n, dtype=float)).panels[0][0]
        lo, hi = panel.band
        reported_std = (hi - lo) / 2.0
        finite = reported_std[np.isfinite(reported_std)]
        # Pre-fix the clip turned a negative computed variance into a zero-width band on most buckets.
        assert finite.size > 0
        assert np.all(finite > 0.0)
        assert np.allclose(finite, scale, rtol=0.1)


class TestTiesAreNotBrokenByIndexOrder:
    """A zero-information predictor must score at chance regardless of how the classes happen to be ordered."""

    @pytest.mark.parametrize("prevalences", [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [1 / 3, 1 / 3, 1 / 3]])
    def test_uniform_predictor_scores_at_chance(self, prevalences):
        """Top-k accuracy of a zero-information predictor is 1/K under any class ordering."""
        rng = np.random.default_rng(5)
        y = rng.choice(3, 20_000, p=prevalences)
        proba = np.full((20_000, 3), 1 / 3)
        # Pre-fix argsort broke ties by class index: 0.798 for the first ordering, 0.100 for the second.
        assert _top_k_acc_panel(y, proba, [0, 1, 2]).y[0] == pytest.approx(1 / 3)

    def test_two_constant_models_do_not_correlate_perfectly(self):
        """Constant prediction columns correlate as NaN, never as a confident 1.000."""
        rng = np.random.default_rng(6)
        scores = np.column_stack([np.ones(500), np.full(500, 3.0), rng.random(500)])
        corr = _spearman_corr_matrix(scores)
        # Pre-fix ordinal ranks gave the constant columns row-order variance and rho came out exactly 1.000.
        assert np.isnan(corr[0, 1])
        assert corr[2, 2] == pytest.approx(1.0)

    def test_tied_scores_get_average_ranks(self):
        """Tied values share a mid-rank, so Spearman matches the textbook result."""
        # A half-tied column against a strictly increasing one: mid-ranks are the only correct answer.
        scores = np.column_stack([np.array([1.0, 1.0, 2.0, 2.0]), np.arange(4.0)])
        assert _spearman_corr_matrix(scores)[0, 1] == pytest.approx(0.894427, abs=1e-5)


class TestColumnNameMismatchIsLoud:
    """A shape mismatch between data and its labels must be an error, never a silent truncation."""

    def test_short_feature_names_do_not_silently_drop_columns(self):
        """A feature_names/columns length mismatch raises instead of truncating."""
        pd = pytest.importorskip("pandas")
        frame = pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0) * 2})
        # Pre-fix a bare zip() truncated to one column and every downstream diagnostic ran on a subset.
        with pytest.raises(ValueError, match="one-to-one"):
            _resolve_feature_matrix(frame, ["only_one"])
