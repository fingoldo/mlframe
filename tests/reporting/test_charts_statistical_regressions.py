"""Regression tests for the statistical defects the reporting audit confirmed live.

Each test reproduces the exact regime in which a chart stated something false about the data, and would fail on
the pre-fix code with the reported wrong value. They are grouped by the bug CLASS rather than by module, because
several of these classes recur across builders and this file is where the next instance should land.
"""

import numpy as np
import pytest

from mlframe.reporting.charts.binary import _ScoreSort, _pit_panel
from mlframe.reporting.charts.calibration_by_feature import compute_calibration_by_feature_heterogeneity
from mlframe.reporting.charts.calibration_heatmap_2d import compute_calibration_heatmap_2d
from mlframe.reporting.charts.decision_curve import build_decision_curve_spec, effective_binary_n
from mlframe.reporting.charts.drift import (
    CUSUM_DECISION_H,
    _adversarial_auc_bar,
    _cusum_tabular_loop,
    cusum_h_for_length,
    residual_vs_time,
)
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


class TestCusumStaysQuietOnDriftFreeSeries:
    """A fixed decision interval cannot hold a false-alarm rate as the series gets longer."""

    def test_pure_noise_does_not_raise_a_change_point(self):
        """A drift-free series of the length this chart targets must not report a structural break."""
        h = cusum_h_for_length(6000)
        # At the previous fixed h=8 this crossed on 3 of these 4 seeds; Siegmund puts the two-sided ARL_0 there
        # near 9,500, so a 6000-row series false-alarmed about half the time.
        for seed in range(4):
            z = np.random.default_rng(seed).normal(0.0, 1.0, 6000)
            assert _cusum_tabular_loop(z, 0.5, h)[2] == -1

    def test_a_real_sustained_shift_is_still_detected_promptly(self):
        """Raising h to control false alarms must not cost the detection it exists for."""
        h = cusum_h_for_length(6000)
        z = np.random.default_rng(0).normal(0.0, 1.0, 6000)
        z[4000:] += 1.5
        cross = _cusum_tabular_loop(z, 0.5, h)[2]
        assert 4000 <= cross < 4100

    def test_h_grows_with_series_length(self):
        """The decision interval is solved from the length, and never drops below the old fixed default."""
        assert cusum_h_for_length(500) < cusum_h_for_length(6000) < cusum_h_for_length(50_000)
        assert cusum_h_for_length(10) >= CUSUM_DECISION_H


class TestAdversarialVerdictScalesWithRowCount:
    """AUC 0.6 is noise on a few hundred rows per side and a real shift on two hundred thousand."""

    def test_no_shift_bar_shrinks_as_the_sets_grow(self):
        """The bar above 0.5 is the AUC's own null standard error, so it must fall as 1/sqrt(n)."""
        bars = [_adversarial_auc_bar(n, n) for n in (200, 2000, 50_000, 200_000)]
        assert bars == sorted(bars, reverse=True)
        assert bars[0] > 0.05  # a 0.55 AUC on 200 rows/side says nothing
        assert bars[-1] < 0.01  # a 0.52 AUC on 200k rows/side is a real, reproducible shift


class TestPitIsUniformForACalibratedBinaryModel:
    """The plain PIT is only uniform for a CONTINUOUS outcome; a binary one needs the randomised transform."""

    def test_calibrated_model_reads_as_uniform(self):
        """A model calibrated by construction must not be condemned by its own PIT panel."""
        rng = np.random.default_rng(0)
        n = 200_000
        p = rng.random(n)
        y = (rng.random(n) < p).astype(int)  # y ~ Bernoulli(p): perfectly calibrated
        panel = _pit_panel(y, p, sort=_ScoreSort(y, p), threshold=0.5)
        # Pre-fix this was 0.247, with a triangular density rising 0.10 -> 1.90 across the deciles.
        ks = float(panel.title.split("=")[1].rstrip(")"))
        assert ks < 0.02

    def test_the_randomising_draw_cannot_collide_with_the_callers_stream(self):
        """Seeding from the data, not a constant, keeps a caller using default_rng(0) from cancelling the fix."""
        n = 50_000
        for seed in (0, 7, 12345):
            rng = np.random.default_rng(seed)
            p = rng.random(n)
            y = (rng.random(n) < p).astype(int)
            panel = _pit_panel(y, p, sort=_ScoreSort(y, p), threshold=0.5)
            assert float(panel.title.split("=")[1].rstrip(")")) < 0.02

    def test_real_miscalibration_is_still_detected(self):
        """Restoring uniformity must not cost the detection the panel exists for."""
        rng = np.random.default_rng(0)
        n = 200_000
        p = rng.random(n)
        over = (rng.random(n) < np.clip(0.5 + (p - 0.5) * 0.4, 0.0, 1.0)).astype(int)
        panel = _pit_panel(over, p, sort=_ScoreSort(over, p), threshold=0.5)
        assert float(panel.title.split("=")[1].rstrip(")")) > 0.02


class TestPerCellCalibrationCannotCancelOrTrackResolution:
    """A per-cell 'ECE' that is really a mean gap hides the worst cells; a fixed bar grades the grid, not the model."""

    def test_a_cell_whose_miscalibration_cancels_is_still_flagged(self):
        """Half the rows at score 0.9/target 0 and half at 0.1/target 1 is maximally miscalibrated."""
        rng = np.random.default_rng(0)
        n = 20_000
        score = np.where(rng.random(n) < 0.5, 0.9, 0.1)
        y = np.where(score > 0.5, 0.0, 1.0)
        grid = compute_calibration_heatmap_2d(y, score, rng.random(n), rng.random(n), n_bins=3)
        # Pre-fix both means were 0.5, the gap was 0.000 and the panel painted green.
        assert grid["worst_ece"] > 0.5
        assert grid["traffic_light"] == "red"

    @pytest.mark.parametrize("n_bins", [3, 5, 8, 16])
    def test_a_calibrated_model_reads_green_at_every_grid_resolution(self, n_bins):
        """The verdict must describe the model, not how finely the grid happens to be cut."""
        rng = np.random.default_rng(0)
        n = 20_000
        p = rng.random(n)
        y = (rng.random(n) < p).astype(float)
        grid = compute_calibration_heatmap_2d(y, p, rng.random(n), rng.random(n), n_bins=n_bins)
        # Pre-fix the same calibrated model read red at 40 rows/cell and green at 4,000.
        assert grid["traffic_light"] == "green"

    def test_a_real_miscalibrated_pocket_is_still_caught(self):
        """Scaling the bar to the grid must not silence a genuine pocket."""
        rng = np.random.default_rng(0)
        n = 20_000
        p = rng.random(n)
        feat = rng.random(n)
        y = (rng.random(n) < p).astype(float)
        bad = feat > 0.9
        y[bad] = (rng.random(int(bad.sum())) < np.clip(p[bad] * 0.15, 0.0, 1.0)).astype(float)
        grid = compute_calibration_heatmap_2d(y, p, feat, rng.random(n), n_bins=5)
        assert grid["traffic_light"] == "red"


class TestPerBinEceIsDebiased:
    """The plug-in binned ECE is positively biased, and the bias differs per bin because bins differ in size."""

    def test_heterogeneity_of_a_calibrated_model_stays_inside_its_noise_floor(self):
        """Calibration that does not vary with the feature must not read as heterogeneous."""
        rng = np.random.default_rng(0)
        n = 200_000
        p = rng.random(n)
        y = (rng.random(n) < p).astype(float)
        res = compute_calibration_by_feature_heterogeneity(y, p, rng.random(n))
        assert res["heterogeneity"] < res["noise_floor"]
        assert res["traffic_light"] == "green"

    def test_feature_dependent_miscalibration_is_still_flagged(self):
        """A genuine pocket where the model is wrong about one feature range must read red."""
        rng = np.random.default_rng(0)
        n = 200_000
        p = rng.random(n)
        feat = rng.random(n)
        y = (rng.random(n) < p).astype(float)
        bad = feat > 0.8
        y[bad] = (rng.random(int(bad.sum())) < np.clip(p[bad] * 0.2, 0.0, 1.0)).astype(float)
        assert compute_calibration_by_feature_heterogeneity(y, p, feat)["traffic_light"] == "red"
