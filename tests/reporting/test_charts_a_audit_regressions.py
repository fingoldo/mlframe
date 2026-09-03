"""Regression tests for the reporting_charts_a audit findings."""

import numpy as np
import pytest

from mlframe.reporting.charts.binary import _ScoreSort, _finite_binary, _ks_curve, _score_dist_panel, bootstrap_ap_ci
from mlframe.reporting.charts.calibration_drift import build_calibration_drift_spec, calibration_drift
from mlframe.reporting.charts.category_discriminability import category_discriminability_table


class TestKsUsesRealisableThresholds:
    """Stepping the ECDFs through a tied block evaluates at cut points no threshold can produce."""

    @pytest.mark.parametrize("levels", [20, 5])
    def test_ks_matches_a_realisable_threshold_reference_on_quantised_scores(self, levels):
        """A tree ensemble emits heavily quantised scores, which is exactly where the bias showed."""
        rng = np.random.default_rng(0)
        n = 200_000
        scores = np.round(rng.random(n) * levels) / float(levels)
        labels = (rng.random(n) < np.clip(scores, 0.0, 1.0)).astype(np.int8)
        _, _, _, ks, _ = _ks_curve(_ScoreSort(labels, scores))
        pos, neg = np.sort(scores[labels == 1]), np.sort(scores[labels == 0])
        thresholds = np.unique(scores)
        reference = np.abs(np.searchsorted(neg, thresholds, side="right") / neg.size - np.searchsorted(pos, thresholds, side="right") / pos.size).max()
        # Pre-fix, mid-tie ranks biased this UPWARD (0.494441 against a 0.493870 reference at 20 levels).
        assert ks == pytest.approx(reference, abs=1e-12)


class TestApBootstrapInterval:
    """A full-n point estimate must not carry the uncertainty of a subsampled study."""

    def test_interval_matches_an_uncapped_bootstrap(self, monkeypatch):
        """Rescaling by sqrt(m/n) recovers the width the uncapped bootstrap would have produced."""
        import mlframe.reporting.charts.binary as binary_mod

        rng = np.random.default_rng(0)
        n = 200_000
        scores = rng.random(n)
        labels = (rng.random(n) < np.clip(scores * 0.4, 0.0, 1.0)).astype(np.int8)
        _, lo, hi = bootstrap_ap_ci(labels, scores, n_boot=300)
        monkeypatch.setattr(binary_mod, "_AP_BOOTSTRAP_ROW_CAP", n)
        _, lo_full, hi_full = bootstrap_ap_ci(labels, scores, n_boot=300)
        # Pre-fix this ratio was ~2 at n=200k against the 50k cap, and ~6 at n=2M.
        assert (hi - lo) / (hi_full - lo_full) == pytest.approx(1.0, abs=0.15)

    def test_interval_narrows_as_n_grows(self):
        """A bootstrap standard error scales as 1/sqrt(rows); the reported width must too."""
        rng = np.random.default_rng(0)
        widths = []
        for n in (200_000, 2_000_000):
            scores = rng.random(n)
            labels = (rng.random(n) < np.clip(scores * 0.4, 0.0, 1.0)).astype(np.int8)
            _, lo, hi = bootstrap_ap_ci(labels, scores, n_boot=150)
            widths.append(hi - lo)
        assert widths[1] < widths[0] * 0.6


class TestOutOfRangeLabelsAreLoud:
    """Binary panels are one-vs-rest by definition, so dropping is right -- doing it silently is not."""

    def test_dropping_non_binary_labels_warns_with_the_count(self, caplog):
        """A mis-passed multiclass target produced confident curves on a silent subset."""
        labels = np.array([0, 1, 2, 1, 0, 3], dtype=float)
        scores = np.array([0.1, 0.9, 0.5, 0.8, 0.2, 0.4])
        with caplog.at_level("WARNING"):
            kept_labels, kept_scores = _finite_binary(labels, scores)
        assert kept_labels.size == 4 and kept_scores.size == 4
        assert any("outside {0, 1}" in r.message for r in caplog.records)


class TestScoreDistributionShowsImbalance:
    """Two densities each integrate to 1, so a 1000:1 imbalance draws two equally tall humps."""

    def test_labels_carry_counts_and_shares(self):
        """The counts are the only place the imbalance becomes visible."""
        rng = np.random.default_rng(0)
        n = 100_000
        labels = (rng.random(n) < 0.002).astype(np.int8)
        scores = rng.random(n)
        panel = _score_dist_panel(labels, scores, sort=_ScoreSort(labels, scores), threshold=0.5)
        assert all("n=" in lab for lab in panel.series_labels)
        assert "0.2%" in panel.series_labels[1]


class TestCalibrationDriftReadability:
    """The module exists to answer "where did calibration bend"."""

    def _result(self):
        """A drift result over datetime windows."""
        rng = np.random.default_rng(0)
        n = 20_000
        scores = rng.random(n)
        labels = (rng.random(n) < scores).astype(float)
        stamps = np.datetime64("2024-01-01") + rng.integers(0, 300, n).astype("timedelta64[D]")
        return calibration_drift(labels, scores, stamps, n_windows=8)

    def test_the_promised_noise_floor_band_exists(self):
        """The docstring promised a band; without it a thin window looks like a drifting one."""
        panel = build_calibration_drift_spec(self._result()).panels[0][0]
        assert panel.band is not None
        assert "noise floor" in (panel.band_label or "")

    def test_windows_are_labelled_by_date_not_index(self):
        """ "w3" cannot be mapped back to a period, which is the whole question."""
        figure = build_calibration_drift_spec(self._result())
        if len(figure.panels) < 2:
            pytest.skip("reliability small-multiple not built for this fixture")
        labels = figure.panels[1][0].series_labels
        assert any("2024-" in lab for lab in labels)
        assert not any(lab.startswith("w") and lab[1:].isdigit() for lab in labels)


class TestTopKIsValidated:
    """Silently promoting 0 to 1 answers a different question than the one asked."""

    def test_zero_raises(self):
        """A caller requesting zero rows got exactly one."""
        import pandas as pd

        rng = np.random.default_rng(0)
        n = 5000
        frame = pd.DataFrame({"c": rng.choice(list("abc"), n)})
        target = (rng.random(n) < 0.3).astype(float)
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            category_discriminability_table(frame, target, top_k=0)
