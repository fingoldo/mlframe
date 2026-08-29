"""Regression tests for the reporting_charts_c audit findings.

The theme of this cluster is a panel reporting a NUMBER where it had nothing to measure: a crossing rate of
0.000% over zero surviving rows, an "agreement" fraction pinned at 0.5 by its own default threshold, a
"(centered)" label over an uncentred curve. Each reads as a measurement, and each was unfalsifiable.
"""

import numpy as np
import pytest

from mlframe.reporting.charts.prediction_stability import compute_prediction_stability
from mlframe.reporting.charts.quantile import _quantile_crossing_panel, compose_quantile_figure
from mlframe.reporting.spec import AnnotationPanelSpec


class TestCrossingPanelDoesNotIssueACleanBillOfHealthFromNoData:
    """Dropping every row then reporting rate 0.0 says "no crossings found", not "nothing could be checked"."""

    def test_all_rows_non_finite_returns_an_annotation_naming_the_cause(self):
        """The old form titled itself "0 rows, 0.000%" -- indistinguishable from a perfect model."""
        panel = _quantile_crossing_panel(np.zeros(500), np.full((500, 3), np.nan), np.array([0.1, 0.5, 0.9]))
        assert isinstance(panel, AnnotationPanelSpec)
        assert "not checked" in panel.text and "500" in panel.text

    def test_a_real_crossing_is_still_reported_with_its_denominator(self):
        """The rate must carry the row count it was computed over, so a thin panel is visibly thin."""
        rng = np.random.default_rng(0)
        preds = np.sort(rng.random((500, 3)), axis=1)
        preds[:20, 1] = preds[:20, 2] + 0.1
        title = _quantile_crossing_panel(np.zeros(500), preds, np.array([0.1, 0.5, 0.9])).title
        assert "20 of 500 rows" in title and "4.000%" in title


class TestComposerValidatesTheAlphaGrid:
    """Three helpers read the grid positionally as ascending; an unsorted one silently inverts the bands."""

    @pytest.fixture
    def data(self):
        """(y, preds) with three quantile columns."""
        rng = np.random.default_rng(0)
        return rng.normal(size=200), np.sort(rng.normal(size=(200, 3)), axis=1)

    @pytest.mark.parametrize("alphas", [(0.9, 0.5, 0.1), (0.1, 0.1, 0.9)])
    def test_a_non_ascending_grid_raises(self, data, alphas):
        """Descending and tied grids both produced negative nominal coverages rather than an error."""
        with pytest.raises(ValueError, match="strictly increasing"):
            compose_quantile_figure(data[0], data[1], alphas)

    @pytest.mark.parametrize("alphas", [(0.1, 0.5, 1.0), (0.0, 0.5, 0.9)])
    def test_a_grid_touching_the_unit_bounds_raises(self, data, alphas):
        """alpha = 0 or 1 is not a quantile these panels can place."""
        with pytest.raises(ValueError, match="inside"):
            compose_quantile_figure(data[0], data[1], alphas)

    def test_an_ascending_grid_still_composes(self, data):
        """The guard must not cost the valid case."""
        assert compose_quantile_figure(data[0], data[1], (0.1, 0.5, 0.9)) is not None


class TestAgreementMeasuresTheEnsembleNotItsOwnMedian:
    """A default threshold taken from the sample being measured yields exactly 0.5 whatever the data says."""

    @staticmethod
    def _ensemble(member_sd, seed=0):
        """2000 rows, 5 members, disagreeing by member_sd."""
        rng = np.random.default_rng(seed)
        base = rng.normal(0.0, 1.0, 2000)
        return base[:, None] + rng.normal(0.0, member_sd, (2000, 5))

    def test_a_tight_and_a_wide_ensemble_score_differently(self):
        """Pre-fix both scored 0.5000 exactly, because the threshold was the median of the spread itself."""
        tight = compute_prediction_stability(self._ensemble(0.001)).agreement
        wide = compute_prediction_stability(self._ensemble(1.0)).agreement
        assert tight - wide > 0.1

    def test_an_explicit_threshold_is_still_honoured(self):
        """Callers supplying their own domain threshold must be unaffected by the new default."""
        res = compute_prediction_stability(self._ensemble(1.0), low_spread_threshold=10.0)
        assert res.agreement == pytest.approx(1.0)
