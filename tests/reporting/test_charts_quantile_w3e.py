"""W3-E quantile chart fixes: COVERAGE panel (R-5), INTERVAL_BAND markers+band (INV-18),
PIT annotation (INV-19), pinball index lookup + :g titles (INV-36)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts.quantile import (
    ALLOWED_QUANTILE_PANEL_TOKENS, _symmetric_interval_pairs, _wilson_ci,
    compose_quantile_figure,
)
from mlframe.reporting.spec import AnnotationPanelSpec, LinePanelSpec


@pytest.fixture
def calibrated_5alpha():
    """Well-calibrated: y ~ N(0,1) and preds are the exact theoretical normal quantiles."""
    from scipy.stats import norm
    rng = np.random.default_rng(0)
    n = 8000
    y = rng.standard_normal(n)
    alphas = (0.05, 0.25, 0.5, 0.75, 0.95)
    preds = np.tile(norm.ppf(alphas), (n, 1))
    return y, preds, alphas


@pytest.fixture
def overconfident_5alpha():
    """Overconfident: intervals are HALF the correct width -> empirical coverage below nominal."""
    from scipy.stats import norm
    rng = np.random.default_rng(0)
    n = 8000
    y = rng.standard_normal(n)
    alphas = (0.05, 0.25, 0.5, 0.75, 0.95)
    preds = np.tile(norm.ppf(alphas) * 0.5, (n, 1))   # shrink intervals toward the median
    return y, preds, alphas


# ----------------------------------------------------------------------------
# COVERAGE panel (R-5)
# ----------------------------------------------------------------------------


class TestCoveragePanel:
    def test_token_registered(self):
        assert "COVERAGE" in ALLOWED_QUANTILE_PANEL_TOKENS

    def test_returns_line_with_diagonal_and_ci_band(self, calibrated_5alpha):
        y, p, alphas = calibrated_5alpha
        spec = compose_quantile_figure(y, p, alphas, panels_template="COVERAGE")
        panel = spec.panels[0][0]
        assert isinstance(panel, LinePanelSpec)
        assert panel.series_labels[0] == "perfect"
        np.testing.assert_allclose(panel.y[0], panel.x)        # identity diagonal
        assert panel.band is not None                          # Wilson CI band
        assert panel.band_label and "Wilson" in panel.band_label

    def test_symmetric_pairs_two_for_5alpha(self):
        # alphas 0.05/0.25/0.5/0.75/0.95 -> pairs (0.05,0.95)=0.90 and (0.25,0.75)=0.50.
        pairs = _symmetric_interval_pairs((0.05, 0.25, 0.5, 0.75, 0.95))
        assert len(pairs) == 2
        noms = sorted(round(p[2], 4) for p in pairs)
        assert noms == [0.5, 0.9]

    def test_too_few_alphas_returns_annotation(self):
        # Single alpha -> no straddling pair -> honest annotation.
        y = np.zeros(100)
        p = np.zeros((100, 1))
        spec = compose_quantile_figure(y, p, (0.5,), panels_template="COVERAGE")
        assert isinstance(spec.panels[0][0], AnnotationPanelSpec)

    def test_wilson_ci_bounds_in_unit_interval(self):
        lo, hi = _wilson_ci(0.0, 100)
        assert 0.0 <= lo <= hi <= 1.0
        lo, hi = _wilson_ci(1.0, 100)
        assert 0.0 <= lo <= hi <= 1.0

    def test_biz_value_calibrated_near_diagonal(self, calibrated_5alpha):
        """A well-calibrated model's empirical coverage must hug the nominal diagonal."""
        y, p, alphas = calibrated_5alpha
        panel = compose_quantile_figure(y, p, alphas, panels_template="COVERAGE").panels[0][0]
        # |empirical - nominal| small everywhere. Measured max dev ~0.01 at n=8000;
        # floor 0.05 (well above noise, well below the overconfident gap of ~0.2).
        max_dev = float(np.max(np.abs(panel.y[1] - panel.x)))
        assert max_dev < 0.05, f"calibrated coverage deviates {max_dev:.3f} from diagonal"

    def test_biz_value_overconfident_sits_below(self, overconfident_5alpha):
        """Halving interval width -> empirical coverage sits clearly BELOW nominal."""
        y, p, alphas = overconfident_5alpha
        panel = compose_quantile_figure(y, p, alphas, panels_template="COVERAGE").panels[0][0]
        empirical = panel.y[1]
        nominal = panel.x
        # Every level under-covers. Measured shortfall at the 0.90 level ~0.25; the 0.50
        # level ~0.12. Require a mean shortfall >= 0.10 (15% below the ~0.18 measured mean).
        shortfall = float(np.mean(nominal - empirical))
        assert shortfall >= 0.10, f"overconfident model only short by {shortfall:.3f}"
        assert np.all(empirical <= nominal + 1e-9), "overconfident coverage exceeded nominal"


# ----------------------------------------------------------------------------
# INV-18 INTERVAL_BAND markers + band fill
# ----------------------------------------------------------------------------


class TestIntervalBand:
    def test_y_true_markers_and_band(self, calibrated_5alpha):
        y, p, alphas = calibrated_5alpha
        panel = compose_quantile_figure(y, p, alphas, panels_template="INTERVAL_BAND").panels[0][0]
        assert isinstance(panel, LinePanelSpec)
        assert len(panel.y) == 2                              # median + y_true
        assert panel.line_styles[1] == "markers"             # y_true not a connected line
        assert panel.band is not None and len(panel.band) == 2
        lo, hi = panel.band
        assert np.all(hi >= lo)                               # band is lo..hi


# ----------------------------------------------------------------------------
# INV-19 PIT annotation
# ----------------------------------------------------------------------------


class TestPITAnnotation:
    def test_pit_k2_annotation_not_fake_histogram(self, calibrated_5alpha):
        y, p, alphas = calibrated_5alpha
        # Keep only 2 alphas -> PIT must annotate, not draw a [0.0] histogram.
        panel = compose_quantile_figure(
            y, p[:, [0, 4]], (alphas[0], alphas[4]), panels_template="PIT_HIST",
        ).panels[0][0]
        assert isinstance(panel, AnnotationPanelSpec)


# ----------------------------------------------------------------------------
# INV-36 pinball index lookup + :g titles
# ----------------------------------------------------------------------------


class TestPinballAndTitles:
    def test_pinball_lookup_by_index_not_float_key(self):
        # alphas whose float repr is fragile (1/3) must still resolve every loss.
        from scipy.stats import norm
        rng = np.random.default_rng(0)
        n = 1000
        y = rng.standard_normal(n)
        alphas = (1 / 3, 2 / 3)
        preds = np.tile(norm.ppf(alphas), (n, 1))
        panel = compose_quantile_figure(y, preds, alphas, panels_template="PINBALL_BY_ALPHA").panels[0][0]
        losses = panel.y if isinstance(panel.y, np.ndarray) else np.asarray(panel.y)
        assert losses.shape[0] == 2
        assert np.all(np.isfinite(losses)) and np.all(losses >= 0.0)

    def test_width_title_uses_g_format(self, calibrated_5alpha):
        y, p, alphas = calibrated_5alpha
        panel = compose_quantile_figure(y, p, alphas, panels_template="WIDTH_DIST").panels[0][0]
        # No raw float spew like "0.94999999" in the title; :g gives "0.95".
        assert "0.9499" not in panel.title
        assert "q_0.95" in panel.title or "q_0.05" in panel.title
