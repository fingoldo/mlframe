"""Tests for ``mlframe.reporting.charts.quantile``.

Covers all 5 panel tokens (RELIABILITY / PINBALL_BY_ALPHA /
INTERVAL_BAND / WIDTH_DIST / PIT_HIST) + composer + token routing +
matplotlib + plotly render smoke.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from mlframe.reporting.charts.quantile import (
    ALLOWED_QUANTILE_PANEL_TOKENS, compose_quantile_figure,
)
from mlframe.reporting.output import parse_plot_output_dsl
from mlframe.reporting.renderers import render_and_save
from mlframe.reporting.spec import (
    FigureSpec, HistogramPanelSpec, LinePanelSpec,
)


@pytest.fixture
def synth_qr_3alpha():
    """500 rows, std-normal y, theoretical quantile predictions at 10/50/90."""
    rng = np.random.default_rng(0)
    n = 500
    y = rng.standard_normal(n)
    preds = np.column_stack([
        np.full(n, -1.2815515655446004),
        np.zeros(n),
        np.full(n, 1.2815515655446004),
    ])
    alphas = (0.1, 0.5, 0.9)
    return y, preds, alphas


@pytest.fixture
def synth_qr_5alpha():
    """500 rows, 5 alphas (0.05/0.25/0.5/0.75/0.95) -- big enough for PIT."""
    rng = np.random.default_rng(0)
    n = 500
    y = rng.standard_normal(n)
    from scipy.stats import norm
    alphas = (0.05, 0.25, 0.5, 0.75, 0.95)
    preds = np.tile(norm.ppf(alphas), (n, 1))
    return y, preds, alphas


# ----------------------------------------------------------------------------
# Allowed tokens
# ----------------------------------------------------------------------------


class TestAllowedTokens:
    def test_allowed_set(self):
        assert ALLOWED_QUANTILE_PANEL_TOKENS == frozenset({
            "RELIABILITY", "COVERAGE", "PINBALL_BY_ALPHA", "INTERVAL_BAND",
            "WIDTH_DIST", "PIT_HIST",
        })


# ----------------------------------------------------------------------------
# Per-token spec shape
# ----------------------------------------------------------------------------


class TestPanelTypes:
    def test_reliability_returns_line_with_diagonal(self, synth_qr_3alpha):
        y, p, alphas = synth_qr_3alpha
        spec = compose_quantile_figure(y, p, alphas, panels_template="RELIABILITY")
        panel = spec.panels[0][0]
        assert isinstance(panel, LinePanelSpec)
        assert isinstance(panel.y, tuple) and len(panel.y) == 2
        assert panel.series_labels[0] == "perfect"
        assert panel.series_labels[1] == "empirical"

    def test_pinball_by_alpha_returns_line(self, synth_qr_3alpha):
        y, p, alphas = synth_qr_3alpha
        spec = compose_quantile_figure(y, p, alphas, panels_template="PINBALL_BY_ALPHA")
        panel = spec.panels[0][0]
        assert isinstance(panel, LinePanelSpec)
        # x = alphas; y = pinball losses
        assert len(panel.x) == 3
        assert isinstance(panel.y, np.ndarray) or len(panel.y) == 3

    def test_interval_band_returns_line(self, synth_qr_3alpha):
        y, p, alphas = synth_qr_3alpha
        spec = compose_quantile_figure(y, p, alphas, panels_template="INTERVAL_BAND")
        panel = spec.panels[0][0]
        assert isinstance(panel, LinePanelSpec)
        # median line + y_true markers, with the lo..hi interval drawn as a filled band.
        assert len(panel.y) == 2
        assert "y_true" in panel.series_labels
        assert panel.band is not None and len(panel.band) == 2
        # y_true is rendered as markers, not a connected line.
        assert panel.line_styles[1] == "markers"

    def test_width_dist_returns_histogram(self, synth_qr_3alpha):
        y, p, alphas = synth_qr_3alpha
        spec = compose_quantile_figure(y, p, alphas, panels_template="WIDTH_DIST")
        panel = spec.panels[0][0]
        assert isinstance(panel, HistogramPanelSpec)
        # Constant predictions -> width = 2.563 everywhere (degenerate
        # histogram; not a problem for the spec, just for visual reading).
        assert panel.values.shape == (len(y),)

    def test_pit_hist_returns_histogram(self, synth_qr_5alpha):
        y, p, alphas = synth_qr_5alpha
        spec = compose_quantile_figure(y, p, alphas, panels_template="PIT_HIST")
        panel = spec.panels[0][0]
        assert isinstance(panel, HistogramPanelSpec)
        assert panel.values.shape == (len(y),)
        assert "PIT" in panel.title

    def test_pit_hist_placeholder_when_K_lt_3(self, synth_qr_3alpha):
        # K = 2 alphas (drop median) -> honest annotation placeholder, not a fake histogram.
        from mlframe.reporting.spec import AnnotationPanelSpec
        y, p, alphas = synth_qr_3alpha
        spec = compose_quantile_figure(y, p[:, [0, 2]], (alphas[0], alphas[2]),
                                        panels_template="PIT_HIST")
        panel = spec.panels[0][0]
        assert isinstance(panel, AnnotationPanelSpec)
        assert "requires k >= 3" in panel.text.lower()


# ----------------------------------------------------------------------------
# Composer
# ----------------------------------------------------------------------------


class TestComposer:
    def test_default_template_returns_5_panels(self, synth_qr_3alpha):
        y, p, alphas = synth_qr_3alpha
        spec = compose_quantile_figure(y, p, alphas)
        # 5 tokens -> 3 rows × 2 cols (one cell padded with None)
        n_set = sum(1 for r in spec.panels for c in r if c is not None)
        assert n_set == 5

    def test_subset_template(self, synth_qr_3alpha):
        y, p, alphas = synth_qr_3alpha
        spec = compose_quantile_figure(y, p, alphas, panels_template="RELIABILITY WIDTH_DIST")
        n_set = sum(1 for r in spec.panels for c in r if c is not None)
        assert n_set == 2

    def test_unknown_token_raises(self, synth_qr_3alpha):
        y, p, alphas = synth_qr_3alpha
        with pytest.raises(ValueError, match="Unknown quantile"):
            compose_quantile_figure(y, p, alphas, panels_template="RELIABILITY BOGUS")

    def test_shape_mismatch_raises_2d(self, synth_qr_3alpha):
        y, p, alphas = synth_qr_3alpha
        # Wrong K
        with pytest.raises(ValueError, match="preds_NK.shape\\[1\\]"):
            compose_quantile_figure(y, p[:, :2], alphas)

    def test_n_mismatch_raises(self, synth_qr_3alpha):
        y, p, alphas = synth_qr_3alpha
        with pytest.raises(ValueError, match="preds_NK.shape\\[0\\]"):
            compose_quantile_figure(y[:-1], p, alphas)

    def test_1d_preds_raises(self, synth_qr_3alpha):
        y, _, alphas = synth_qr_3alpha
        with pytest.raises(ValueError, match="2-D preds_NK"):
            compose_quantile_figure(y, np.zeros(len(y)), alphas)

    def test_suptitle(self, synth_qr_3alpha):
        y, p, alphas = synth_qr_3alpha
        spec = compose_quantile_figure(y, p, alphas, panels_template="RELIABILITY",
                                        suptitle="QR baseline")
        assert spec.suptitle == "QR baseline"


# ----------------------------------------------------------------------------
# Render smoke
# ----------------------------------------------------------------------------


class TestRender:
    def test_matplotlib_render(self, synth_qr_3alpha, tmp_path):
        y, p, alphas = synth_qr_3alpha
        spec = compose_quantile_figure(y, p, alphas)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_and_save(spec, parse_plot_output_dsl("matplotlib[png]"),
                            str(tmp_path / "qr"))
        assert os.path.exists(tmp_path / "qr.png")
        assert os.path.getsize(tmp_path / "qr.png") > 5000

    def test_plotly_render(self, synth_qr_3alpha, tmp_path):
        y, p, alphas = synth_qr_3alpha
        spec = compose_quantile_figure(y, p, alphas)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_and_save(spec, parse_plot_output_dsl("plotly[html]"),
                            str(tmp_path / "qr"))
        assert os.path.exists(tmp_path / "qr.html")
