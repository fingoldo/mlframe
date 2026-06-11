"""Tests for the A2 diagnostic chart builders.

Covers four new tokens across three modules:

* ``quantile.FAN_CHART`` -- nested predictive-interval bands over the forecast horizon. biz_value:
  on a widening-uncertainty synthetic the band width MUST grow with the horizon.
* ``regression.WORM`` -- de-trended QQ (QQ residual minus the y=x line) with a CI band. biz_value:
  heavy-tailed residuals MUST leave the CI band at the tails.
* ``regression.RESID_ACF`` -- residual autocorrelation by lag with Bartlett bounds. biz_value: AR(1)
  residuals MUST show a lag-1 ACF bar above the significance bound.
* (temporal target ACF/PACF lives in ``test_charts_temporal.py`` additions.)
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from mlframe.reporting.charts._acf import acf_fft, pacf_levinson, significance_band
from mlframe.reporting.charts.quantile import (
    ALLOWED_QUANTILE_PANEL_TOKENS, compose_quantile_figure,
)
from mlframe.reporting.charts.regression import (
    ALLOWED_REGRESSION_PANEL_TOKENS, compose_regression_figure,
)
from mlframe.reporting.output import parse_plot_output_dsl
from mlframe.reporting.renderers import render_and_save
from mlframe.reporting.spec import (
    AnnotationPanelSpec, BarPanelSpec, LinePanelSpec, ScatterPanelSpec,
)


def _flat(fig):
    return [p for row in fig.panels for p in row if p is not None]


# ----------------------------------------------------------------------------
# FAN_CHART (quantile.py)
# ----------------------------------------------------------------------------


@pytest.fixture
def synth_fan_widening():
    """A widening-uncertainty forecast: interval half-width grows linearly with the horizon index."""
    rng = np.random.default_rng(0)
    n = 4000
    idx = np.arange(n)
    horizon = idx / n
    center = np.sin(horizon * 6.0)
    half = 0.1 + 2.0 * horizon  # half-width grows 20x from start to end
    alphas = (0.05, 0.25, 0.5, 0.75, 0.95)
    z = {0.05: -1.6449, 0.25: -0.6745, 0.5: 0.0, 0.75: 0.6745, 0.95: 1.6449}
    preds = np.column_stack([center + z[a] * half for a in alphas])
    y = center + rng.standard_normal(n) * half
    return y, preds, alphas


class TestFanChart:
    def test_token_registered(self):
        assert "FAN_CHART" in ALLOWED_QUANTILE_PANEL_TOKENS

    def test_fan_chart_shape(self, synth_fan_widening):
        y, p, alphas = synth_fan_widening
        fig = compose_quantile_figure(y, p, alphas, panels_template="FAN_CHART")
        panels = _flat(fig)
        assert len(panels) == 1
        panel = panels[0]
        assert isinstance(panel, LinePanelSpec)
        assert panel.band is not None
        lo, hi = panel.band
        # Downsampled to <= 400 horizon buckets, never raw n.
        assert lo.shape[0] <= 400
        assert hi.shape[0] == lo.shape[0]
        # Outer band is the widest interval -> hi >= lo everywhere.
        assert np.all(hi >= lo - 1e-9)

    def test_fan_chart_skips_on_single_alpha(self):
        y = np.zeros(10)
        p = np.zeros((10, 1))
        fig = compose_quantile_figure(y, p, (0.5,), panels_template="FAN_CHART")
        panel = _flat(fig)[0]
        assert isinstance(panel, AnnotationPanelSpec)
        assert "needs >= 2" in panel.text

    def test_biz_val_fan_chart_band_widens_with_horizon(self, synth_fan_widening):
        """Band width MUST grow with the horizon on a widening-uncertainty forecast.

        The synthetic half-width grows 20x from start to end; the late-horizon outer band must be at
        least 3x the early-horizon band (floor well below the ~12x designed-in widening to absorb the
        per-bucket averaging + quantile noise).
        """
        y, p, alphas = synth_fan_widening
        fig = compose_quantile_figure(y, p, alphas, panels_template="FAN_CHART")
        panel = _flat(fig)[0]
        lo, hi = panel.band
        width = hi - lo
        q = width.shape[0] // 5
        early = float(np.mean(width[:q]))
        late = float(np.mean(width[-q:]))
        assert late >= 3.0 * early, f"fan band must widen with horizon: early={early:.3f} late={late:.3f}"

    def test_fan_chart_render_matplotlib(self, synth_fan_widening, tmp_path):
        y, p, alphas = synth_fan_widening
        spec = compose_quantile_figure(y, p, alphas, panels_template="FAN_CHART")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_and_save(spec, parse_plot_output_dsl("matplotlib[png]"), str(tmp_path / "fan"))
        assert os.path.exists(tmp_path / "fan.png")
        assert os.path.getsize(tmp_path / "fan.png") > 5000

    def test_fan_chart_render_plotly(self, synth_fan_widening, tmp_path):
        y, p, alphas = synth_fan_widening
        spec = compose_quantile_figure(y, p, alphas, panels_template="FAN_CHART")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_and_save(spec, parse_plot_output_dsl("plotly[html]"), str(tmp_path / "fan"))
        assert os.path.exists(tmp_path / "fan.html")
