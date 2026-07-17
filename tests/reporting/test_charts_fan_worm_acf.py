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
    ALLOWED_QUANTILE_PANEL_TOKENS,
    compose_quantile_figure,
)
from mlframe.reporting.charts.regression import (
    ALLOWED_REGRESSION_PANEL_TOKENS,
    compose_regression_figure,
)
from mlframe.reporting.output import parse_plot_output_dsl
from mlframe.reporting.renderers import render_and_save
from mlframe.reporting.spec import (
    AnnotationPanelSpec,
    BarPanelSpec,
    LinePanelSpec,
    ScatterPanelSpec,
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


# ----------------------------------------------------------------------------
# WORM (regression.py)
# ----------------------------------------------------------------------------


@pytest.fixture
def synth_resid_normal():
    """Clean normal residuals: the worm should stay inside the CI band."""
    rng = np.random.default_rng(1)
    n = 3000
    yt = rng.normal(0.0, 1.0, n)
    yp = yt + rng.normal(0.0, 0.5, n)
    return yt, yp


@pytest.fixture
def synth_resid_heavy_tailed():
    """Heavy-tailed (Student-t df=2) residuals: the worm tails must leave the CI band."""
    rng = np.random.default_rng(2)
    n = 3000
    yt = rng.normal(0.0, 1.0, n)
    yp = yt - rng.standard_t(2.0, n)  # residual = yt - yp = t-distributed
    return yt, yp


class TestWorm:
    def test_token_registered(self):
        assert "WORM" in ALLOWED_REGRESSION_PANEL_TOKENS

    def test_worm_shape(self, synth_resid_normal):
        yt, yp = synth_resid_normal
        fig = compose_regression_figure(yt, yp, panels_template="WORM")
        panel = _flat(fig)[0]
        assert isinstance(panel, LinePanelSpec)
        assert panel.band is not None
        lo, hi = panel.band
        # Decimated to <= 2000 plotted points.
        assert panel.x.shape[0] <= 2000
        assert lo.shape[0] == panel.x.shape[0]
        # x (theoretical quantile) is sorted ascending.
        assert np.all(np.diff(panel.x) >= -1e-9)

    def test_worm_skips_on_constant_resid(self):
        yt = np.arange(50.0)
        yp = yt.copy()  # zero residual -> zero variance
        fig = compose_regression_figure(yt, yp, panels_template="WORM")
        panel = _flat(fig)[0]
        assert isinstance(panel, AnnotationPanelSpec)

    def test_biz_val_worm_normal_resid_stays_in_band(self, synth_resid_normal):
        """Clean normal residuals MUST mostly stay inside the CI band (>= 85% of plotted points)."""
        yt, yp = synth_resid_normal
        fig = compose_regression_figure(yt, yp, panels_template="WORM")
        panel = _flat(fig)[0]
        detrended = panel.y[0]
        lo, hi = panel.band
        inside = np.mean((detrended >= lo) & (detrended <= hi))
        assert inside >= 0.85, f"normal residuals should mostly stay in band: inside={inside:.3f}"

    def test_biz_val_worm_heavy_tails_leave_band(self, synth_resid_heavy_tailed):
        """Heavy-tailed residuals MUST leave the CI band at the tails.

        The extreme-tail detrended points (outer 5% of theoretical quantiles) on a Student-t(2) residual
        must escape the pointwise CI band; a normal residual would not. Threshold: at least one tail
        point clears the band by a clear margin (the t(2) tails are far heavier than normal).
        """
        yt, yp = synth_resid_heavy_tailed
        fig = compose_regression_figure(yt, yp, panels_template="WORM")
        panel = _flat(fig)[0]
        detrended = panel.y[0]
        lo, hi = panel.band
        x = panel.x
        # Outer tails: lowest / highest 5% of theoretical quantiles.
        tail_mask = (x <= np.quantile(x, 0.05)) | (x >= np.quantile(x, 0.95))
        out = (detrended[tail_mask] < lo[tail_mask]) | (detrended[tail_mask] > hi[tail_mask])
        frac_out = float(np.mean(out))
        assert frac_out >= 0.30, f"heavy tails must escape the CI band at the tails: frac_out={frac_out:.3f}"

    def test_worm_render_matplotlib(self, synth_resid_heavy_tailed, tmp_path):
        yt, yp = synth_resid_heavy_tailed
        spec = compose_regression_figure(yt, yp, panels_template="WORM")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_and_save(spec, parse_plot_output_dsl("matplotlib[png]"), str(tmp_path / "worm"))
        assert os.path.exists(tmp_path / "worm.png")


# ----------------------------------------------------------------------------
# RESID_ACF (regression.py)
# ----------------------------------------------------------------------------


@pytest.fixture
def synth_resid_white():
    """White-noise residuals: no lag should clear the Bartlett band (beyond chance ~5%)."""
    rng = np.random.default_rng(3)
    n = 5000
    yt = rng.normal(0.0, 1.0, n)
    yp = yt + rng.normal(0.0, 0.5, n)
    return yt, yp


@pytest.fixture
def synth_resid_ar1():
    """AR(1) residuals (phi=0.6): the lag-1 ACF must clear the Bartlett band."""
    rng = np.random.default_rng(4)
    n = 5000
    e = rng.normal(0.0, 1.0, n)
    resid = np.zeros(n)
    for i in range(1, n):
        resid[i] = 0.6 * resid[i - 1] + e[i]
    yt = rng.normal(0.0, 1.0, n)
    yp = yt - resid  # residual = yt - yp = AR(1) series
    return yt, yp


class TestResidAcf:
    def test_token_registered(self):
        assert "RESID_ACF" in ALLOWED_REGRESSION_PANEL_TOKENS

    def test_resid_acf_shape(self, synth_resid_white):
        yt, yp = synth_resid_white
        fig = compose_regression_figure(yt, yp, panels_template="RESID_ACF")
        panel = _flat(fig)[0]
        assert isinstance(panel, BarPanelSpec)
        assert panel.hline is not None
        # Lags capped at MAX_ACF_LAGS.
        assert len(panel.categories) <= 50
        assert panel.values.shape[0] == len(panel.categories)

    def test_biz_val_resid_acf_ar1_lag1_above_band(self, synth_resid_ar1):
        """AR(1) residuals MUST show a lag-1 ACF bar clearly above the Bartlett band.

        Designed phi=0.6, so the lag-1 ACF should sit near 0.6 and far above the ~0.028 band at n=5000.
        Floor 0.45 (well below the ~0.6 measured, accounting for the yt-yp mixing) and require it clears
        the band by >= 5x.
        """
        yt, yp = synth_resid_ar1
        fig = compose_regression_figure(yt, yp, panels_template="RESID_ACF")
        panel = _flat(fig)[0]
        lag1 = float(panel.values[0])
        band = panel.hline[0]
        assert lag1 >= 0.45, f"AR(1) lag-1 ACF should be >= 0.45, got {lag1:.3f}"
        assert lag1 >= 5.0 * band, f"lag-1 ACF {lag1:.3f} should clear band {band:.4f} by >=5x"

    def test_biz_val_resid_acf_white_noise_stays_in_band(self, synth_resid_white):
        """White-noise residuals: lag-1 ACF should sit inside the Bartlett band."""
        yt, yp = synth_resid_white
        fig = compose_regression_figure(yt, yp, panels_template="RESID_ACF")
        panel = _flat(fig)[0]
        lag1 = abs(float(panel.values[0]))
        band = panel.hline[0]
        assert lag1 <= band, f"white-noise lag-1 ACF {lag1:.4f} should be within band {band:.4f}"

    def test_resid_acf_render_matplotlib(self, synth_resid_ar1, tmp_path):
        yt, yp = synth_resid_ar1
        spec = compose_regression_figure(yt, yp, panels_template="RESID_ACF")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_and_save(spec, parse_plot_output_dsl("matplotlib[png]"), str(tmp_path / "acf"))
        assert os.path.exists(tmp_path / "acf.png")


# ----------------------------------------------------------------------------
# _acf helper direct coverage
# ----------------------------------------------------------------------------


class TestAcfHelper:
    def test_acf_ar1_recovers_phi(self):
        rng = np.random.default_rng(5)
        n = 20000
        e = rng.standard_normal(n)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.7 * x[i - 1] + e[i]
        acf_lags, n_used = acf_fft(x, nlags=10)
        assert n_used == n
        assert abs(acf_lags[0] - 0.7) < 0.05

    def test_pacf_ar1_spike_only_at_lag1(self):
        rng = np.random.default_rng(6)
        n = 20000
        e = rng.standard_normal(n)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.7 * x[i - 1] + e[i]
        pacf_lags, _ = pacf_levinson(x, nlags=10)
        # AR(1): PACF[lag1] ~ 0.7, all higher lags ~ 0.
        assert abs(pacf_lags[0] - 0.7) < 0.05
        assert np.all(np.abs(pacf_lags[1:]) < 0.05)

    def test_acf_tail_caps_long_series(self):
        from mlframe.reporting.charts._acf import MAX_ACF_SERIES

        rng = np.random.default_rng(7)
        x = rng.standard_normal(MAX_ACF_SERIES + 50_000)
        _, n_used = acf_fft(x, nlags=5)
        assert n_used == MAX_ACF_SERIES
