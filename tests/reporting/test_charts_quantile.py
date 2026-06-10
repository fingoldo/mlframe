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
    _model_diagnostics_decompose,
)
from mlframe.reporting.output import parse_plot_output_dsl
from mlframe.reporting.renderers import render_and_save
from mlframe.reporting.spec import (
    AnnotationPanelSpec, BarPanelSpec, FigureSpec, HistogramPanelSpec, LinePanelSpec,
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
            "QUANTILE_RELIABILITY", "PINBALL_DECOMP", "QUANTILE_CROSSING",
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
    def test_default_template_renders_all_default_tokens(self, synth_qr_3alpha):
        from mlframe.reporting.charts.quantile import DEFAULT_QUANTILE_PANELS

        y, p, alphas = synth_qr_3alpha
        spec = compose_quantile_figure(y, p, alphas)
        n_set = sum(1 for r in spec.panels for c in r if c is not None)
        # The R-6 reliability tokens (QUANTILE_RELIABILITY / PINBALL_DECOMP / QUANTILE_CROSSING) are default-on.
        assert n_set == len(DEFAULT_QUANTILE_PANELS.split())

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


# ----------------------------------------------------------------------------
# Reliability extension (R-6): QUANTILE_RELIABILITY / PINBALL_DECOMP / QUANTILE_CROSSING
# ----------------------------------------------------------------------------


@pytest.fixture
def synth_qr_calibrated():
    """Heteroscedastic-free linear DGP with well-calibrated, varying, monotone quantile preds.

    y = x + N(0, 0.5). The true tau-quantile of y|x is ``x + 0.5 * Phi^{-1}(tau)``,
    so feeding exactly that as q_tau yields a calibrated, never-crossing predictor.
    """
    from scipy.stats import norm
    rng = np.random.default_rng(0)
    n = 6000
    x = rng.standard_normal(n)
    y = x + rng.standard_normal(n) * 0.5
    alphas = (0.1, 0.5, 0.9)
    preds = np.column_stack([x + 0.5 * norm.ppf(a) for a in alphas])
    return y, preds, alphas


@pytest.fixture
def synth_qr_miscalibrated():
    """Same DGP, but every quantile shifted DOWN by 1.0 -> systematically over-predicting q.

    With q_tau pushed below the true conditional quantile, FEWER y fall below q_tau, so the
    recalibrated observed coverage sits well BELOW nominal tau -- a large, detectable deviation.
    """
    from scipy.stats import norm
    rng = np.random.default_rng(0)
    n = 6000
    x = rng.standard_normal(n)
    y = x + rng.standard_normal(n) * 0.5
    alphas = (0.1, 0.5, 0.9)
    preds = np.column_stack([x + 0.5 * norm.ppf(a) - 1.0 for a in alphas])
    return y, preds, alphas


@pytest.fixture
def synth_qr_crossing():
    """Independent-head quantile preds that cross: the tau=0.9 column is forced BELOW tau=0.1.

    Produces a clearly non-zero adjacent-pair crossing rate.
    """
    rng = np.random.default_rng(1)
    n = 4000
    x = rng.standard_normal(n)
    y = x + rng.standard_normal(n) * 0.5
    alphas = (0.1, 0.5, 0.9)
    q10 = x + 1.0
    q50 = x
    q90 = x - 1.0  # deliberately inverted vs q10
    preds = np.column_stack([q10, q50, q90])
    return y, preds, alphas


class TestQuantileReliabilityPanelTypes:
    def test_quantile_reliability_returns_line(self, synth_qr_calibrated):
        y, p, alphas = synth_qr_calibrated
        spec = compose_quantile_figure(y, p, alphas, panels_template="QUANTILE_RELIABILITY")
        panel = spec.panels[0][0]
        assert isinstance(panel, LinePanelSpec)
        # K observed curves + K nominal lines.
        assert len(panel.y) == 2 * len(alphas)
        assert any("nominal" in s for s in panel.series_labels)
        assert any("obs" in s for s in panel.series_labels)

    def test_pinball_decomp_returns_bar(self, synth_qr_calibrated):
        y, p, alphas = synth_qr_calibrated
        spec = compose_quantile_figure(y, p, alphas, panels_template="PINBALL_DECOMP")
        panel = spec.panels[0][0]
        assert isinstance(panel, BarPanelSpec)
        assert len(panel.categories) == len(alphas)

    def test_pinball_decomp_corp_three_series_when_md_available(self, synth_qr_calibrated):
        if _model_diagnostics_decompose() is None:
            pytest.skip("model-diagnostics not importable")
        y, p, alphas = synth_qr_calibrated
        spec = compose_quantile_figure(y, p, alphas, panels_template="PINBALL_DECOMP")
        panel = spec.panels[0][0]
        assert isinstance(panel.values, tuple) and len(panel.values) == 3
        assert panel.series_labels == ("miscalibration", "discrimination", "uncertainty")

    def test_quantile_crossing_returns_bar(self, synth_qr_calibrated):
        y, p, alphas = synth_qr_calibrated
        spec = compose_quantile_figure(y, p, alphas, panels_template="QUANTILE_CROSSING")
        panel = spec.panels[0][0]
        assert isinstance(panel, BarPanelSpec)
        # K-1 adjacent pairs.
        assert len(panel.categories) == len(alphas) - 1
        assert panel.values.shape == (len(alphas) - 1,)

    def test_quantile_crossing_placeholder_when_single_alpha(self, synth_qr_calibrated):
        y, p, alphas = synth_qr_calibrated
        spec = compose_quantile_figure(
            y, p[:, [1]], (alphas[1],), panels_template="QUANTILE_CROSSING",
        )
        panel = spec.panels[0][0]
        assert isinstance(panel, AnnotationPanelSpec)
        assert "needs >= 2" in panel.text

    def test_new_tokens_render_matplotlib(self, synth_qr_calibrated, tmp_path):
        y, p, alphas = synth_qr_calibrated
        spec = compose_quantile_figure(
            y, p, alphas,
            panels_template="QUANTILE_RELIABILITY PINBALL_DECOMP QUANTILE_CROSSING",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_and_save(spec, parse_plot_output_dsl("matplotlib[png]"),
                            str(tmp_path / "qr6"))
        assert os.path.exists(tmp_path / "qr6.png")
        assert os.path.getsize(tmp_path / "qr6.png") > 5000


class TestQuantileReliabilityBizValue:
    """Quantitative wins: calibrated -> near-nominal + zero crossing; broken -> detectable."""

    def _reliability_curves(self, y, p, alphas):
        spec = compose_quantile_figure(y, p, alphas, panels_template="QUANTILE_RELIABILITY")
        panel = spec.panels[0][0]
        K = len(alphas)
        obs = panel.y[:K]
        nominal = panel.y[K:]
        return obs, nominal

    def test_biz_calibrated_reliability_tracks_nominal(self, synth_qr_calibrated):
        y, p, alphas = synth_qr_calibrated
        obs, nominal = self._reliability_curves(y, p, alphas)
        for k, a in enumerate(alphas):
            mean_abs_dev = float(np.mean(np.abs(obs[k] - a)))
            # Calibrated: recalibrated coverage stays within 0.05 of nominal tau across the range.
            assert mean_abs_dev < 0.05, f"tau={a} dev {mean_abs_dev}"

    def test_biz_miscalibrated_reliability_off_nominal(self, synth_qr_miscalibrated):
        y, p, alphas = synth_qr_miscalibrated
        obs, nominal = self._reliability_curves(y, p, alphas)
        # q shifted below the true quantile => observed coverage well BELOW nominal at tau=0.5.
        mid = alphas.index(0.5)
        dev = float(np.mean(obs[mid] - 0.5))
        assert dev < -0.2, f"miscalibrated mid-tau drop only {dev}"

    def test_biz_calibrated_crossing_near_zero(self, synth_qr_calibrated):
        y, p, alphas = synth_qr_calibrated
        spec = compose_quantile_figure(y, p, alphas, panels_template="QUANTILE_CROSSING")
        rates = spec.panels[0][0].values
        assert float(rates.max()) < 0.01, f"calibrated crossing rate {rates.max()}"

    def test_biz_crossing_rate_clearly_positive(self, synth_qr_crossing):
        y, p, alphas = synth_qr_crossing
        spec = compose_quantile_figure(y, p, alphas, panels_template="QUANTILE_CROSSING")
        rates = spec.panels[0][0].values
        # Inverted heads => every row violates both adjacent pairs.
        assert float(rates.max()) > 0.9, f"crossing rate too low {rates.max()}"

    def test_biz_corp_miscalibration_jumps_on_shift(
        self, synth_qr_calibrated, synth_qr_miscalibrated,
    ):
        if _model_diagnostics_decompose() is None:
            pytest.skip("model-diagnostics not importable")
        yc, pc, alphas = synth_qr_calibrated
        ym, pm, _ = synth_qr_miscalibrated
        mid = alphas.index(0.5)
        spec_c = compose_quantile_figure(yc, pc, alphas, panels_template="PINBALL_DECOMP")
        spec_m = compose_quantile_figure(ym, pm, alphas, panels_template="PINBALL_DECOMP")
        miscal_c = float(spec_c.panels[0][0].values[0][mid])
        miscal_m = float(spec_m.panels[0][0].values[0][mid])
        # CORP miscalibration component must rise sharply for the shifted predictor.
        assert miscal_m > miscal_c + 0.1, f"calib {miscal_c} vs miscal {miscal_m}"


class TestQuantileReliabilityPerf:
    """The isotonic / CORP fits are subsample-capped so the panels stay fast at large n."""

    def test_reliability_subsample_cap_keeps_curve_size(self):
        # Above the 100k cap the panel must still emit the same fixed-resolution curve, fast.
        from mlframe.reporting.charts.quantile import _RELIABILITY_GRID
        from scipy.stats import norm
        rng = np.random.default_rng(0)
        n = 200_000
        x = rng.standard_normal(n)
        y = x + rng.standard_normal(n) * 0.5
        alphas = (0.1, 0.5, 0.9)
        p = np.column_stack([x + 0.5 * norm.ppf(a) for a in alphas])
        spec = compose_quantile_figure(y, p, alphas, panels_template="QUANTILE_RELIABILITY")
        panel = spec.panels[0][0]
        assert panel.x.shape == (_RELIABILITY_GRID,)
        # Calibrated even after subsampling: observed curve stays near nominal.
        K = len(alphas)
        for k, a in enumerate(alphas):
            assert float(np.mean(np.abs(panel.y[k] - a))) < 0.05

    def test_corp_decomp_capped_fast_at_large_n(self):
        # Regression sensor for the uncapped-CORP perf bug (243s -> sub-second at n=1e6 via cap).
        if _model_diagnostics_decompose() is None:
            pytest.skip("model-diagnostics not importable")
        import time
        from scipy.stats import norm
        rng = np.random.default_rng(0)
        n = 1_000_000
        x = rng.standard_normal(n)
        y = x + rng.standard_normal(n) * 0.5
        alphas = (0.1, 0.5, 0.9)
        p = np.column_stack([x + 0.5 * norm.ppf(a) for a in alphas])
        t0 = time.perf_counter()
        spec = compose_quantile_figure(y, p, alphas, panels_template="PINBALL_DECOMP")
        elapsed = time.perf_counter() - t0
        assert isinstance(spec.panels[0][0].values, tuple)
        # Uncapped this was ~243 s; the 100k cap brings it well under 5 s with wide margin.
        assert elapsed < 5.0, f"CORP decomp too slow at n=1e6: {elapsed:.1f}s (cap regressed?)"
