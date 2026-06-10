"""Tests for ``mlframe.reporting.charts.regression``.

Covers the panel-token composer (SCATTER / RESID_HIST / RESID_VS_PRED /
ERR_BY_DECILE), token routing, the legacy adapter, the density-scatter
threshold, extremes-preserving sampling, the "showing X / N" title note,
and matplotlib + plotly render smoke. Includes biz_value tests for the
two diagnostics that must produce a known verdict:

* ERR_BY_DECILE bias -- a model that under-predicts the top decile MUST
  show a clearly positive mean-signed-residual bar there.
* RESID_VS_PRED funnel -- a heteroscedastic synth (variance grows with
  y_hat) MUST widen the IQR band toward high predictions even though the
  running median stays ~flat (Spearman of signed resid ~ 0).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from mlframe.reporting.charts.regression import (
    ALLOWED_REGRESSION_PANEL_TOKENS, DEFAULT_HEXBIN_THRESHOLD,
    DEFAULT_REGRESSION_PANELS, build_regression_panel_spec,
    compose_regression_figure,
)
from mlframe.reporting.output import parse_plot_output_dsl
from mlframe.reporting.renderers import render_and_save
from mlframe.reporting.spec import (
    BarPanelSpec, FigureSpec, HeatmapPanelSpec, HistogramPanelSpec,
    LinePanelSpec, ScatterPanelSpec,
)
from mlframe.training.targets import audit_residuals


def _flat(fig):
    return [p for row in fig.panels for p in row if p is not None]


@pytest.fixture
def synth_reg():
    rng = np.random.default_rng(0)
    n = 2000
    yt = rng.normal(0.0, 1.0, n)
    yp = yt + rng.normal(0.0, 0.3, n)
    return yt, yp


def test_allowed_tokens_are_the_four_panels():
    assert ALLOWED_REGRESSION_PANEL_TOKENS == frozenset(
        {"SCATTER", "RESID_HIST", "RESID_VS_PRED", "ERR_BY_DECILE"}
    )
    assert DEFAULT_REGRESSION_PANELS == "SCATTER RESID_HIST RESID_VS_PRED ERR_BY_DECILE"


def test_default_template_packs_four_panels(synth_reg):
    yt, yp = synth_reg
    fig = compose_regression_figure(yt, yp, audit=audit_residuals(yt, yp))
    panels = _flat(fig)
    types = {type(p).__name__ for p in panels}
    assert {"ScatterPanelSpec", "HistogramPanelSpec", "LinePanelSpec", "BarPanelSpec"} == types
    assert len(panels) == 4


def test_unknown_token_raises(synth_reg):
    yt, yp = synth_reg
    with pytest.raises(ValueError, match="Unknown regression panel tokens"):
        compose_regression_figure(yt, yp, panels_template="SCATTER BOGUS")


def test_scatter_has_perfect_fit_diagonal(synth_reg):
    yt, yp = synth_reg
    fig = compose_regression_figure(yt, yp, panels_template="SCATTER")
    sc = _flat(fig)[0]
    assert isinstance(sc, ScatterPanelSpec)
    assert sc.perfect_fit_line is True


def test_scatter_subsample_keeps_max_error_and_shows_count():
    rng = np.random.default_rng(1)
    n = 12_000
    yt = rng.normal(0.0, 1.0, n)
    yp = yt + rng.normal(0.0, 0.3, n)
    yt[0] = 50.0  # MaxError point the title quotes -- must survive the subsample.
    fig = compose_regression_figure(yt, yp, panels_template="SCATTER", metrics_str="MAE=0.2")
    sc = _flat(fig)[0]
    assert isinstance(sc, ScatterPanelSpec)
    assert "showing 5,000 / 12,000 sampled" in sc.title
    assert float(np.max(sc.y)) > 40.0  # the extreme is plotted, not dropped


def test_scatter_density_heatmap_above_threshold():
    rng = np.random.default_rng(2)
    n = DEFAULT_HEXBIN_THRESHOLD + 5_000
    yt = rng.normal(0.0, 1.0, n)
    yp = yt + rng.normal(0.0, 0.3, n)
    fig = compose_regression_figure(yt, yp, panels_template="SCATTER")
    panel = _flat(fig)[0]
    assert isinstance(panel, HeatmapPanelSpec)
    assert panel.colorbar_label == "log(1 + count)"
    assert np.all(np.isfinite(panel.matrix))


def test_resid_vs_pred_has_iqr_band(synth_reg):
    yt, yp = synth_reg
    fig = compose_regression_figure(yt, yp, audit=audit_residuals(yt, yp),
                                    panels_template="RESID_VS_PRED")
    line = _flat(fig)[0]
    assert isinstance(line, LinePanelSpec)
    assert line.band is not None
    lower, upper = line.band
    assert np.all(upper >= lower)  # q75 >= q25 by construction


def test_legacy_adapter_two_panel_keeps_figsize(synth_reg):
    yt, yp = synth_reg
    fig = build_regression_panel_spec(
        yt, yp, audit=audit_residuals(yt, yp), header_str="h", metrics_str="m",
        panels_template="SCATTER RESID_HIST", figsize=(16.0, 5.0),
    )
    assert isinstance(fig, FigureSpec)
    assert fig.figsize == (16.0, 5.0)
    assert {type(p).__name__ for p in _flat(fig)} == {"ScatterPanelSpec", "HistogramPanelSpec"}


def test_audit_none_does_not_crash(synth_reg):
    yt, yp = synth_reg
    fig = compose_regression_figure(yt, yp, audit=None)
    assert len(_flat(fig)) == 4


@pytest.mark.parametrize("backend", ["matplotlib[png]", "plotly[html]"])
def test_render_smoke(synth_reg, tmp_path, backend):
    yt, yp = synth_reg
    fig = compose_regression_figure(yt, yp, audit=audit_residuals(yt, yp), suptitle="S")
    base = os.path.join(str(tmp_path), "reg")
    render_and_save(fig, parse_plot_output_dsl(backend), base)
    assert any(os.scandir(str(tmp_path)))


# ---------------------------------------------------------------------------
# biz_value: the diagnostics must produce a KNOWN verdict on a designed synth.
# ---------------------------------------------------------------------------


def test_biz_val_err_by_decile_top_decile_underprediction_bias():
    """A model that under-predicts the top target decile MUST show a clearly positive mean-signed-residual bar there.

    Synthetic: y_pred = y_true except the top decile is compressed toward the mean (GBM extreme-compression). Then
    resid = y_true - y_pred is large positive in the top decile. Measured top-decile mean signed resid ~ +3.5 on this
    synth; floor at +1.0 (well below measured, well above the ~0 of the other deciles).
    """
    rng = np.random.default_rng(3)
    n = 5000
    yt = rng.uniform(0.0, 10.0, n)
    yp = yt + rng.normal(0.0, 0.1, n)  # near-perfect everywhere ...
    top = yt >= np.quantile(yt, 0.9)
    yp[top] = yt[top] - (yt[top] - 5.0) * 0.7  # ... except compress the top decile toward the mean (5.0)

    fig = compose_regression_figure(yt, yp, panels_template="ERR_BY_DECILE")
    bar = _flat(fig)[0]
    assert isinstance(bar, BarPanelSpec)
    mean_abs, mean_signed = bar.values
    assert mean_signed.shape[0] == 10
    # Top decile (D10, index 9) under-predicted -> signed resid clearly positive.
    assert mean_signed[9] >= 1.0, f"top-decile signed resid {mean_signed[9]:.3f} should be >= 1.0 (under-prediction)"
    # The lower deciles are near-perfect -> their signed resid is ~0.
    assert abs(mean_signed[0]) < 0.3
    # Magnitude bar must also flag the top decile as the worst.
    assert mean_abs[9] == pytest.approx(np.max(mean_abs), rel=1e-6)


def test_biz_val_resid_vs_pred_funnel_widens_band_under_heteroscedasticity():
    """A heteroscedastic synth (residual std grows with y_hat) MUST widen the IQR band toward high predictions.

    The running median stays ~flat (no bias), so this is detected by band WIDTH, not by the median. Measured: top-bin
    IQR width / bottom-bin IQR width ~ 8x; floor at 3x. Audit flags heteroscedastic.
    """
    rng = np.random.default_rng(4)
    n = 8000
    yp = rng.uniform(0.0, 10.0, n)
    noise = rng.normal(0.0, 1.0, n) * (0.2 + yp)  # sigma grows with prediction
    yt = yp + noise
    audit = audit_residuals(yt, yp)
    fig = compose_regression_figure(yt, yp, audit=audit, panels_template="RESID_VS_PRED")
    line = _flat(fig)[0]
    lower, upper = line.band
    widths = upper - lower
    ratio = widths[-1] / max(widths[0], 1e-9)
    assert ratio >= 3.0, f"IQR band should widen >=3x across prediction range, got {ratio:.2f}x"
    assert audit.hetero_significant, "heteroscedastic synth should trip the audit hetero flag"


def test_biz_val_resid_vs_pred_flat_band_when_homoscedastic():
    """Homoscedastic counterpart: constant-variance residuals keep the band ~uniform (ratio near 1), Spearman ~ 0.

    This pins the OTHER side of the funnel detector so a future change can't make every model look heteroscedastic.
    """
    rng = np.random.default_rng(5)
    n = 8000
    yp = rng.uniform(0.0, 10.0, n)
    yt = yp + rng.normal(0.0, 1.0, n)  # constant-variance noise
    audit = audit_residuals(yt, yp)
    fig = compose_regression_figure(yt, yp, audit=audit, panels_template="RESID_VS_PRED")
    line = _flat(fig)[0]
    lower, upper = line.band
    widths = upper - lower
    ratio = widths[-1] / max(widths[0], 1e-9)
    assert 0.5 <= ratio <= 2.0, f"homoscedastic band ratio should be ~1, got {ratio:.2f}x"
    assert not audit.hetero_significant
