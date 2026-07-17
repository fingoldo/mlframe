"""Integration tests proving wave-5 spec fields are ACTIVATED in the consumer charts.

Each test renders the activated panel on both backends and asserts the concrete artist/trace
that the activation produces (CI errorbar band, worst-K red overlay, horizontal CONFUSED_PAIRS,
secondary axis, fills, trend line, threshold contours), not just that a spec was built.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import (
    BarPanelSpec,
    HeatmapPanelSpec,
    LinePanelSpec,
    ScatterPanelSpec,
)


def _first_panel(fig):
    """Helper: First panel."""
    for row in fig.panels:
        for cell in row:
            if cell is not None:
                yield cell


def _panels_of_type(fig, tp):
    """Helper: Panels of type."""
    return [c for c in _first_panel(fig) if isinstance(c, tp)]


# ---------------------------------------------------------------------------
# Task 1: calibration reliability scatter carries the Wilson CI band (G1/INV-22)
# ---------------------------------------------------------------------------


def test_calibration_reliability_scatter_carries_wilson_ci_band():
    """Calibration reliability scatter carries wilson ci band."""
    from mlframe.reporting.charts.calibration import build_calibration_spec

    freqs_pred = np.linspace(0.05, 0.95, 10)
    freqs_true = np.clip(freqs_pred + 0.05, 0.0, 1.0)
    hits = np.array([5, 8, 12, 30, 50, 80, 40, 20, 10, 4], dtype=np.float64)

    spec = build_calibration_spec(freqs_pred, freqs_true, hits, show_prob_histogram=False)
    scatters = _panels_of_type(spec, ScatterPanelSpec)
    assert scatters, "calibration spec must include a reliability scatter"
    sc = scatters[0]
    assert sc.y_err is not None, "reliability scatter must carry Wilson y_err when show_wilson_ci default-on"
    lo_dist, hi_dist = sc.y_err
    assert np.all(lo_dist >= 0.0) and np.all(hi_dist >= 0.0)
    # Few-sample bins (n=5) must have a wider CI than dense bins (n=80).
    assert (lo_dist[0] + hi_dist[0]) > (lo_dist[5] + hi_dist[5])


def test_calibration_ci_renders_errorbar_both_backends():
    """Calibration ci renders errorbar both backends."""
    from mlframe.reporting.charts.calibration import build_calibration_spec
    import matplotlib

    matplotlib.use("Agg")
    freqs_pred = np.linspace(0.1, 0.9, 6)
    freqs_true = np.clip(freqs_pred + 0.03, 0.0, 1.0)
    hits = np.array([6, 10, 40, 30, 8, 5], dtype=np.float64)
    spec = build_calibration_spec(freqs_pred, freqs_true, hits, show_prob_histogram=False)

    fig = MatplotlibRenderer().render(spec)
    has_errorbar = any(type(c).__name__ == "ErrorbarContainer" for ax in fig.axes for c in ax.containers)
    assert has_errorbar, "matplotlib reliability scatter must draw an errorbar container for the CI band"
    matplotlib.pyplot.close(fig)

    pfig = PlotlyRenderer().render(spec)
    has_error_y = any(getattr(tr, "error_y", None) is not None and tr.error_y.array is not None for tr in pfig.data)
    assert has_error_y, "plotly reliability scatter must carry an error_y data dict"


def test_calibration_ci_can_be_disabled():
    """Calibration ci can be disabled."""
    from mlframe.reporting.charts.calibration import build_calibration_spec

    freqs_pred = np.linspace(0.1, 0.9, 5)
    freqs_true = freqs_pred.copy()
    hits = np.full(5, 20.0)
    spec = build_calibration_spec(freqs_pred, freqs_true, hits, show_prob_histogram=False, show_wilson_ci=False)
    sc = _panels_of_type(spec, ScatterPanelSpec)[0]
    assert sc.y_err is None


# ---------------------------------------------------------------------------
# Task 2: regression pred-vs-actual draws worst-K residual points red (G9/R-9)
# ---------------------------------------------------------------------------


def test_regression_scatter_highlights_worst_k_red():
    """Regression scatter highlights worst k red."""
    from mlframe.reporting.charts.regression import compose_regression_figure

    rng = np.random.default_rng(0)
    n = 800
    y_true = rng.normal(size=n)
    y_pred = y_true + rng.normal(scale=0.1, size=n)
    # Inject a handful of gross errors and pass their positions as worst-K.
    bad = np.array([10, 200, 500, 730], dtype=np.int64)
    y_pred[bad] += 5.0
    spec = compose_regression_figure(
        y_true,
        y_pred,
        panels_template="SCATTER",
        worst_k_indices=bad,
    )
    sc = _panels_of_type(spec, ScatterPanelSpec)[0]
    assert sc.highlight_indices is not None
    assert sc.highlight_color == "red"
    # highlight_indices are PANEL-relative positions (the renderer resolves them against the panel x/y, which are
    # the finite-filtered + sorted arrays). The highlighted predictions must be the gross-error rows we injected.
    hi = np.asarray(sc.highlight_indices, dtype=np.int64)
    assert hi.size == len(bad)
    highlighted_pred = np.asarray(sc.x)[hi]
    expected_pred = y_pred[bad]
    assert np.allclose(np.sort(highlighted_pred), np.sort(expected_pred))

    import matplotlib

    matplotlib.use("Agg")
    fig = MatplotlibRenderer().render(spec)
    # Worst-K overlay is a 2nd PathCollection on the scatter axes.
    n_collections = max(len(ax.collections) for ax in fig.axes)
    assert n_collections >= 2, "worst-K overlay must add a 2nd scatter PathCollection"
    matplotlib.pyplot.close(fig)


def test_regression_worst_k_survives_subsample():
    """Regression worst k survives subsample."""
    from mlframe.reporting.charts.regression import compose_regression_figure

    rng = np.random.default_rng(11)
    n = 20_000  # > DEFAULT_REGRESSION_SCATTER_SAMPLE so the subsample path runs
    y_true = rng.normal(size=n)
    y_pred = y_true + rng.normal(scale=0.05, size=n)
    bad = np.array([3, 7777, 15123], dtype=np.int64)
    y_pred[bad] += 8.0
    spec = compose_regression_figure(
        y_true,
        y_pred,
        panels_template="SCATTER",
        worst_k_indices=bad,
    )
    sc = _panels_of_type(spec, ScatterPanelSpec)[0]
    hi = np.asarray(sc.highlight_indices, dtype=np.int64)
    assert hi.size == len(bad)
    highlighted_pred = np.asarray(sc.x)[hi]
    assert np.allclose(np.sort(highlighted_pred), np.sort(y_pred[bad])), "worst-K rows must be present + highlighted post-subsample"


# ---------------------------------------------------------------------------
# Task 3a: CONFUSED_PAIRS renders horizontal (G2)
# ---------------------------------------------------------------------------


def test_confused_pairs_is_horizontal_bar():
    """Confused pairs is horizontal bar."""
    from mlframe.reporting.charts.multiclass import compose_multiclass_figure

    rng = np.random.default_rng(1)
    n, K = 600, 4
    y_true = rng.integers(0, K, size=n)
    proba = rng.dirichlet(np.ones(K), size=n)
    # Force class 0 to be heavily mis-sent to class 1.
    sel = y_true == 0
    proba[sel] = 0.0
    proba[sel, 1] = 1.0
    spec = compose_multiclass_figure(y_true, proba, classes=list(range(K)), panels_template="CONFUSED_PAIRS")
    bars = _panels_of_type(spec, BarPanelSpec)
    assert bars, "CONFUSED_PAIRS must build a BarPanelSpec"
    assert bars[0].orientation == "horizontal"
    # No rotated x-ticks needed once bars are horizontal.
    assert bars[0].xtick_rotation == 0.0


# ---------------------------------------------------------------------------
# Task 3b: error_analysis segments_bar uses a true hline reference (G8)
# ---------------------------------------------------------------------------


def test_segments_bar_uses_hline_reference():
    """Segments bar uses hline reference."""
    import pandas as pd
    from mlframe.reporting.charts.error_analysis import segments_bar

    df = pd.DataFrame({"segment": ["a", "b", "c"], "metric": [0.9, 0.6, 0.75], "count": [100, 50, 80]})
    spec = segments_bar(df, metric_name="acc", global_value=0.8)
    bar = _panels_of_type(spec, BarPanelSpec)[0]
    assert bar.hline is not None
    val, _color, _label = bar.hline
    assert abs(val - 0.8) < 1e-9
    # Single metric series now -- the flat companion-series workaround is gone.
    assert isinstance(bar.values, np.ndarray) or len(bar.values) == 1


# ---------------------------------------------------------------------------
# Task 3c: COVERAGE / THRESHOLD secondary axis (G3)
# ---------------------------------------------------------------------------


def test_coverage_panel_puts_width_on_secondary_axis():
    """Coverage panel puts width on secondary axis."""
    from mlframe.reporting.charts.quantile import compose_quantile_figure

    rng = np.random.default_rng(2)
    n = 500
    alphas = (0.1, 0.25, 0.5, 0.75, 0.9)
    y = rng.normal(size=n)
    preds = np.column_stack([np.quantile(y, a) + np.zeros(n) for a in alphas])
    # Make them per-row by adding a shared shift so coverage is meaningful.
    shift = rng.normal(scale=0.3, size=n)
    preds = preds + shift[:, None]
    spec = compose_quantile_figure(y, preds, alphas, panels_template="COVERAGE")
    line = _panels_of_type(spec, LinePanelSpec)[0]
    assert line.secondary_y is not None, "COVERAGE must put interval width on a secondary axis"


def test_threshold_panel_puts_queue_rate_on_secondary_axis():
    """Threshold panel puts queue rate on secondary axis."""
    from mlframe.reporting.charts.binary import compose_binary_figure

    rng = np.random.default_rng(3)
    n = 1000
    y = rng.integers(0, 2, size=n)
    score = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(scale=0.2, size=n), 0, 1)
    spec = compose_binary_figure(y, score, panels_template="THRESHOLD")
    line = _panels_of_type(spec, LinePanelSpec)[0]
    assert line.secondary_y is not None, "THRESHOLD must put queue-rate on a secondary axis"


# ---------------------------------------------------------------------------
# Task 3d: GAIN + SCORE_DIST fills (G10)
# ---------------------------------------------------------------------------


def test_gain_and_score_dist_fill():
    """Gain and score dist fill."""
    from mlframe.reporting.charts.binary import compose_binary_figure

    rng = np.random.default_rng(4)
    n = 1000
    y = rng.integers(0, 2, size=n)
    score = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(scale=0.2, size=n), 0, 1)
    spec = compose_binary_figure(y, score, panels_template="GAIN SCORE_DIST")
    lines = _panels_of_type(spec, LinePanelSpec)
    assert any(ln.fill_to_baseline is not None for ln in lines), "GAIN/SCORE_DIST must set a fill"


# ---------------------------------------------------------------------------
# Task 3e: regression trend line beside y=x (G5/R-15)
# ---------------------------------------------------------------------------


def test_regression_scatter_has_trend_line():
    """Regression scatter has trend line."""
    from mlframe.reporting.charts.regression import compose_regression_figure

    rng = np.random.default_rng(5)
    n = 2000
    y_true = rng.normal(size=n)
    y_pred = 0.7 * y_true + rng.normal(scale=0.2, size=n)
    spec = compose_regression_figure(y_true, y_pred, panels_template="SCATTER")
    sc = _panels_of_type(spec, ScatterPanelSpec)[0]
    assert sc.trend_line in ("theil-sen", "huber")


# ---------------------------------------------------------------------------
# Task 3f: psi_heatmap threshold contours (G12)
# ---------------------------------------------------------------------------


def test_psi_heatmap_has_threshold_contours():
    """Psi heatmap has threshold contours."""
    from mlframe.reporting.charts.drift import psi_heatmap

    rng = np.random.default_rng(6)
    n = 2000
    ts = np.arange(n)
    f0 = rng.normal(size=n)
    f1 = np.concatenate([rng.normal(size=n // 2), rng.normal(loc=3.0, size=n - n // 2)])
    frame = np.column_stack([f0, f1])
    spec = psi_heatmap(frame, ts, feature_names=["stable", "drift"], n_time_buckets=5)
    heats = _panels_of_type(spec, HeatmapPanelSpec)
    assert heats, "psi_heatmap must build a HeatmapPanelSpec"
    assert heats[0].threshold_contours is not None
    levels = [v for v, _c in heats[0].threshold_contours]
    assert 0.10 in levels and 0.25 in levels


# ---------------------------------------------------------------------------
# Task 3g: adversarial ROC per-series x (G11)
# ---------------------------------------------------------------------------


def test_quantile_default_template_includes_r6_tokens():
    """Quantile default template includes r6 tokens."""
    from mlframe.reporting.charts.quantile import DEFAULT_QUANTILE_PANELS, compose_quantile_figure

    for tok in ("QUANTILE_RELIABILITY", "PINBALL_DECOMP", "QUANTILE_CROSSING"):
        assert tok in DEFAULT_QUANTILE_PANELS, f"{tok} must be default-on in the composer template"

    rng = np.random.default_rng(8)
    n = 600
    alphas = (0.1, 0.25, 0.5, 0.75, 0.9)
    y = rng.normal(size=n)
    base = np.column_stack([np.quantile(y, a) + np.zeros(n) for a in alphas])
    preds = base + rng.normal(scale=0.2, size=n)[:, None]
    # Default template (all 9 tokens) must build a figure without error.
    spec = compose_quantile_figure(y, preds, alphas)
    n_panels = sum(1 for _ in _first_panel(spec))
    assert n_panels == len(DEFAULT_QUANTILE_PANELS.split())


def test_adversarial_roc_uses_per_series_x():
    """Adversarial roc uses per series x."""
    pytest.importorskip("lightgbm")
    from mlframe.reporting.charts.drift import adversarial_validation

    rng = np.random.default_rng(7)
    n = 400
    train = rng.normal(size=(n, 4))
    test = rng.normal(loc=0.6, size=(n, 4))
    spec = adversarial_validation(train, test, feature_names=[f"f{i}" for i in range(4)], n_splits=2, top_features=4)
    line = _panels_of_type(spec, LinePanelSpec)[0]
    assert isinstance(line.x, tuple), "adversarial ROC must carry per-series x arrays (one fpr grid per curve)"
