"""Cluster 20 regression: reliability curves sit at each bin's mean prediction, one x array per series (audit 2026-11-20, 20-F8)."""

import numpy as np

from mlframe.reporting.charts.multilabel import _calib_grid_panel


def test_a_calibrated_off_centre_bin_lies_on_the_diagonal():
    """Scores of 0.11 with an 11% positive rate: the point is at x=0.11, y=0.11, not at the 0.15 bin centre (20-F8)."""
    n = 10000
    y = np.zeros((n, 1))
    y[: int(n * 0.11), 0] = 1
    proba = np.full((n, 1), 0.11)
    panel = _calib_grid_panel(y, proba, ["a"])
    assert isinstance(panel.x, tuple) and len(panel.x) == len(panel.y) == 2
    curve_x, curve_y = np.asarray(panel.x[1]), np.asarray(panel.y[1])
    ok = ~np.isnan(curve_y)
    assert ok.sum() == 1
    np.testing.assert_allclose(curve_x[ok], [0.11])
    np.testing.assert_allclose(curve_y[ok], [0.11], atol=1e-9)
    assert panel.xlabel == "Mean predicted P(label)"


def test_empty_bins_have_nan_x_and_the_diagonal_keeps_the_grid():
    """Bins with no rows are NaN in both x and y; the 'perfect' diagonal still uses the uniform grid (20-F8)."""
    rng = np.random.default_rng(0)
    y = (rng.random((500, 2)) < 0.3).astype(float)
    proba = rng.uniform(0.0, 0.5, size=(500, 2))
    panel = _calib_grid_panel(y, proba, ["a", "b"])
    assert len(panel.x) == len(panel.y) == 3
    np.testing.assert_allclose(panel.x[0], np.linspace(0.05, 0.95, 10))
    for xs, ys in zip(panel.x[1:], panel.y[1:]):
        assert np.array_equal(np.isnan(xs), np.isnan(np.asarray(ys)))
        assert np.isnan(xs).any()


def test_pooled_average_series_gets_its_own_mean_prediction_x():
    """With a label subset, the pooled series is one more (x, y) pair at the pooled per-bin mean prediction (20-F8)."""
    rng = np.random.default_rng(1)
    y = (rng.random((800, 3)) < 0.4).astype(float)
    proba = rng.uniform(size=(800, 3))
    panel = _calib_grid_panel(y, proba, ["a", "b", "c"], label_subset=[0, 2])
    assert len(panel.x) == len(panel.y) == 4


def test_both_backends_render_the_per_series_x_panel():
    """The panel with a tuple of x arrays renders on plotly and on matplotlib without error (20-F8)."""
    import matplotlib

    matplotlib.use("Agg")
    from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
    from mlframe.reporting.renderers.plotly import PlotlyRenderer
    from mlframe.reporting.spec import FigureSpec

    rng = np.random.default_rng(2)
    y = (rng.random((600, 2)) < 0.3).astype(float)
    proba = rng.uniform(size=(600, 2))
    spec = FigureSpec(panels=((_calib_grid_panel(y, proba, ["a", "b"]),),), suptitle="reliability")
    assert len(PlotlyRenderer().render(spec).data) >= 3
    assert MatplotlibRenderer().render(spec) is not None
