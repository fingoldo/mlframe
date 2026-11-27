"""Cluster 21 regression: the plotly scatter keeps its error bars when it is downsampled, on the same rows as x/y (audit 2026-11-20, 21-F9)."""

import numpy as np

from mlframe.reporting.renderers._plotly_scatter import _subset_error_spec


def test_symmetric_and_asymmetric_errors_follow_the_kept_rows():
    """Per-point arrays are indexed by the selection; a scalar passes through; no selection or no error returns the input (21-F9)."""
    err = np.arange(10, dtype=float)
    idx = np.array([7, 2, 5])
    np.testing.assert_array_equal(_subset_error_spec(err, idx, 10), [7.0, 2.0, 5.0])
    lo, hi = _subset_error_spec((err, err + 100), idx, 10)
    np.testing.assert_array_equal(lo, [7.0, 2.0, 5.0])
    np.testing.assert_array_equal(hi, [107.0, 102.0, 105.0])
    assert _subset_error_spec(0.5, idx, 10) == 0.5
    assert _subset_error_spec(None, idx, 10) is None
    assert _subset_error_spec(err, None, 10) is err


def test_downsampled_scatter_renders_error_bars(monkeypatch):
    """With the cap lowered below the point count, the figure's trace still carries error_y with the sampled length (21-F9)."""
    from mlframe.reporting.renderers import _plotly_scatter as ps
    from mlframe.reporting.renderers.plotly import PlotlyRenderer
    from mlframe.reporting.spec import FigureSpec, ScatterPanelSpec

    monkeypatch.setattr(ps, "_SCATTER_MAX_POINTS", 50)
    n = 400
    x = np.linspace(0, 1, n)
    panel = ScatterPanelSpec(x=x, y=np.sin(6 * x), y_err=np.full(n, 0.1) + 0.001 * np.arange(n))
    fig = PlotlyRenderer().render(FigureSpec(panels=((panel,),), suptitle="t"))
    traced = [t for t in fig.data if getattr(t, "error_y", None) is not None and getattr(t.error_y, "array", None) is not None]
    assert traced, "an error_y trace must survive the downsample"
    assert all(len(t.error_y.array) == len(t.x) for t in traced)
    assert max(len(t.x) for t in traced) <= 50 + 5
