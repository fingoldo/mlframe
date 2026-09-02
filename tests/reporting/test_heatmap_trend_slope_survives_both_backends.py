"""The plotly heatmap moved the robust trend line's endpoints, which changed its slope.

`robust_fit_endpoints` returns y PREDICTED at the x extremes, so with a slope above 1 those endpoints fall
outside the data's own range. matplotlib maps value space to bin-index space with a pure affine map and lets
`set_xlim`/`set_ylim` clip the drawn segment, so the visible portion keeps the correct slope. plotly instead
rounded and CLAMPED each endpoint to the nearest category label, moving the endpoint itself to the axis edge --
so the two-point segment was drawn at a different slope, defeating the panel's stated purpose: "so a systematic
slope bias is visible even when the cloud hugs the diagonal".

The same helper resolved BOTH coordinates against `col_labels`. That is latent for the hexbin builder, which
sets row and column labels from the same centres, but any spec with `trend_line` and asymmetric labels put the
trend at a y category that does not exist on the y axis -- and plotly appends such a category rather than
raising.

The map now lives in `_shared_helpers` so the two backends cannot drift on it again.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.renderers._shared_helpers import heatmap_value_to_index


class TestTheSharedMapIsAffineAndUnclamped:
    """Clipping is the axis's job; moving an endpoint changes the line."""

    MAP = staticmethod(heatmap_value_to_index(0.0, 10.0, 11))

    def test_the_endpoints_of_the_range_land_on_the_end_bins(self):
        """The basic contract both renderers relied on."""
        assert self.MAP(0.0) == pytest.approx(0.0) and self.MAP(10.0) == pytest.approx(10.0)

    def test_it_is_linear_in_between(self):
        """A non-affine map would bend the trend line."""
        assert self.MAP(2.5) == pytest.approx(2.5) and self.MAP(7.5) == pytest.approx(7.5)

    def test_an_extrapolated_value_is_not_clamped(self):
        """The defect: clamping to `[0, n-1]` is what moved the endpoint and changed the slope."""
        assert self.MAP(12.0) > 10.0 and self.MAP(-2.0) < 0.0

    def test_it_returns_fractional_positions(self):
        """Rounding to the nearest bin quantises the endpoint, which also perturbs the slope."""
        assert self.MAP(0.5) == pytest.approx(0.5)

    def test_a_degenerate_range_does_not_divide_by_zero(self):
        """`hi == lo` happens on a constant column."""
        assert heatmap_value_to_index(3.0, 3.0, 11)(3.0) == 0.0


class TestTheTwoBackendsAgreeOnTheSlope:
    """The panel's purpose is the slope, so that is what must match."""

    def _endpoints(self, n_bins=80):
        """A trend whose predicted endpoints fall outside the data range, which is where the clamp bit."""
        lo, hi = 0.0, 10.0
        to_idx = heatmap_value_to_index(lo, hi, n_bins)
        # slope 2 through the middle: at x=0 the prediction is -5, at x=10 it is 15 -- both outside [0, 10]
        return to_idx(0.0), to_idx(-5.0), to_idx(10.0), to_idx(15.0)

    def test_the_mapped_slope_equals_the_value_space_slope(self):
        """An affine map preserves slope exactly; a clamped one does not."""
        x0, y0, x1, y1 = self._endpoints()
        assert (y1 - y0) / (x1 - x0) == pytest.approx(2.0)

    def test_a_clamping_map_would_have_changed_it(self):
        """States what the old behaviour did, so the test above is not vacuous."""
        n_bins = 80
        to_idx = heatmap_value_to_index(0.0, 10.0, n_bins)

        def _clamped(v):
            """The old plotly form: affine, then round and clamp to a real category index."""
            return float(round(min(max(to_idx(v), 0.0), n_bins - 1.0)))

        slope = (_clamped(15.0) - _clamped(-5.0)) / (_clamped(10.0) - _clamped(0.0))
        assert slope != pytest.approx(2.0), "the clamp no longer changes the slope; this fixture proves nothing"

    def test_the_y_map_uses_the_row_axis(self):
        """It resolved both coordinates against `col_labels`; asymmetric labels put y on a nonexistent category."""
        import ast
        import pathlib

        src = (pathlib.Path(__file__).resolve().parents[2] / "src" / "mlframe" / "reporting" / "renderers" / "_plotly_heatmap.py").read_text(encoding="utf-8")
        calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "heatmap_value_to_index"]
        assert len(calls) == 2, f"expected an x map and a y map, found {len(calls)}"
        args = [ast.dump(c.args[-1]) for c in calls]
        assert any("row_labels" in a for a in args), f"neither map is built from row_labels: {args}"


def test_the_plotly_trace_carries_numeric_positions():
    """A category axis accepts a numeric position; the old code sent label STRINGS, which is what snapped them."""
    pytest.importorskip("plotly")
    from mlframe.reporting.renderers.plotly import PlotlyRenderer
    from mlframe.reporting.spec import FigureSpec, HeatmapPanelSpec

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    y = 2.0 * x + rng.normal(0, 0.2, 500)
    nb = 20
    edges = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), nb + 1)
    centres = tuple(f"{v:.2f}" for v in (edges[:-1] + edges[1:]) / 2)
    mat, _, _ = np.histogram2d(y, x, bins=[edges, edges])
    panel = HeatmapPanelSpec(
        matrix=mat,
        row_labels=centres,
        col_labels=centres,
        title="t",
        trend_line="theil-sen",
        trend_xy=(x, y),
    )
    fig = PlotlyRenderer().render(FigureSpec(suptitle="s", panels=((panel,),)))
    trend = [tr for tr in fig.data if "robust fit" in (tr.name or "")]
    assert trend, "the robust-fit trace was not drawn"
    assert all(isinstance(v, (int, float, np.floating)) for v in list(trend[0].x) + list(trend[0].y)), (trend[0].x, trend[0].y)
