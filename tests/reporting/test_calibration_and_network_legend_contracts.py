"""Two reporting contracts that said one thing and did another.

`build_calibration_spec` documented three parameters as "(default on)" while all three default to False, and
inline comments ninety lines below in the same function stated the opposite. A caller reading the docstring
expected three overlays and got none.

`_plotly_network` set `showlegend=True` at figure level from inside the panel body to make its node-class
legend visible -- but `render()` sets `layout.showlegend` AFTER every panel has run, so the set was
overwritten and the legend silently vanished on the default interactive HTML backend while matplotlib drew
it. The reader saw green / red / amber nodes with nothing saying which class was which. It is the same
figure-level-property-set-from-one-panel trap the `barmode` comment in the same file already documents.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from mlframe.reporting.charts.calibration import build_calibration_spec


class TestTheDocumentedDefaultsAreTheRealOnes:
    """A docstring that contradicts the signature is worse than no docstring."""

    DOC = build_calibration_spec.__doc__ or ""
    SIG = inspect.signature(build_calibration_spec)

    @pytest.mark.parametrize("name", ["show_ece_annotation", "reliability_smoothed", "reliability_band"])
    def test_an_off_parameter_is_not_documented_as_on(self, name):
        """All three defaulted to False while the docstring said "(default on)"."""
        assert self.SIG.parameters[name].default is False
        assert f"``{name}`` (default on" not in self.DOC

    @pytest.mark.parametrize("name", ["show_ece_annotation", "reliability_smoothed", "reliability_band"])
    def test_the_docstring_says_it_is_off(self, name):
        """Silence would leave the reader guessing; the text has to state the default."""
        idx = self.DOC.find(f"``{name}``")
        assert idx >= 0, f"{name} is no longer documented at all"
        assert "off by default" in self.DOC[idx : idx + 200]

    def test_a_genuinely_on_parameter_is_still_documented_as_on(self):
        """The correction must not flip a claim that was right."""
        assert self.SIG.parameters["show_wilson_ci"].default is True
        assert "``show_wilson_ci`` (default on)" in self.DOC


class TestTheNetworkLegendSurvivesRender:
    """`render()` runs after the panels, so a panel cannot set a figure-level property and expect it to stick."""

    def _spec(self):
        """A one-panel network figure carrying node-class legend keys."""
        from mlframe.reporting.spec import FigureSpec, NetworkPanelSpec

        rng = np.random.default_rng(0)
        n = 6
        panel = NetworkPanelSpec(
            node_x=rng.random(n),
            node_y=rng.random(n),
            node_size=np.full(n, 10.0),
            node_color=tuple(["#2ca02c", "#d62728", "#ff7f0e"] * 2),
            node_label=tuple(f"n{i}" for i in range(n)),
            edge_src=np.array([0, 1]),
            edge_dst=np.array([1, 2]),
            edge_weight=np.array([0.5, 0.8]),
            node_legend=(("kept", "#2ca02c"), ("dropped", "#d62728"), ("borderline", "#ff7f0e")),
            title="net",
        )
        return FigureSpec(suptitle="s", panels=((panel,),))

    def test_the_figure_shows_a_legend(self):
        """The panel's own `update_layout(showlegend=True)` was overwritten a few lines later."""
        pytest.importorskip("plotly")
        from mlframe.reporting.renderers.plotly import PlotlyRenderer

        fig = PlotlyRenderer().render(self._spec())
        assert fig.layout.showlegend, "the node-class legend is off, so the colours mean nothing to the reader"

    def test_the_legend_proxy_traces_are_present(self):
        """A legend with no keys would be just as useless."""
        pytest.importorskip("plotly")
        from mlframe.reporting.renderers.plotly import PlotlyRenderer

        fig = PlotlyRenderer().render(self._spec())
        named = {tr.name for tr in fig.data if tr.showlegend}
        assert {"kept", "dropped", "borderline"} <= named, named

    def test_the_legend_does_not_depend_on_panel_ORDER(self):
        """The panel setting `showlegend` itself is what made the outcome depend on which panel rendered last.

        Asserted by rendering the same two panels in both orders: the legend must be on either way. The old
        form checked that the renderer's source no longer contains `update_layout(showlegend=True)`, which
        says nothing about the property -- the figure-level switch still has to end up True, as the sibling
        above pins, and what broke was only WHERE it was set.
        """
        pytest.importorskip("plotly")
        from mlframe.reporting.renderers.plotly import PlotlyRenderer
        from mlframe.reporting.spec import FigureSpec, LinePanelSpec

        net = self._spec().panels[0][0]
        line = LinePanelSpec(x=np.arange(5), y=(np.arange(5).astype(float),), series_labels=("a",), title="plain")

        for order in ((net, line), (line, net)):
            fig = PlotlyRenderer().render(FigureSpec(suptitle="s", panels=(order,)))
            assert fig.layout.showlegend, f"the legend is off when the network panel is rendered {'first' if order[0] is net else 'last'}"

    def test_a_panel_without_a_legend_does_not_force_one_on(self):
        """The predicate must stay specific, or every multi-panel figure grows legend soup."""
        pytest.importorskip("plotly")
        from mlframe.reporting.renderers.plotly import PlotlyRenderer
        from mlframe.reporting.spec import FigureSpec, LinePanelSpec

        panel = LinePanelSpec(x=np.arange(5.0), y=np.arange(5.0), title="t")
        assert not PlotlyRenderer().render(FigureSpec(suptitle="s", panels=((panel,),))).layout.showlegend
