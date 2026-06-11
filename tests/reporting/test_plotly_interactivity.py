"""Plotly renderer interactivity: HTML reports must be explorable, with each interactivity prop gated to the panel
types where it is correct (unified hover on lines, rangeslider on temporal only, rich hovertemplates on ROC/PR/calib).
HTML isn't visually diff-able here, so these assert the resulting go.Figure carries the expected props.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.renderers import get_renderer
from mlframe.reporting.spec import (
    FigureSpec, HeatmapPanelSpec, LinePanelSpec, ScatterPanelSpec,
)


@pytest.fixture
def renderer():
    return get_renderer("plotly")


def _layout(fig):
    return fig.layout.to_plotly_json()


def _rangeslider_visibles(fig):
    lay = _layout(fig)
    return [lay[k].get("rangeslider", {}).get("visible") for k in lay if k.startswith("xaxis")]


@pytest.fixture
def roc_panel():
    x = np.linspace(0, 1, 12)
    return LinePanelSpec(x=(x, x), y=(np.sqrt(x), x), series_labels=("model", "chance"),
                         title="ROC", xlabel="FPR", ylabel="TPR")


@pytest.fixture
def temporal_panel():
    return LinePanelSpec(x=np.arange(20.0), y=np.random.default_rng(0).random(20),
                         title="metric over time", xlabel="time", ylabel="AUC", x_is_time=True)


@pytest.fixture
def scatter_panel():
    x = np.linspace(0, 1, 50)
    return ScatterPanelSpec(x=x, y=x + 0.1, title="resid", xlabel="pred", ylabel="resid")


@pytest.fixture
def heatmap_panel():
    m = np.array([[0.9, 0.1], [0.2, 0.8]])
    return HeatmapPanelSpec(matrix=m, row_labels=("a", "b"), col_labels=("p", "q"), title="cm")


def _fig(renderer, panel):
    return renderer.render(FigureSpec(panels=((panel,),), figsize=(6, 4)))


def test_line_panel_gets_unified_hover(renderer, roc_panel):
    assert _layout(_fig(renderer, roc_panel)).get("hovermode") == "x unified"


def test_scatter_does_not_get_unified_hover(renderer, scatter_panel):
    # Unified hover misreads a scatter (it would show every point at a shared x); must stay plotly default.
    assert _layout(_fig(renderer, scatter_panel)).get("hovermode") != "x unified"


def test_heatmap_does_not_get_unified_hover(renderer, heatmap_panel):
    assert _layout(_fig(renderer, heatmap_panel)).get("hovermode") != "x unified"


def test_mixed_line_and_heatmap_skips_unified_hover(renderer, roc_panel, heatmap_panel):
    fig = renderer.render(FigureSpec(panels=((roc_panel, heatmap_panel),), figsize=(12, 4)))
    assert _layout(fig).get("hovermode") != "x unified"


def test_legend_click_toggles_set(renderer, roc_panel):
    leg = _layout(_fig(renderer, roc_panel))["legend"]
    assert leg.get("itemclick") == "toggle"
    assert leg.get("itemdoubleclick") == "toggleothers"


def test_temporal_panel_gets_rangeslider(renderer, temporal_panel):
    assert True in _rangeslider_visibles(_fig(renderer, temporal_panel))


def test_nontemporal_line_has_no_rangeslider(renderer, roc_panel):
    # Pin: a rangeslider must NOT creep onto non-temporal charts (it wastes vertical space).
    assert True not in _rangeslider_visibles(_fig(renderer, roc_panel))


def test_scatter_has_no_rangeslider(renderer, scatter_panel):
    assert True not in _rangeslider_visibles(_fig(renderer, scatter_panel))


def test_heatmap_has_no_rangeslider(renderer, heatmap_panel):
    assert True not in _rangeslider_visibles(_fig(renderer, heatmap_panel))


def test_roc_traces_carry_rich_hovertemplate(renderer, roc_panel):
    fig = _fig(renderer, roc_panel)
    line_tmpls = [t.hovertemplate for t in fig.data if "lines" in (t.mode or "")]
    assert line_tmpls and all(t is not None and "FPR=" in t and "TPR=" in t for t in line_tmpls)


def test_generic_line_gets_fallback_hovertemplate(renderer):
    x = np.arange(8.0)
    p = LinePanelSpec(x=x, y=(x, x * 2), series_labels=("s1", "s2"), xlabel="step", ylabel="loss")
    fig = _fig(renderer, p)
    line_tmpls = [t.hovertemplate for t in fig.data if "lines" in (t.mode or "")]
    assert line_tmpls and all(t is not None for t in line_tmpls)


def test_modebar_drops_lasso_and_logo(renderer, roc_panel):
    fig = _fig(renderer, roc_panel)
    assert "lasso2d" in (fig.layout.modebar.remove or ())
    from mlframe.reporting.renderers._plotly_interactivity import html_config
    cfg = html_config()
    assert cfg["displaylogo"] is False
    assert "lasso2d" in cfg["modeBarButtonsToRemove"]


def test_html_output_carries_interactivity(renderer, temporal_panel, tmp_path):
    fig = _fig(renderer, temporal_panel)
    out = str(tmp_path / "t.html")
    renderer.save(fig, out, "html")
    with open(out, encoding="utf-8") as f:
        html = f.read()
    assert "x unified" in html
    assert "rangeslider" in html
    assert '"displaylogo": false' in html
