"""Cluster 21 regression: the heatmap tick budget is applied once, after the final layout (audit 2026-11-20, finding 21-F19).

Pinned by source: _heatmap no longer calls apply_heatmap_tick_budget, and the renderer's post-layout loop still does.
"""

from pathlib import Path

from mlframe.reporting.renderers import _plotly_heatmap as ph
from mlframe.reporting.renderers import plotly as pr

_HEAT = Path(ph.__file__).resolve()
_PLOTLY = Path(pr.__file__).resolve()


def test_heatmap_does_not_budget_ticks_early():
    """The panel draw does not call the tick budget itself (21-F19)."""
    text = _HEAT.read_text(encoding="utf-8")
    assert text.count("apply_heatmap_tick_budget(fig, p, row, col)") == 0


def test_renderer_still_applies_the_budget_after_layout():
    """The post-layout loop applies the budget to every heatmap panel (21-F19)."""
    text = _PLOTLY.read_text(encoding="utf-8")
    assert "apply_heatmap_tick_budget(fig, _panel, _r, _c)" in text
