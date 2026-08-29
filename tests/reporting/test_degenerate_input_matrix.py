"""Degenerate-input matrix: every panel type, every backend, on empty / all-NaN / single-row input.

The class of bug this exists to catch is a spec ONE backend renders happily and the other raises on -- so the PNG
and the HTML disagree with no signal, and the failure only surfaces on whichever backend a given suite happens to
run. Asserting "both backends behave the same way" is the point; asserting "it renders" is secondary.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from mlframe.reporting.renderers import get_renderer
from mlframe.reporting.spec import (
    AnnotationPanelSpec, BarPanelSpec, ConfusionMarginsPanelSpec, FigureSpec, HeatmapPanelSpec,
    HistogramPanelSpec, LinePanelSpec, ScatterPanelSpec, ViolinPanelSpec,
)

EMPTY = np.empty(0, dtype=np.float64)
NANS = np.full(8, np.nan, dtype=np.float64)
ONE = np.array([1.0])


def _panels():
    """(id, panel) for every panel type under each degenerate shape it can be handed."""
    labels = ("a", "b")
    return [
        ("scatter-empty", ScatterPanelSpec(x=EMPTY, y=EMPTY)),
        ("scatter-all-nan", ScatterPanelSpec(x=NANS, y=NANS)),
        ("scatter-single-row", ScatterPanelSpec(x=ONE, y=ONE)),
        ("line-empty", LinePanelSpec(x=EMPTY, y=EMPTY)),
        ("line-all-nan", LinePanelSpec(x=NANS, y=NANS)),
        ("line-single-row", LinePanelSpec(x=ONE, y=ONE)),
        ("bar-empty", BarPanelSpec(categories=(), values=EMPTY)),
        ("bar-single", BarPanelSpec(categories=("a",), values=ONE)),
        ("bar-all-nan", BarPanelSpec(categories=("a", "b"), values=np.full(2, np.nan))),
        ("hist-empty", HistogramPanelSpec(values=EMPTY)),
        ("hist-all-nan", HistogramPanelSpec(values=NANS)),
        ("violin-empty-group", ViolinPanelSpec(groups=(np.random.default_rng(0).normal(size=20), EMPTY), group_labels=labels)),
        ("violin-all-empty", ViolinPanelSpec(groups=(EMPTY,), group_labels=("a",))),
        ("violin-all-nan", ViolinPanelSpec(groups=(NANS,), group_labels=("a",))),
        ("heatmap-0x0", HeatmapPanelSpec(matrix=np.empty((0, 0)), row_labels=(), col_labels=())),
        ("heatmap-all-nan", HeatmapPanelSpec(matrix=np.full((2, 2), np.nan), row_labels=labels, col_labels=labels)),
        ("confusion-margins-zero", ConfusionMarginsPanelSpec(
            matrix=np.zeros((2, 2)), row_labels=labels, col_labels=labels,
            row_margin=np.zeros(2), col_margin=np.zeros(2),
            row_margin_label="support", col_margin_label="volume",
        )),
        ("annotation", AnnotationPanelSpec(text="nothing to show", title="t")),
    ]


@pytest.mark.parametrize(("panel_id", "panel"), _panels(), ids=[p[0] for p in _panels()])
def test_both_backends_agree_on_degenerate_input(panel_id, panel):
    """Neither backend may raise where the other succeeds -- that is the divergence this matrix exists to catch."""
    outcomes = {}
    for backend in ("matplotlib", "plotly"):
        try:
            get_renderer(backend).render(FigureSpec(panels=((panel,),), figsize=(5.0, 4.0)))
            outcomes[backend] = "rendered"
        except Exception as exc:  # the exception TYPE is the assertion subject, so it must be caught here
            outcomes[backend] = f"{type(exc).__name__}: {exc}"
    assert (
        outcomes["matplotlib"] == outcomes["plotly"] == "rendered"
    ), f"{panel_id}: backends disagree -- matplotlib={outcomes['matplotlib']!r}, plotly={outcomes['plotly']!r}"


def test_the_matrix_covers_every_panel_type():
    """A new panel type must be added here too, or it ships without degenerate coverage."""
    from mlframe.reporting import spec as spec_mod

    covered = {type(p).__name__ for _pid, p in _panels()}
    declared = {name for name in dir(spec_mod) if name.endswith("PanelSpec") and name not in ("PanelSpec",)}
    # NetworkPanelSpec is exercised by the spectral-embedding builder's own degenerate tests, which construct a real
    # graph; a bare node/edge array here would not resemble anything a builder produces.
    missing = declared - covered - {"NetworkPanelSpec"}
    assert not missing, f"panel types with no degenerate-input coverage: {sorted(missing)}"
