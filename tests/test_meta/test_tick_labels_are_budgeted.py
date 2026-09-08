"""Every place a renderer writes its own tick labels must budget them against the axis first.

This wave fixed the same defect three times -- heatmap ticks, violin group labels, bar category labels --
in each case a renderer choosing HOW MANY labels to draw from the category count rather than from the room
the axis actually has. The fixes now share one budget (``ticks_that_fit`` plus a pitch helper), and this
gate exists so the fourth site cannot quietly skip it: a new panel type that sets ``ticktext`` or
``set_?ticklabels`` from an unbudgeted list is exactly how the class comes back.

Structural by necessity. "This call site consulted the budget" has no behavioural signature that a test
could read off a rendered figure without also fixing the figure size, the label lengths and the backend --
and it is the ABSENCE of a consultation that is the bug. Asserted on the parsed function so reformatting
and renamed locals do not move it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

RENDERERS = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "reporting" / "renderers"

#: Calls that put a caller-chosen list of labels on an axis. Each one has to be preceded, in the same
#: function, by a consultation of the shared budget.
LABEL_SETTERS = ("set_xticklabels", "set_yticklabels")

#: The shared budget helpers. A function may reach the budget through any of them.
BUDGET_NAMES = (
    "ticks_that_fit",
    "_thin_tick_positions",
    "rotated_tick_pitch_in",
    "label_width_pitch_in",
    "network_label_indices",
    "non_colliding_label_indices",
    # Delegating to one of the dedicated budget passes counts: those run after the layout is final, which
    # is the only point at which the axis length is known on the plotly side.
    "_bar_tick_budget",
    "_violin_tick_budget",
    "apply_heatmap_tick_budget",
)

#: Sites that legitimately draw a FIXED, tiny label set that no axis length can fail to hold, with the
#: reason each one is exempt. A new entry here needs the same kind of reason, not just a name.
EXEMPT = {
    # Two ticks by construction (the low/high ends of a colour scale), so there is nothing to thin.
    "MatplotlibRenderer._confusion_margins",
    # Sets the FULL truncated label list on purpose and is thinned afterwards by
    # ``PlotlyRenderer._bar_tick_budget``, which ``render()`` runs once the final ``update_layout`` has
    # fixed the margins -- the axis length simply is not known while the panel is being drawn. The gate
    # below pins that the budget pass still exists and still reaches the bar axis.
    "PlotlyRenderer._bar",
}


def _functions_with_label_setters():
    """``(module, qualified function name, node)`` for every function that sets tick labels itself."""
    found = []
    for path in sorted(RENDERERS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = {node: node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = {child.func.attr for child in ast.walk(node) if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)}
            keywords = {kw.arg for child in ast.walk(node) if isinstance(child, ast.Call) for kw in child.keywords}
            if not (calls & set(LABEL_SETTERS) or "ticktext" in keywords):
                continue
            owner = classes.get(parents.get(node))
            found.append((path.name, f"{owner}.{node.name}" if owner else node.name, node))
    return found


def test_the_gate_is_actually_watching_something():
    """A structural gate that matches nothing passes forever; pin that it still sees the known sites."""
    found = _functions_with_label_setters()
    assert len(found) >= 5, f"the tick-label scan found only {len(found)} sites; the patterns have gone stale"


@pytest.mark.parametrize("case", _functions_with_label_setters(), ids=lambda c: f"{c[0]}::{c[1]}")
def test_a_tick_label_writer_consults_the_shared_budget(case):
    """Choosing the label count from the category count is what put 30 overlapping labels on a 7in axis."""
    module, qualname, node = case
    if qualname in EXEMPT:
        pytest.skip(f"{qualname} is exempt for a recorded reason; see the EXEMPT entry above it")
    names = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
    names |= {child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)}
    assert names & set(BUDGET_NAMES), (
        f"{module}::{qualname} sets tick labels without consulting any of {BUDGET_NAMES}. "
        "Deriving the label count from the category count instead of the axis length is the defect this gate exists for."
    )


def test_the_deferred_bar_budget_still_runs_after_the_layout():
    """``PlotlyRenderer._bar`` is exempt only because a later pass thins its labels; pin that pass.

    Without this the exemption above would be a hole: dropping the ``_bar_tick_budget`` call from
    ``render()`` would leave every bar label drawn and the gate silent about it.
    """
    import numpy as np

    from mlframe.reporting.renderers.plotly import PlotlyRenderer
    from mlframe.reporting.spec import BarPanelSpec, FigureSpec

    cats = tuple(f"category_number_{i}" for i in range(80))
    panel = BarPanelSpec(categories=cats, values=np.linspace(0.0, 1.0, len(cats)), title="t", orientation="horizontal")
    fig = PlotlyRenderer().render(FigureSpec(panels=((panel,),), figsize=(6.0, 4.0)))
    drawn = list(fig.layout.yaxis.ticktext or ())
    assert 0 < len(drawn) < len(cats), f"{len(drawn)} of {len(cats)} bar labels survived; the deferred budget did not run"
