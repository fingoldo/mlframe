"""Charts must carry enough on-figure context to be actionable without external documentation.

Three mechanisms, all previously under-used:

* ``FigureSpec.caption`` -- the "how to read this" footnote. It existed and both renderers drew it, but
  exactly one of 42 chart builders populated it.
* Heatmap tooltips -- plotly's default reads "x: 1 / y: 13 / z: 0.684 / trace 804": grid indices, an
  unlabelled number and an internal trace id.
* Per-cell support -- a PDP surface is evaluated on a regular grid, so cells the training data barely
  covers render identically to well-supported ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.renderers import get_renderer
from mlframe.reporting.spec import FigureSpec, HeatmapPanelSpec


def test_decision_curve_caption_states_the_decision_rule_and_a_verdict():
    """The chart's whole point is a comparison against two reference policies -- say so on the figure."""
    from mlframe.reporting.charts.decision_curve import build_decision_curve_spec

    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 2, n)
    # A genuinely informative score, so the "useful" branch of the verdict is exercised.
    score = np.clip(0.5 * y + 0.5 * rng.random(n), 0, 1)

    caption = build_decision_curve_spec(y, score).figure.caption

    assert caption, "decision curve has no how-to-read caption"
    assert "net benefit" in caption.lower()
    assert "VERDICT" in caption
    # The reader must be told what the x axis MEANS operationally, not just its name.
    assert "threshold" in caption.lower()
    assert "treat all" in caption and "treat none" in caption


def test_decision_curve_caption_reports_the_negative_verdict_honestly():
    """A model that loses to the trivial policies must say so, and point at the likely cause."""
    from mlframe.reporting.charts.decision_curve import build_decision_curve_spec

    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 2, n)
    # An ANTI-correlated score, not a random one: at n=2000 a random score clears the builder's 1e-3
    # usefulness margin on noise alone (measured advantage 0.0100), so it would exercise the positive branch.
    score = 1.0 - (0.5 * y + 0.5 * rng.random(n))
    res = build_decision_curve_spec(y, score)

    assert res.useful is False
    caption = res.figure.caption
    assert "never clears" in caption
    # Naming the confound (calibration) is what makes the verdict actionable rather than just discouraging.
    assert "calibration" in caption.lower()


def test_heatmap_tooltip_names_the_axes_and_value_instead_of_x_y_z():
    """plotly's default heatmap tooltip exposes grid indices and an internal trace id; ours must not."""
    pytest.importorskip("plotly")
    panel = HeatmapPanelSpec(
        matrix=np.array([[0.1, 0.2], [0.3, 0.4]]),
        row_labels=("r0", "r1"),
        col_labels=("c0", "c1"),
        xlabel="job_post_flow_type",
        ylabel="delivery_days",
        colorbar_label="P(y=1)",
    )
    fig = get_renderer("plotly").render(FigureSpec(panels=((panel,),), figsize=(6, 4)))
    tmpl = fig.data[0].hovertemplate

    assert "job_post_flow_type" in tmpl and "delivery_days" in tmpl
    assert "P(y=1)" in tmpl
    # "<extra></extra>" is what suppresses plotly's "trace N" box.
    assert "<extra></extra>" in tmpl


def test_heatmap_cell_hovertext_is_surfaced_when_supplied():
    """A builder that knows per-cell support must be able to put it in the tooltip."""
    pytest.importorskip("plotly")
    panel = HeatmapPanelSpec(
        matrix=np.array([[0.1, 0.2]]),
        row_labels=("r0",),
        col_labels=("c0", "c1"),
        cell_hovertext=np.array([["100 rows (15.0% of 667)", "3 rows (0.4% of 667)"]], dtype=object),
    )
    fig = get_renderer("plotly").render(FigureSpec(panels=((panel,),), figsize=(6, 4)))

    assert "%{text}" in fig.data[0].hovertemplate
    assert fig.data[0].text is not None


def test_pdp_2d_tooltip_carries_row_counts_and_percentages():
    """PDP cells with almost no data must be distinguishable from well-supported ones."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier

    from mlframe.reporting.charts.pdp_ice import pdp_2d_panel

    rng = np.random.default_rng(0)
    n = 400
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=8, random_state=0).fit(X, y)

    panel = pdp_2d_panel(model, X, (0, 1), grid=5, sample=200)

    assert panel.cell_hovertext is not None
    flat = [str(s) for row in panel.cell_hovertext for s in row]
    assert all("rows" in s and "%" in s for s in flat), flat[:3]
    # Counts must sum to the finite row count, i.e. every row is attributed to exactly one cell.
    counts = [int(s.split(" rows")[0].replace(",", "")) for s in flat]
    assert sum(counts) == n, (sum(counts), n)
    # A corner of a 2-D gaussian grid is genuinely empty -- that is the signal this tooltip exists to carry.
    assert any(c == 0 for c in counts)


def _degenerate_woe_inputs(n: int = 400):
    """Frame whose every categorical level is rarer than the default ``min_support``."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"c": [f"lvl{i}" for i in range(n)]}), rng.integers(0, 2, n)


def test_woe_chart_explains_itself_instead_of_drawing_empty_axes():
    """With nothing above ``min_support`` the chart must say why, not render a blank grid.

    The degenerate branch used to return a single zero-height bar labelled "(no level above min_support)",
    which renders as full-size EMPTY axes spanning a meaningless -0.04..0.04 range -- and repeated the
    figure's own suptitle as the panel title, so the reader saw the same words twice above nothing.
    """
    from mlframe.reporting.charts.category_discriminability import compose_category_discriminability_figure
    from mlframe.reporting.spec import AnnotationPanelSpec

    X, y = _degenerate_woe_inputs()
    fig = compose_category_discriminability_figure(X, y, ["c"])
    panel = fig.panels[0][0]

    assert isinstance(panel, AnnotationPanelSpec), type(panel)
    # The reason AND the knob to turn, not just "no data".
    assert "min_support" in panel.text
    assert "high-cardinality" in panel.text
    # No duplicated heading.
    assert fig.suptitle != panel.title
    # A short message must not be stretched over a tall figure sized for a bar list.
    assert fig.figsize[1] <= 4.0, fig.figsize


def test_woe_chart_carries_a_how_to_read_caption_when_it_has_data():
    """WoE is a log-odds ratio -- a unit the reader cannot be assumed to carry in their head."""
    pd = pytest.importorskip("pandas")
    from mlframe.reporting.charts.category_discriminability import compose_category_discriminability_figure

    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame({"c": rng.choice(["a", "b", "c"], n)})
    y = X["c"].eq("a").astype(int).to_numpy()

    caption = compose_category_discriminability_figure(X, y, ["c"]).caption

    assert caption
    assert "log-odds" in caption
    assert "min_support" in caption  # the support caveat is the actionable part
