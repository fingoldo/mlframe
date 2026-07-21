"""Regression guards for pyplot lifecycle fixes.

INV-50: plot_pit_diagram + the extractors-showcase histogram created pyplot figures and
called a bare plt.show() that never closed them -> figure-registry leaks in long sessions
and discarded work in headless runs. They now save (when given a path) and always close.

INV-53: the lifecycle helpers used plt.ion(), which is process-global and never reverted,
leaking interactive mode into the user's session. _show_plots_unless_agg now uses
plt.show(block=False) and never flips plt.isinteractive().
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlframe.calibration.quality import build_pit_diagram_spec, plot_pit_diagram
from mlframe.metrics.calibration import _show_plots_unless_agg


def test_plot_pit_diagram_saves_and_closes(tmp_path):
    """Plot pit diagram saves and closes."""
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.0, 1.0, size=500)
    labels = (rng.uniform(size=500) < probs).astype(int)
    out = str(tmp_path / "pit")  # extension-less on purpose
    n_before = len(plt.get_fignums())
    plot_pit_diagram(predicted_probs=probs, true_labels=labels, plot_file=out)
    import os

    assert os.path.exists(out + ".png"), "PIT diagram should save to plot_file (.png appended)"
    assert len(plt.get_fignums()) == n_before, "PIT figure must be closed (INV-50 leak fix)"


def test_build_pit_diagram_spec_is_a_histogram_with_ks_title():
    """INV-42: the orphan PIT now has a spec builder (single source, like the binary PIT panel)."""
    from mlframe.reporting.spec import FigureSpec, HistogramPanelSpec

    rng = np.random.default_rng(0)
    pit = rng.uniform(0.0, 1.0, size=400)
    spec = build_pit_diagram_spec(pit, bins=20)
    assert isinstance(spec, FigureSpec)
    panel = spec.panels[0][0]
    assert isinstance(panel, HistogramPanelSpec)
    assert "KS-vs-uniform=" in panel.title
    # Pre-binned density histogram: one height per bin.
    assert len(panel.values) == 20
    assert panel.bin_centers is not None and len(panel.bin_centers) == 20


def test_plot_pit_diagram_routes_through_spec_pipeline(tmp_path, monkeypatch):
    """INV-42: plot_pit_diagram must hand a HistogramPanelSpec FigureSpec to render_and_save,
    NOT draw a standalone plt.hist. On the pre-fix code it called plt.hist directly and never
    touched render_and_save, so this spy would never fire."""
    import mlframe.reporting.renderers as renderers
    from mlframe.reporting.spec import FigureSpec, HistogramPanelSpec

    captured = {}
    real = renderers.render_and_save

    def _spy(spec, output, base_path, **kwargs):
        """Returns ``real(spec, output, base_path, **kwargs)`` (after 1 setup step)."""
        captured["spec"] = spec
        return real(spec, output, base_path, **kwargs)

    monkeypatch.setattr("mlframe.reporting.renderers.render_and_save", _spy)

    rng = np.random.default_rng(1)
    probs = rng.uniform(0.0, 1.0, size=300)
    labels = (rng.uniform(size=300) < probs).astype(int)
    plot_pit_diagram(predicted_probs=probs, true_labels=labels, plot_file=str(tmp_path / "pit.png"))

    spec = captured.get("spec")
    assert isinstance(spec, FigureSpec), "PIT must route through render_and_save with a FigureSpec"
    assert isinstance(spec.panels[0][0], HistogramPanelSpec)


def test_plot_pit_diagram_multi_backend_dsl(tmp_path):
    """plot_outputs DSL renders both backends through the single spec pipeline."""
    import os

    rng = np.random.default_rng(2)
    probs = rng.uniform(0.0, 1.0, size=200)
    labels = (rng.uniform(size=200) < probs).astype(int)
    base = str(tmp_path / "pit")
    plot_pit_diagram(
        predicted_probs=probs,
        true_labels=labels,
        plot_file=base,
        plot_outputs="matplotlib[png] + plotly[html]",
    )
    assert os.path.exists(base + ".matplotlib.png")
    assert os.path.exists(base + ".plotly.html")


def test_show_plots_unless_agg_does_not_flip_interactive_mode():
    """On the Agg backend the helper must be a no-op and never enable interactive mode (INV-53)."""
    was_interactive = plt.isinteractive()
    fig = plt.figure()
    try:
        shown = _show_plots_unless_agg()
        # Agg backend -> not shown, and crucially interactive state unchanged.
        assert shown is False
        assert plt.isinteractive() == was_interactive, "_show_plots_unless_agg must not leak plt.ion() into the process (INV-53)"
    finally:
        plt.close(fig)
