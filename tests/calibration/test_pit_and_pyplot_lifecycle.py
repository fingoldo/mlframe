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

from mlframe.calibration.quality import plot_pit_diagram
from mlframe.metrics.calibration import _show_plots_unless_agg


def test_plot_pit_diagram_saves_and_closes(tmp_path):
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.0, 1.0, size=500)
    labels = (rng.uniform(size=500) < probs).astype(int)
    out = str(tmp_path / "pit")  # extension-less on purpose
    n_before = len(plt.get_fignums())
    plot_pit_diagram(predicted_probs=probs, true_labels=labels, plot_file=out)
    import os
    assert os.path.exists(out + ".png"), "PIT diagram should save to plot_file (.png appended)"
    assert len(plt.get_fignums()) == n_before, "PIT figure must be closed (INV-50 leak fix)"


def test_show_plots_unless_agg_does_not_flip_interactive_mode():
    """On the Agg backend the helper must be a no-op and never enable interactive mode (INV-53)."""
    was_interactive = plt.isinteractive()
    fig = plt.figure()
    try:
        shown = _show_plots_unless_agg()
        # Agg backend -> not shown, and crucially interactive state unchanged.
        assert shown is False
        assert plt.isinteractive() == was_interactive, (
            "_show_plots_unless_agg must not leak plt.ion() into the process (INV-53)"
        )
    finally:
        plt.close(fig)
