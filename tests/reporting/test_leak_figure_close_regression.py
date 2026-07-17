"""Regression sensors: chart helpers must not leak matplotlib figures into the global registry.

LEAK1 (``diagnostics_dispatch._save_figure``): ``plt.close(fig)`` lived inside the ``if "png" in ...`` block, so an
early non-png return -- or a savefig failure -- left the figure open. The fix moves the close into a ``finally`` so it
runs on every exit path. Pre-fix, the non-png path grows ``plt.get_fignums()`` by one per call.

LEAK3 (``shap_per_instance._render``): the small-multiples figure was saved/returned but never closed (no
``figs_before``/``_close_figs`` guard like its sibling ``shap_panels``). Pre-fix, every call leaks one figure.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlframe.reporting.diagnostics_dispatch import _save_figure
from mlframe.reporting.charts.shap_per_instance import _render


def _open_fig_count() -> int:
    return len(plt.get_fignums())


def test_save_figure_closes_on_non_png_path():
    """Non-png ``plot_outputs`` returns early WITHOUT a savefig but must still close the handed-in figure."""
    plt.close("all")
    before = _open_fig_count()
    fig = plt.figure()
    assert _open_fig_count() == before + 1
    # "svg" (no png) -> the function returns False before any savefig; the figure must not leak.
    ok = _save_figure(fig, plot_outputs="svg", base_path="unused")
    assert ok is False
    assert _open_fig_count() == before, "non-png path leaked a figure"


def test_save_figure_closes_on_savefig_failure(tmp_path):
    """A savefig failure path must also close the figure (it lives in the same png branch)."""
    plt.close("all")
    before = _open_fig_count()
    fig = plt.figure()

    class _Boom:
        def savefig(self, *a, **k):
            raise RuntimeError("disk full")

    # Wrap so close still targets the real figure: monkeypatch savefig on the real fig instead.
    orig = fig.savefig
    fig.savefig = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full"))  # type: ignore[assignment]
    try:
        ok = _save_figure(fig, plot_outputs="png", base_path=str(tmp_path / "x"))
    finally:
        fig.savefig = orig  # type: ignore[assignment]
    assert ok is False
    assert _open_fig_count() == before, "savefig-failure path leaked a figure"


def test_save_figure_closes_on_success(tmp_path):
    plt.close("all")
    before = _open_fig_count()
    fig = plt.figure()
    ok = _save_figure(fig, plot_outputs="png", base_path=str(tmp_path / "ok"))
    assert ok is True
    assert _open_fig_count() == before, "success path leaked a figure"


def test_shap_per_instance_render_closes_figure():
    """``_render`` saves+returns a figure; it must close every figure it opened, so the registry does not grow."""
    plt.close("all")
    before = _open_fig_count()
    k = 3
    worst_idx = np.arange(k)
    severities = np.array([0.9, 0.8, 0.7])
    contributions = [[("f0", 0.5), ("f1", -0.3)] for _ in range(k)]
    y_true = np.array([1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.2])

    fig, paths = _render(
        worst_idx=worst_idx,
        severities=severities,
        contributions=contributions,
        y_true=y_true,
        y_score=y_score,
        binary=True,
        no_errors=False,
        plot_file=None,
        plot_outputs=None,
    )
    assert fig is not None
    assert paths == []
    assert _open_fig_count() == before, "_render leaked a figure into the registry"
