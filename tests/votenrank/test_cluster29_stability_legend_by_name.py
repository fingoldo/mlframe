"""Cluster 29 regression: the stability plot legend is relabelled by method name, not by position (audit 2026-11-20, finding 29-F36).

Positional indexing into the legend texts breaks when the number of plotted methods changes. Each legend entry must be relabelled by its own
label, whatever its position. The test renders the figure, reads the legend it produced, and checks the labels.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import mlframe.metrics.shared as metrics_shared
from mlframe.votenrank import stability_exp

_RANGE = [0, 5, 10, 15, 20]
_OG = "$\\sigma^{og}$"
_AM = "$\\sigma^{am}$"


def _legend_texts_after_render(monkeypatch, exp_res: dict) -> list[str]:
    """Render the stability figure for ``exp_res`` and return the legend label texts as they were before the figure closed."""
    captured: dict[str, list[str]] = {}
    real_close = metrics_shared.close_unless_interactive

    def _capture_then_close(fig, was_shown=False):
        """Record the legend texts on the live axes, then close as the real helper does."""
        legend = plt.gca().get_legend()
        captured["texts"] = [t.get_text() for t in legend.get_texts()]
        real_close(fig, was_shown=was_shown)

    monkeypatch.setattr(metrics_shared, "close_unless_interactive", _capture_then_close)
    try:
        stability_exp.create_exp_pic(_RANGE, exp_res)
    finally:
        plt.close("all")
    return captured["texts"]


def test_legend_entries_are_relabelled_by_name_whatever_their_order(monkeypatch):
    """With the mean method first, its entry is relabelled sigma-am and the gap entry sigma-og, following each label (29-F36)."""
    exp_res = {
        "mean, Params: {}": [0.8, 0.7, 0.6, 0.5, 0.4],
        "optimality_gap, Params: {}": [0.9, 0.8, 0.7, 0.6, 0.5],
    }
    assert _legend_texts_after_render(monkeypatch, exp_res) == [_AM, _OG]


def test_unrelated_legend_label_keeps_its_text(monkeypatch):
    """Control: an entry matching neither label is left as the method name (29-F36)."""
    exp_res = {
        "other_method, Params: {}": [0.3, 0.3, 0.3, 0.3, 0.3],
        "optimality_gap, Params: {}": [0.9, 0.8, 0.7, 0.6, 0.5],
    }
    texts = _legend_texts_after_render(monkeypatch, exp_res)
    assert texts[0].startswith("other_method")
    assert texts[1] == _OG
