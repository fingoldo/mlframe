"""Cluster 20 regression: model-comparison bar colours use each model's own index, and the Spearman comment appears once (audit 2026-11-20, finding 20-F33).

Source-level pins: the O(K^2) names.index lookup is gone, and the duplicated comment block is reduced to one copy.
"""

from pathlib import Path

from mlframe.reporting.charts import model_comparison as mc

_SRC = Path(mc.__file__).resolve()


def test_bar_colours_use_the_loop_index():
    """Bar colours come from the model's own index, not a list search (20-F33)."""
    text = _SRC.read_text(encoding="utf-8")
    assert "names.index(names[i])" not in text
    assert "_MODEL_COLORS[i % len(_MODEL_COLORS)] for i in order_list" in text


def test_spearman_comment_appears_once():
    """The duplicated scale-pinning comment is reduced to a single copy (20-F33)."""
    text = _SRC.read_text(encoding="utf-8")
    assert text.count("pinned to that range rather than to the observed one") == 1
