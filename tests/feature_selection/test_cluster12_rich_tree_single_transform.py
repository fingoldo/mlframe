"""Cluster 12 regression: the rich-tree bench transforms the test split once per fit (audit 2026-11-20, finding 12-F93).

Source-level pin, since the bench needs real data: the test transform is computed into one name and reused for the AUC and the width,
not recomputed for each use.
"""

from pathlib import Path

_BENCH = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "_benchmarks" / "fs_hybrid" / "round4_rich_tree_bench.py"


def test_test_split_is_transformed_once():
    """The test transform is reused; the width is read from it, not from a second h.transform call (12-F93)."""
    text = _BENCH.read_text(encoding="utf-8")
    assert "Xte_t = h.transform(Xte)" in text
    assert "n=int(Xte_t.shape[1])" in text
    assert "h.transform(Xte).shape" not in text
