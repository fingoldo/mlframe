"""Cluster 12 regression: the confirm bench labels its non-regression bar as a one-sided floor and transforms the test split once (audit 2026-11-20, finding 12-F96).

Source-level pins, since the bench needs real data: the printed bar says "one-sided floor" for the non-regression beds, and the test
transform is computed once and reused for the AUC and the width.
"""

from pathlib import Path

_BENCH = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "_benchmarks" / "fs_hybrid" / "round4_hybrid_confirm.py"


def test_non_regression_bar_is_labelled_one_sided():
    """The synth/hard_synth bar is printed as a one-sided floor, not as a +/-0.005 band (12-F96)."""
    text = _BENCH.read_text(encoding="utf-8")
    assert "(one-sided floor)" in text
    assert "+/-0.005" not in text


def test_test_split_is_transformed_once():
    """The test transform is reused for the AUC and the width (12-F96)."""
    text = _BENCH.read_text(encoding="utf-8")
    assert "Xte_t = h.transform(Xte)" in text
    assert "h.transform(Xte).shape" not in text
