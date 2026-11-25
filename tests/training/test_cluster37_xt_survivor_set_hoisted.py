"""Cluster 37 regression: the XT-ensemble survivor set is built once (audit 2026-11-20, finding 37-F38).

The keep-mask comprehension rebuilt ``set(_survivors)`` for every OOF name. The set is now built once before the comprehension.
"""

from pathlib import Path

from mlframe.training.core import _phase_composite_post_xt_ensemble as xt


def test_survivor_set_is_built_once():
    """The keep-mask comprehension tests against a prebuilt set (37-F38)."""
    src = Path(xt.__file__).resolve().read_text(encoding="utf-8")
    assert "for n in st._oof_names], dtype=bool)" in src
    assert "n in set(_survivors) for n in st._oof_names" not in src
    assert "_survivor_set = set(_survivors)" in src
    assert "n in _survivor_set for n in st._oof_names" in src
