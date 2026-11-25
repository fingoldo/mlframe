"""Cluster 41 regression: the ranker FS column set is built once per pipeline output (audit 2026-11-20, finding 41-F39).

``set(X_df.columns)`` used to be rebuilt inside the generator for every output column. It is now computed once into ``input_cols``.
"""

from pathlib import Path

from mlframe.training.ranking import _ranker_fs


def test_column_set_not_rebuilt_per_column():
    """The per-column membership test uses the hoisted set, not a fresh set() each time (41-F39)."""
    src = Path(_ranker_fs.__file__).resolve().read_text(encoding="utf-8")
    assert "if c in set(X_df.columns)" not in src
    assert "input_cols = set(X_df.columns)" in src
    assert "if c in input_cols" in src
