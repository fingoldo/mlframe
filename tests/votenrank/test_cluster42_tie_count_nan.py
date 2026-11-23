"""Cluster 42 regression: the leaderboard tie count keeps NaN ranks (audit 2026-11-20, finding 42-F13).

nunique() drops NaN, so NaN rows were understated in the tie count. Pinned by source: both sites use nunique(dropna=False), and a
pandas check shows the difference the fix makes.
"""

from pathlib import Path

import pandas as pd

from mlframe.votenrank.leaderboard import leaderboard_impl as li

_SRC = Path(li.__file__).resolve()


def test_tie_count_counts_nan_ranks():
    """With two NaN ranks the count reflects them; the old nunique() understated it (42-F13)."""
    ranking = pd.Series([1.0, float("nan"), float("nan")], index=["a", "b", "c"])
    assert ranking.shape[0] - ranking.nunique() == 2
    assert ranking.shape[0] - ranking.nunique(dropna=False) == 1


def test_source_uses_dropna_false_at_both_sites():
    """Both tie-count sites in the module pass dropna=False (42-F13)."""
    text = _SRC.read_text(encoding="utf-8")
    assert text.count("ranking.nunique(dropna=False)") == 2
    assert "ranking.shape[0] - ranking.nunique()\n" not in text
