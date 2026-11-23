"""Cluster 21 regression: split names that normalise to the same key are refused, not silently merged (audit 2026-11-20, finding 21-F27).

"Train" and "train" used to overwrite each other in the canonical map. The collision now raises ValueError naming both keys; distinct
names and the lower-case path still work.
"""

import pytest

from mlframe.reporting.charts import split_comparison as sc


def test_case_collision_is_refused():
    """'Train' and 'train' in one map raise, naming both keys (21-F27)."""
    with pytest.raises(ValueError, match="same split"):
        sc._canonical_split_names({"Train": {"a": 1}, "train": {"a": 2}})


def test_distinct_names_are_normalised_and_kept():
    """Control: distinct splits are normalised to lower case and all kept (21-F27)."""
    out = sc._canonical_split_names({" Train ": {"a": 1}, "Test": {"a": 2}, "val": {"a": 3}})
    assert set(out) == {"train", "test", "val"}
