"""The powerset concat refuses a 2^n enumeration above ``max_subsets`` before it builds any composite column.

With ``max_order=None`` and 20 key columns the old code walked about 1M subsets, each a full-length string column. The cap is checked
from the subset count, so the refusal happens before the first composite is built.
"""

from __future__ import annotations

import pandas as pd
import pytest

import importlib

mod = importlib.import_module("mlframe.feature_engineering.categorical_powerset_concat")


def test_powerset_refuses_full_enumeration_over_cap_before_building_columns(monkeypatch):
    """With 20 key columns and default arguments the call raises ValueError without calling the concat helper."""
    built = []

    def _must_not_be_reached(*args, **kwargs):
        """Fail the test if the enumeration reaches the concat helper."""
        built.append(1)
        raise AssertionError("composite column built before the subset cap was checked")

    monkeypatch.setattr(mod, "concat_categorical_group", _must_not_be_reached)
    columns = [f"k{i}" for i in range(20)]
    df = pd.DataFrame({c: ["a", "b"] for c in columns})

    with pytest.raises(ValueError, match="max_subsets"):
        mod.categorical_powerset_concat(df, columns=columns)

    assert built == []
