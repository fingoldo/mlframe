"""A recipe that reads a raw input column absent at transform time raises instead of emitting a silent zero column."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from mlframe.feature_selection.filters._mrmr_validate_transform import _replay_recipes_chained


def _recipe():
    """A chained-capable recipe reading column 'b'."""
    return SimpleNamespace(name="eng", kind="numeric_decompose", src_names=("b",), extra={})


def test_missing_raw_column_raises_a_keyerror_naming_it():
    """'b' was a fit-time input column but is not in X: the replay must fail loudly."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(KeyError, match="'b'"):
        _replay_recipes_chained([_recipe()], X, {}, {}, raw_names=["a", "b"])


def test_missing_pruned_engineered_source_still_gives_a_zero_column():
    """'b' is not a raw fit-time column (its engineered producer was pruned): the documented neutral zero column stays."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    cols = _replay_recipes_chained([_recipe()], X, {}, {}, raw_names=["a"])
    assert list(cols[0]) == [0.0, 0.0, 0.0]
