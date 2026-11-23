"""A string source with no stored ``cat_code_map`` under ``unknown_strategy='clip'`` must not become a silent constant column (AUD23-09).

A recipe written before the code map existed carries none; ``clip`` then mapped every string cell to code 0 without a word, and the engineered column was constant at serving time. The values stay (``clip``
is the documented fallback), a warning is new; ``raise`` still raises.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.feature_selection.filters.engineered_recipes._recipe_extract import _coerce_to_int_with_nan_handling


def _call(strategy: str):
    """The coercion of a string column with no map, the way a legacy recipe reaches it."""
    vals = np.array(["a", "b", "a", "c"], dtype=object)
    return _coerce_to_int_with_nan_handling(vals, 4, "legacy", "cat", strategy)


def test_clip_returns_zeros_and_warns_naming_recipe_and_column(caplog):
    """Constant zeros are the documented fallback; the warning names the recipe and the column."""
    with caplog.at_level(logging.WARNING):
        out = _call("clip")
    assert out.tolist() == [0, 0, 0, 0]
    assert any(r.levelno >= logging.WARNING and "legacy" in r.getMessage() and "cat" in r.getMessage() for r in caplog.records)


def test_raise_still_raises():
    """Contrast: the strict strategy keeps its error."""
    with pytest.raises(ValueError, match="cannot be"):
        _call("raise")
