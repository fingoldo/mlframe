"""The missingness count and pattern replay warn when a fit-time column is absent at transform time (AUD23-57).

The documented contract keeps the output (an absent column contributes 0 / not-missing, "graceful schema drift"), but it returned a plausible wrong value with nothing to say so. The values stay; the warning is new.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _missingness_fe as M


def _frame() -> pd.DataFrame:
    """Three columns with a few NaN."""
    return pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan], "b": [np.nan, 2.0, np.nan, 4.0], "c": [1.0, 2.0, 3.0, 4.0]})


def test_count_warns_names_the_absent_column_and_keeps_the_value(caplog):
    """Fit on a, b, c; transform without b: the count of a alone, and a WARNING that names b."""
    X = _frame()
    recipe = {"cols": ("a", "b", "c")}
    with caplog.at_level(logging.WARNING):
        out = M.apply_missingness_count(X.drop(columns=["b"]), recipe)
    assert out.tolist() == [0, 1, 0, 1]
    assert any(r.levelno >= logging.WARNING and "'b'" in r.getMessage() for r in caplog.records)


def test_count_is_silent_when_every_column_is_present(caplog):
    """Contrast: no absent column, no warning."""
    with caplog.at_level(logging.WARNING):
        M.apply_missingness_count(_frame(), {"cols": ("a", "b", "c")})
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING and "absent" in r.getMessage()]


def test_pattern_warns_for_an_absent_column(caplog):
    """The pattern replay names the absent column too, and still returns one label per row."""
    X = _frame()
    recipe = {"cols": ("a", "b"), "pattern_to_label": {"0": 0, "1": 1, "2": 2, "3": 3}, "other_label": 9}
    with caplog.at_level(logging.WARNING):
        out = M.apply_missingness_pattern(X.drop(columns=["b"]), recipe)
    assert len(out) == len(X)
    assert any(r.levelno >= logging.WARNING and "'b'" in r.getMessage() for r in caplog.records)
