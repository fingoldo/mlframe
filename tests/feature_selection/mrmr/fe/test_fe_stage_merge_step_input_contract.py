"""Step-input contract: FE stage outputs are folded into the accumulated frame in stage order, and no stage sees another stage's columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl._fe_stage_merge import _fe_merge_new_columns


def _base():
    """A small base frame."""
    return pd.DataFrame({"a": np.arange(5.0), "b": np.arange(5.0) * 2})


def test_merge_appends_only_the_columns_a_stage_added():
    """The accumulated frame gains exactly the new columns of each stage, in stage order."""
    base = _base()
    s1 = base.assign(f1=1.0)
    s2 = base.assign(g1=2.0, g2=3.0)
    acc = _fe_merge_new_columns(base, s1, base)
    acc = _fe_merge_new_columns(acc, s2, base)
    assert list(acc.columns) == ["a", "b", "f1", "g1", "g2"]
    np.testing.assert_array_equal(acc["g2"].to_numpy(), np.full(5, 3.0))


def test_merge_is_identity_for_a_stage_that_added_nothing():
    """A no-op stage returns its input object and leaves the accumulator untouched."""
    base = _base()
    acc = base.assign(f1=1.0)
    assert _fe_merge_new_columns(acc, base, base) is acc


def test_merge_keeps_the_first_owner_of_a_duplicate_name():
    """Two stages emitting the same name keep the earlier stage's column."""
    base = _base()
    acc = _fe_merge_new_columns(base, base.assign(x=1.0), base)
    acc = _fe_merge_new_columns(acc, base.assign(x=9.0, y=2.0), base)
    assert list(acc.columns) == ["a", "b", "x", "y"]
    assert float(acc["x"].iloc[0]) == 1.0
