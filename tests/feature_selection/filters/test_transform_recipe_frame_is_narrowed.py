"""transform() must not deep-copy the whole caller frame to replay recipes (mrmr_audit_2026-09-14 PERIPHERY-2).

The replay loop needs a private frame it can append engineered columns onto; it used to take that ownership
by deep-copying ALL of X on every call. Frames here reach 100+ GB and the loop only ever reads single named
columns, so columns no recipe can name were copied for nothing.

``copy(deep=False)`` is deliberately NOT the fix -- this repo has already shipped a bug where a shallow copy
shared the source BlockManager and a later setitem promoted the shared block, leaking the new column back
onto the caller's frame. So the ownership guarantee is pinned here too, not just the narrowing.
"""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._mrmr_validate_transform import _recipe_reachable_columns


class _Recipe:
    """Minimal recipe stand-in carrying just the attributes the reachability walk reads."""

    def __init__(self, name, src_names, extra=None):
        self.name = name
        self.src_names = tuple(src_names)
        self.extra = extra or {}


@pytest.fixture
def wide_frame():
    """A frame far wider than the handful of columns any recipe references."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({f"c{i}": rng.normal(size=32) for i in range(50)})


def test_only_referenced_columns_are_kept(wide_frame):
    """A recipe naming two columns must not drag the other 48 into the copy."""
    got = _recipe_reachable_columns(wide_frame, [_Recipe("eng", ["c3", "c7"])])
    assert got == ["c3", "c7"]


def test_the_frames_own_column_order_is_preserved(wide_frame):
    """Order must follow X, not the recipes' declaration order, so positional assumptions hold."""
    got = _recipe_reachable_columns(wide_frame, [_Recipe("eng", ["c9", "c2"])])
    assert got == ["c2", "c9"]


def test_nested_parent_sources_are_followed(wide_frame):
    """A nested parent carries its own sources; missing them would KeyError mid-replay."""
    parent = _Recipe("parent", ["c11"])
    child = _Recipe("child", ["c1"], extra={"nested_parent_a": parent})
    assert _recipe_reachable_columns(wide_frame, [child]) == ["c1", "c11"]


def test_a_column_named_only_in_extra_is_still_kept(wide_frame):
    """Safety superset: a kind reaching for a group key outside src_names must stay covered."""
    r = _Recipe("agg", ["c4"], extra={"group_col": "c40"})
    assert _recipe_reachable_columns(wide_frame, [r]) == ["c4", "c40"]


def test_an_extra_string_that_is_not_a_column_is_ignored(wide_frame):
    """Only strings that actually name a column count -- an operator name must not widen the set."""
    r = _Recipe("op", ["c4"], extra={"binary": "multiply", "unary_a": "log"})
    assert _recipe_reachable_columns(wide_frame, [r]) == ["c4"]


def test_no_recipes_or_no_match_falls_back_to_the_whole_frame(wide_frame):
    """Never hand the replay loop an empty substrate; degrade to the old behaviour instead."""
    assert _recipe_reachable_columns(wide_frame, []) == list(wide_frame.columns)
    assert _recipe_reachable_columns(wide_frame, [_Recipe("x", ["absent"])]) == list(wide_frame.columns)


def test_a_cyclic_recipe_chain_terminates(wide_frame):
    """A malformed self-referential chain must not recurse forever."""
    a = _Recipe("a", ["c1"])
    a.extra = {"nested_parent_a": a}
    assert _recipe_reachable_columns(wide_frame, [a]) == ["c1"]


def test_appending_to_the_narrowed_copy_never_touches_the_caller_frame(wide_frame):
    """The ownership guarantee the deep copy existed for, pinned explicitly."""
    before = wide_frame.copy(deep=True)
    narrowed = wide_frame[_recipe_reachable_columns(wide_frame, [_Recipe("eng", ["c3"])])].copy()
    narrowed["eng"] = np.arange(len(wide_frame), dtype=np.float64)
    assert "eng" not in wide_frame.columns
    pd.testing.assert_frame_equal(wide_frame, before)
