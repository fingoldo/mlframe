"""The lineage of an appended frame: recorded for a pandas child, forgotten with it, and said out loud when it cannot be kept."""

import gc
import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _fe_frame_ops as ops


@pytest.fixture(autouse=True)
def _fresh_lineage_table(monkeypatch):
    """Every test starts from an empty lineage table, so an entry recorded here cannot reach another test."""
    monkeypatch.setattr(ops, "_APPENDED_FROM", {})


def _parent() -> pd.DataFrame:
    """A small pandas frame to append columns to."""
    return pd.DataFrame({"a": np.arange(5, dtype=np.float32), "b": np.arange(5, dtype=np.float32) * 2})


def test_an_appended_frame_names_its_parent_and_its_new_columns():
    """The child of ``fe_append_columns`` reports the parent it came from and exactly the columns appended."""
    parent = _parent()
    child = ops.fe_append_columns(parent, {"c": np.ones(5, dtype=np.float32)})
    found = ops.appended_from(child)
    assert found is not None
    assert found[0] is parent
    assert found[1] == ("c",)


def test_the_lineage_goes_with_the_child():
    """When the child frame is collected its entry leaves the table, so the table does not hold lineage of frames that no longer exist."""
    parent = _parent()
    child = ops.fe_append_columns(parent, {"c": np.ones(5, dtype=np.float32)})
    assert len(ops._APPENDED_FROM) == 1
    del child
    gc.collect()
    assert len(ops._APPENDED_FROM) == 0


def test_an_object_that_cannot_be_weakly_referenced_keeps_no_lineage_and_says_why(caplog):
    """A child that is not weakly referenceable records nothing, and the debug log names its type."""
    with caplog.at_level(logging.DEBUG, logger=ops.logger.name):
        ops._note_appended(7, _parent(), ["c"])
    assert len(ops._APPENDED_FROM) == 0
    assert ops.appended_from(7) is None
    assert any("cannot be weakly referenced" in r.getMessage() and "int" in r.getMessage() for r in caplog.records)
