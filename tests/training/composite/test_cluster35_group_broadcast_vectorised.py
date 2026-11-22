"""Audit 2026-11-20 finding 35-F31: the group broadcast is one vectorised reindex, and NaN group ids are reported.

The per-row dict loop was O(n) Python work per statistic. A NaN group id used to take the fallback with no log line.
"""

import logging

import numpy as np

from mlframe.training.composite.group_aggregate_macro import _broadcast_group_values


def test_broadcast_matches_the_per_row_lookup():
    """Each row receives its own group's value; unknown groups get the fallback."""
    groups = np.array(["a", "b", "a", "c", "b"], dtype=object)
    unique = np.array(["a", "b"], dtype=object)
    values = np.array([1.0, 2.0])
    out = _broadcast_group_values(groups, unique, values, fallback=9.0)
    np.testing.assert_allclose(out, [1.0, 2.0, 1.0, 9.0, 2.0])


def test_nan_group_rows_take_the_fallback_and_are_logged(caplog):
    """A NaN group id gets the fallback and a WARNING reports how many rows did."""
    groups = np.array([1.0, np.nan, 2.0, np.nan], dtype=float)
    unique = np.array([1.0, 2.0])
    values = np.array([10.0, 20.0])
    with caplog.at_level(logging.WARNING):
        out = _broadcast_group_values(groups, unique, values, fallback=0.5)
    np.testing.assert_allclose(out, [10.0, 0.5, 20.0, 0.5])
    assert any("2 row(s) have a NaN group id" in r.getMessage() for r in caplog.records)
