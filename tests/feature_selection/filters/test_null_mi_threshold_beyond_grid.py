"""The null-MI floor keeps growing with the candidate count past the largest tabulated pool (tracker 2026-10-14, P-F04)."""

import math

import pytest

from mlframe.feature_selection.filters import _null_mi_threshold as nm

pytestmark = pytest.mark.skipif(not nm._entries(), reason="null MI table not present")


def _max_pool(entries):
    """The largest tabulated candidate count and one entry that carries it."""
    top = max(e["candidates"] for e in entries)
    return top, next(e for e in entries if e["candidates"] == top)


def test_a_pool_beyond_the_grid_gets_a_proportionally_larger_floor():
    """For a fingerprint matching the largest entry exactly, 4x the pool scales the floor by the ratio of the logs."""
    entries = nm._entries()
    top, e = _max_pool(entries)
    at_grid = nm.null_mi_threshold(e["n"], e["bins"], e["entropy"], top)
    beyond = nm.null_mi_threshold(e["n"], e["bins"], e["entropy"], 4 * top)
    assert at_grid == pytest.approx(e["threshold"])
    assert beyond == pytest.approx(e["threshold"] * math.log(4 * top) / math.log(top))
    assert beyond > at_grid


def test_a_pool_inside_the_grid_is_unchanged():
    """Inside the tabulated range the lookup still returns the nearest entry's floor, with no rescaling."""
    entries = nm._entries()
    e = entries[0]
    assert nm.null_mi_threshold(e["n"], e["bins"], e["entropy"], e["candidates"]) == pytest.approx(e["threshold"])
