"""A continuous target with few distinct NON-integer values is class-encoded, not truncated (AUD23-60).

``astype(np.int64)`` turned 0.1 .. 0.9 into 0: one class, no split, one bin, no warning. The fix keeps the distinct values apart (dense codes) and leaves integer-valued targets as they were.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._mdlp_validated_split import mdlp_bin_edges_validated
from mlframe.feature_selection.filters.supervised_binning import integer_class_codes, mdlp_bin_edges


def _data(seed: int = 0, n: int = 4000):
    """A feature and nine float levels in 0.1 .. 0.9 that follow it (the target has structure in x)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    levels = np.linspace(0.1, 0.9, 9)
    y = levels[np.clip(np.digitize(x, np.quantile(x, np.linspace(0, 1, 10)[1:-1])), 0, 8)]
    return x, y


def test_integer_class_codes_keeps_distinct_float_levels_apart():
    """0.1 .. 0.9 become nine distinct dense codes; the ordering of the levels is kept."""
    levels = np.linspace(0.1, 0.9, 9)
    codes = integer_class_codes(levels)
    assert len(np.unique(codes)) == 9
    assert np.array_equal(codes, np.arange(9))


def test_integer_valued_targets_are_unchanged():
    """An integer target, and a float target whose values are integers, come back as the same int64 values (no re-coding)."""
    ints = np.array([0, 3, 3, 7, 9], dtype=np.int64)
    assert np.array_equal(integer_class_codes(ints), ints)
    floats = np.array([1.0, 2.0, 2.0, 5.0])
    assert np.array_equal(integer_class_codes(floats), floats.astype(np.int64))


def test_nan_stays_outside_the_codes_of_a_float_target():
    """Non-finite entries are not given a class: the caller masks them; the codes of the finite entries are dense."""
    y = np.array([0.1, np.nan, 0.5, 0.9])
    codes = integer_class_codes(y)
    assert sorted(set(codes[[0, 2, 3]].tolist())) == [0, 1, 2]


@pytest.mark.parametrize("fn", [mdlp_bin_edges, mdlp_bin_edges_validated], ids=["default", "validated"])
def test_a_few_float_levels_give_real_bins(fn):
    """The default and the validated MDLP split a continuous target with nine float levels into bins, as they do for the same levels coded as integers."""
    x, y = _data()
    edges_float = np.asarray(fn(x, y), dtype=np.float64)
    edges_int = np.asarray(fn(x, np.round(y * 10).astype(np.int64)), dtype=np.float64)
    assert len(edges_float) > 2, f"one bin only: edges {edges_float}"
    assert len(edges_float) == len(edges_int), "float levels must give as many edges as the same levels coded as integers"
