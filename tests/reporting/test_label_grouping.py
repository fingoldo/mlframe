"""``values[labels == lab]`` in a loop over labels is O(n x groups) for an O(n) answer.

The trap is the same one the multiclass subsample hit: the obvious replacement is SLOWER. A stable argsort
over string labels costs more than the masks it replaces -- measured at 2M rows and 4 splits, 1219ms
grouped against 711ms masked. Factorising to small integer codes first is what makes it a win, because
numpy radix-sorts narrow integers.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mlframe.reporting.charts._error_analysis_splits import _split_arrays
from mlframe.reporting.charts._grouping import group_indices_by_label, group_values_by_label


def _labelled(n: int = 100_000, k: int = 8, seed: int = 0):
    """A value array plus per-row string labels drawn from ``k`` distinct splits."""
    rng = np.random.default_rng(seed)
    names = np.array([f"split_{i}" for i in range(k)])
    return rng.normal(size=n), names[rng.integers(0, k, n)]


def _masked(values, labels):
    """The per-label boolean-mask grouping this replaces, kept as the identity oracle."""
    vals, labs = np.asarray(values), np.asarray(labels)
    return {str(lab): vals[labs == lab] for lab in dict.fromkeys(labs.tolist())}


@pytest.mark.parametrize("k", [1, 2, 8, 40])
def test_the_groups_match_the_masks_they_replace(k):
    """Same members, same groups, same order -- at one, two, several and many labels."""
    values, labels = _labelled(k=k)
    grouped, masked = group_values_by_label(values, labels), _masked(values, labels)
    assert list(grouped) == list(masked), f"the label order changed: {list(grouped)[:4]} vs {list(masked)[:4]}"
    for lab in masked:
        assert np.array_equal(grouped[lab], masked[lab]), f"group {lab!r} differs from its mask"


def test_rows_stay_in_ascending_order_within_a_group():
    """A boolean mask returns rows in index order, and callers index parallel arrays with the result."""
    _, labels = _labelled(n=5_000, k=4)
    order, bounds, uniq = group_indices_by_label(labels)
    for i in range(len(uniq)):
        rows = order[bounds[i] : bounds[i + 1]]
        assert np.all(np.diff(rows) > 0), f"group {uniq[i]!r} came back out of row order"


def test_first_appearance_order_is_preserved():
    """The splits panel labels its groups in the order they first occur; sorting them would relabel the chart."""
    labels = np.array(["zebra", "alpha", "zebra", "mid", "alpha"])
    assert list(group_values_by_label(np.arange(5.0), labels)) == ["zebra", "alpha", "mid"]


def test_a_single_label_is_one_group():
    """A degenerate split must not lose its rows."""
    values, labels = _labelled(n=1_000, k=1)
    grouped = group_values_by_label(values, labels)
    assert len(grouped) == 1 and np.array_equal(next(iter(grouped.values())), values)


def test_the_codes_are_narrowed_so_the_sort_can_be_a_radix_sort():
    """Without this the grouped form is slower than the masks; it is the whole reason this helper exists."""
    import mlframe.reporting.charts._grouping as grouping

    seen = {}
    real = np.argsort

    def spy(a, **kw):
        """Record the dtype of whatever gets sorted."""
        seen["dtype"] = np.asarray(a).dtype
        return real(a, **kw)

    grouping.np.argsort = spy
    try:
        group_indices_by_label(np.array([f"s{i % 5}" for i in range(1_000)]))
    finally:
        grouping.np.argsort = real
    assert seen.get("dtype") == np.int16, f"the sort key was {seen.get('dtype')}, not a radix-sortable narrow int"


def test_the_splits_helper_beats_the_masks_it_replaced():
    """Measured against the implementation it replaces, not against itself."""
    values, labels = _labelled(n=400_000, k=20)
    _split_arrays(values, labels)  # warm
    _masked(values, labels)

    def best(fn):
        """Best of three, which is what a paired comparison on a shared machine can defend."""
        return min((lambda t0: (fn(values, labels), time.perf_counter() - t0)[1])(time.perf_counter()) for _ in range(3))

    assert best(_split_arrays) < best(_masked), "the grouped split is not faster than the per-label masks"
