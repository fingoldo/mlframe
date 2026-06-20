"""Regression: njit ``digitize`` must clamp out-of-range high values to the top bin.

Pre-fix the inner ``for j, bin_edge`` loop never assigned ``res[i]`` when
``val > bins[-1]`` (no edge satisfied ``val <= bin_edge``), so the result row
kept its uninitialised ``np.empty`` value -- nondeterministic garbage codes.
This bites at transform time when a val/test row exceeds the fit-time max.
The sibling ``quantize_search`` routes such values to the top bin; ``digitize``
must agree.
"""
import numpy as np

from mlframe.feature_selection.filters.discretization import digitize, discretize_array, quantize_search


def test_digitize_value_above_last_edge_clamps_to_top_bin():
    bins = np.array([0.0, 1.0, 2.0])
    arr = np.array([5.0, -5.0, 0.5], dtype=np.float64)
    out = digitize(arr, bins)
    # val=5.0 > bins[-1]=2.0 -> top bin index len(bins)-1 == 2 (not garbage).
    assert out[0] == len(bins) - 1
    # below all edges -> bin 0; in-range -> first edge >= val.
    assert out[1] == 0
    assert out[2] == 1


def test_digitize_all_above_is_deterministic_across_calls():
    bins = np.array([0.0, 1.0, 2.0])
    first = digitize(np.array([9.0, 9.0, 9.0]), bins).copy()
    second = digitize(np.array([9.0, 9.0, 9.0]), bins).copy()
    assert np.array_equal(first, second)
    assert np.all(first == len(bins) - 1)


def test_digitize_max_value_not_dropped():
    # The column max equals the last edge -> must land in the last bin, not be
    # left uninitialised by an off-by-one in the edge scan.
    bins = np.array([0.0, 1.0, 2.0])
    out = digitize(np.array([2.0]), bins)
    assert out[0] == len(bins) - 1


def test_discretize_array_empty_consistent_between_methods():
    # Empty input: quantile path used to raise IndexError from
    # nanpercentile([]) while the uniform sibling returned []. Both must now
    # return an empty array so the degenerate-input contract is consistent.
    empty = np.array([], dtype=np.float64)
    out_q = discretize_array(empty, n_bins=10, method="quantile")
    out_u = discretize_array(empty, n_bins=10, method="uniform")
    assert out_q.shape == (0,)
    assert out_u.shape == (0,)
