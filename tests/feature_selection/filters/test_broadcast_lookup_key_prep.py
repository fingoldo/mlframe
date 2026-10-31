"""``broadcast_lookup`` equals the per-row mapping on repeated calls with one key array, and a new array is prepared on its own."""

import numpy as np

from mlframe.feature_selection.filters._grouped_coerce_shared import broadcast_lookup


def _per_row(keys, lookup, glob):
    """Reference: map every row through ``str(key)`` with the unseen default, then scrub non-finite values."""
    out = np.array([lookup.get(str(k), glob) for k in keys], dtype=np.float64)
    return np.nan_to_num(out, nan=glob, posinf=glob, neginf=glob)


def test_repeated_calls_and_missing_keys_match_the_per_row_mapping():
    """Several lookups on the same array, None and NaN keys, an unseen key, and a lookup holding a NaN value."""
    keys = np.array(["a", "b", None, "a", float("nan"), "zz", "b"], dtype=object)
    lookups = [{"a": 1.0, "b": 2.0, "None": 3.0, "nan": 4.0}, {"a": 5.0, "b": float("nan")}, {}]
    for lookup in lookups * 2:
        np.testing.assert_allclose(broadcast_lookup(keys, lookup, 0.5), _per_row(keys, lookup, 0.5), rtol=1e-6)


def test_a_different_array_is_not_confused_with_the_previous_one():
    """Two arrays of the same length and dtype get their own codes."""
    first = np.array(["a", "b", "a", "b"], dtype=object)
    second = np.array(["b", "b", "a", "a"], dtype=object)
    lookup = {"a": 1.0, "b": 2.0}
    assert broadcast_lookup(first, lookup, 0.0).tolist() == [1.0, 2.0, 1.0, 2.0]
    assert broadcast_lookup(second, lookup, 0.0).tolist() == [2.0, 2.0, 1.0, 1.0]
