"""``_unique_inverse`` on string keys gives what ``np.unique`` gives."""

import numpy as np

from mlframe.feature_selection.filters._composite_group_agg_fe import _unique_inverse


def test_string_keys_match_numpy_unique():
    """Sorted uniques and the inverse agree, with ties, a unicode key, an empty string and a size above the fast-path threshold."""
    rng = np.random.default_rng(1)
    pool = np.array(["a", "b|c", "10", "9", "", "é", "ab", "B"] + [f"k{i}" for i in range(300)], dtype=object)
    keys = pool[rng.integers(0, pool.size, 50000)]
    got_u, got_i = _unique_inverse(keys)
    want_u, want_i = np.unique(keys, return_inverse=True)
    np.testing.assert_array_equal(np.asarray(got_u, dtype=object), np.asarray(want_u, dtype=object))
    np.testing.assert_array_equal(got_i, np.asarray(want_i).reshape(-1))


def test_unorderable_keys_still_return_none():
    """A mixed-type object array keeps the (None, None) answer."""
    keys = np.array(["a", 1, None] * 600, dtype=object)
    assert _unique_inverse(keys) == (None, None)
