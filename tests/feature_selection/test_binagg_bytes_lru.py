"""The binned-agg build's column caches stay under their byte budget, keep the newest entry and refresh an entry on access."""

import numpy as np

from mlframe.feature_selection.filters._binned_numeric_agg_resident import _BytesLRU, _value_bytes


def test_evicts_the_least_recently_used_entries_over_budget():
    """Three 800-byte entries under a 2000-byte budget keep the two newest."""
    c = _BytesLRU(2000)
    for k in "abc":
        c[k] = np.zeros(100)
    assert list(c) == ["b", "c"]


def test_access_refreshes_an_entry():
    """Reading 'a' before inserting 'c' makes 'b' the one evicted."""
    c = _BytesLRU(2000)
    c["a"] = np.zeros(100)
    c["b"] = np.zeros(100)
    assert c.get("a") is not None
    c["c"] = np.zeros(100)
    assert list(c) == ["a", "c"]


def test_the_newest_entry_stays_even_when_alone_over_budget():
    """An entry larger than the budget is kept (the build needs it), the older ones go."""
    c = _BytesLRU(100)
    c["small"] = np.zeros(5)
    c["big"] = np.zeros(1000)
    assert list(c) == ["big"]


def test_value_bytes_counts_nested_arrays():
    """Tuples and dicts of arrays are summed; plain numbers count zero."""
    assert _value_bytes((np.zeros(10), {"t": np.zeros(5)}, 3)) == 15 * 8
