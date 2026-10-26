"""The factorize-based group lookup gives the per-row mapping exactly, including NaN, None and mixed-type keys."""
import numpy as np

from mlframe.feature_selection.filters._grouped_agg_fe import _broadcast_lookup


def _per_row(keys, lookup, global_value):
    """The reference: str() and dict.get per row, NaN-scrubbed."""
    out = np.array([lookup.get(str(k), global_value) for k in keys], dtype=np.float64)
    return np.nan_to_num(out, nan=global_value, posinf=global_value, neginf=global_value)


def test_matches_the_per_row_mapping_on_awkward_keys():
    """Strings, a missing key, NaN, None and an int next to its string twin."""
    keys = np.array(["a", "b", "a", "c", np.nan, None, 1, "1", "zz"], dtype=object)
    lookup = {"a": 1.5, "b": -2.0, "c": float("nan"), "nan": 7.0, "None": 8.0, "1": 9.0}
    got = _broadcast_lookup(keys, lookup, -1.0)
    np.testing.assert_array_equal(got, _per_row(keys, lookup, -1.0))


def test_matches_on_a_large_low_cardinality_column():
    """20000 rows, 12 groups."""
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 12, 20000).astype(str).astype(object)
    lookup = {str(i): float(i) * 0.5 for i in range(10)}
    np.testing.assert_array_equal(_broadcast_lookup(keys, lookup, 3.0), _per_row(keys, lookup, 3.0))
