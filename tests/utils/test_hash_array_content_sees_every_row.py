"""Cache keys for results that depend on every row must change when any row does."""

import numpy as np

from mlframe.utils.disk_cache import hash_array_content, hash_array_summary


def test_a_middle_row_change_changes_the_content_hash():
    """The summary hash reads head/tail rows plus sum/min/max, so a swap in the middle leaves it unchanged."""
    a = np.arange(1000, dtype=np.float64)
    b = a.copy()
    b[400], b[600] = b[600], b[400]  # same head, tail, sum, min and max
    assert hash_array_summary(a) == hash_array_summary(b), "precondition: the summary cannot see this change"
    assert hash_array_content(a) != hash_array_content(b)


def test_equal_arrays_share_a_key_across_dtype_views():
    a = np.arange(10, dtype=np.float64)
    assert hash_array_content(a) == hash_array_content(a.copy())
    assert hash_array_content(a) != hash_array_content(a.astype(np.float32)), "dtype is part of the identity"


def test_per_feature_edges_keys_on_the_full_column(tmp_path):
    """Supervised edges depend on the whole column-to-y pairing; a middle-row swap must not replay cached edges."""
    from mlframe.feature_selection.filters._adaptive_nbins import per_feature_edges

    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 1))
    y = (x[:, 0] > 0).astype(np.int64)
    y_swapped = np.concatenate([y[:500], y[1000:1500], y[500:1000], y[1500:]])  # same head, tail, sum, min, max
    assert hash_array_summary(y) == hash_array_summary(y_swapped), "precondition: the summary cannot see this change"
    first = per_feature_edges(x, y=y, method="fayyad_irani", cache_dir=str(tmp_path))
    second = per_feature_edges(x, y=y_swapped, method="fayyad_irani", cache_dir=str(tmp_path))
    fresh = per_feature_edges(x, y=y_swapped, method="fayyad_irani")
    assert [e.tolist() for e in second] == [e.tolist() for e in fresh], "the second call replayed the first call's edges"
    assert first is not None
