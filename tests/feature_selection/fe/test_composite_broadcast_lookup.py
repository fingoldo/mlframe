"""The composite broadcast replaces non-finite group values before the gather and still equals the per-row replacement."""

import numpy as np

from mlframe.feature_selection.filters._composite_group_agg_fe import _broadcast_lookup


def test_gather_then_replace_equals_replace_then_gather():
    """NaN / inf group values and unseen keys all end as the global value, in the requested dtype."""
    rng = np.random.default_rng(0)
    keys = rng.choice(["a", "b", "c", "d", "unseen"], size=500).astype(object)
    lookup = {"a": 1.5, "b": float("nan"), "c": float("inf"), "d": -float("inf")}
    got = _broadcast_lookup(keys, lookup, 7.0, out_dtype=np.float32)
    per_row = np.array([lookup.get(k, 7.0) for k in keys], dtype=np.float64)
    want = np.nan_to_num(per_row, nan=7.0, posinf=7.0, neginf=7.0).astype(np.float32)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, want)
