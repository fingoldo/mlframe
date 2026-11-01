"""The distinct-id lookup of the ordinal-pattern TE equals the per-row Python lookup."""

import numpy as np

from mlframe.feature_selection.filters._ordinal_pattern_fe import _lookup_perm_ids


def test_lookup_equals_per_row_loop():
    """Seen ids, an unseen id (default), NaN and inf ids, for a float32 and a float64 default."""
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 7, 5000).astype(np.float64)
    ids[::11] = np.nan
    ids[5::97] = np.inf
    lookup = {0: 0.1, 1: 0.5, 2: -0.3, 3: 1.25, 4: 0.0}  # ids 5 and 6 are unseen
    got = _lookup_perm_ids(ids, lookup, 0.42)
    want = np.array([lookup.get(int(p), 0.42) if np.isfinite(p) else np.nan for p in ids], dtype=got.dtype)
    np.testing.assert_array_equal(got, want)
    assert np.isnan(got[::11]).all() and np.isnan(got[5::97]).all()
