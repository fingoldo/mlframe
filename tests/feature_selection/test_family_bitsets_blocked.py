"""The blocked bitset null kernel (four permutations per pass over the column bitsets) returns exactly the values of the one-permutation kernel."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _family_maxt_null_v2_njit as V


@pytest.mark.parametrize("n_y", [2, 3])
@pytest.mark.parametrize("b_count", [1, 3, 4, 7, 10])
@pytest.mark.parametrize("n", [100, 1000, 4099])
def test_blocked_kernel_equals_the_single_permutation_kernel(n_y, b_count, n):
    """Bit-equal maxima for permutation counts that do and do not fill the last block, label counts and row counts that do and do not fill the last word."""
    rng = np.random.default_rng(n + b_count)
    k = 7
    cols = rng.integers(0, 6, (k, n)).astype(np.int64)
    n_x = np.full(k, 6, dtype=np.int64)
    y = rng.integers(0, n_y, n).astype(np.int64)
    y_perms = np.stack([y[rng.permutation(n)] for _ in range(b_count)])
    table, c_a, px, groups = V.bitset_groups(cols, n_x)
    bits = groups[0][2]
    want = V.family_null_max_bitsets(bits, n_x, y_perms, n_y, table, c_a, px)
    got = V.family_null_max_bitsets_blocked(bits, n_x, y_perms, n_y, table, c_a, px)
    assert np.array_equal(want, got)
    assert np.array_equal(V.null_max_bitsets((table, c_a, px, groups), n_x, y_perms, n_y), want)
