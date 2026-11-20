"""The memoized device pair-index arrays equal the per-call upload and are reused between calls ."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _resident_candidate_mi as rcm


@pytest.mark.parametrize("m", [0, 2, 3, 5])
def test_cached_pair_indices_equal_upload(m):
    """Values match the uncached upload for each operand count, and a second call returns the same device arrays."""
    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    ref = rcm._pair_index_arrays_upload(cp, pairs)
    first = rcm._pair_index_arrays(cp, pairs)
    second = rcm._pair_index_arrays(cp, pairs)
    for r, f in zip(ref, first):
        np.testing.assert_array_equal(cp.asnumpy(r), cp.asnumpy(f))
    assert first[0] is second[0] and first[1] is second[1]
