"""The permuted label block built from sub-blocks of permutations equals the one-permutation-at-a-time form for any sub-block size."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _null_max_perm_gpu as nm


@pytest.mark.parametrize("upload_bytes", [4, 4 * 1000 * 3, 64 * 1024 * 1024])
@pytest.mark.parametrize("width", [None, 1024])
def test_permuted_labels_subblocks_equal_rows(monkeypatch, upload_bytes, width):
    """Whole block, three-permutation sub-blocks and one-permutation sub-blocks give the same matrix, with and without zero padding."""
    monkeypatch.setattr(nm, "_PERM_UPLOAD_BYTES", upload_bytes)
    rng = np.random.default_rng(1)
    n, count = 1000, 7
    y8 = cp.asarray(rng.integers(0, 2, n).astype(np.uint8))
    perms = np.stack([rng.permutation(n).astype(np.int32) for _ in range(count)])
    got = cp.asnumpy(nm._permuted_labels(cp, y8, perms, 2, 7, width=width))
    ref = np.zeros((5, width or n), dtype=np.uint8)
    for r, p in enumerate(range(2, 7)):
        ref[r, :n] = cp.asnumpy(y8)[perms[p]]
    np.testing.assert_array_equal(got, ref)
