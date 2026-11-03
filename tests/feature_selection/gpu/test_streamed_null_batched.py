"""The streamed null max from one batched launch per block equals the per-permutation launches, float32 and float64 blocks."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._orthogonal_univariate_fe._streamed_gpu_mi import null_max_streamed


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_batched_equals_per_permutation(monkeypatch, dtype, n_classes):
    """Signal and noise columns over two blocks, several permutations."""
    rng = np.random.default_rng(3)
    n = 5000
    X = rng.standard_normal((n, 12)).astype(dtype)
    y = (X[:, 0] > 0).astype(np.int64) + (X[:, 1] > 1).astype(np.int64) * (n_classes - 2)
    perms = [rng.permutation(n) for _ in range(9)]

    def run():
        """The streamed null max over the two blocks."""
        return null_max_streamed([X[:, :7], X[:, 7:]], y, perms, nbins=10, n_rows=n, dtype=dtype, n_cols=12)

    monkeypatch.setenv("MLFRAME_STRICT_NULL_BATCHED", "0")
    want = run()
    monkeypatch.setenv("MLFRAME_STRICT_NULL_BATCHED", "1")
    got = run()
    assert np.array_equal(got, want)
