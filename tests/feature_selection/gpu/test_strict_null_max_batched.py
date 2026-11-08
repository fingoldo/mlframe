"""The STRICT local-gate null max from one batched launch equals the per-permutation loop it replaces."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_mi_null_max as N


@pytest.mark.parametrize("n_classes", [2, 3, 5])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batched_equals_loop(monkeypatch, n_classes, dtype):
    """Distinct permutations, continuous and tied columns: bit-identical per-permutation maxima."""
    pytest.importorskip("cupy")
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    rng = np.random.default_rng(n_classes)
    n = 8000
    X = np.column_stack([rng.standard_normal((n, 6)), np.round(rng.standard_normal((n, 4)) * 2)]).astype(dtype)
    y = rng.integers(0, n_classes, n)
    perms = [rng.permutation(n) for _ in range(25)]
    monkeypatch.setenv("MLFRAME_STRICT_NULL_BATCHED", "0")
    want = N._null_max_strict_gpu(X, y, perms, 10)
    monkeypatch.setenv("MLFRAME_STRICT_NULL_BATCHED", "1")
    got = N._null_max_strict_gpu(X, y, perms, 10)
    assert want is not None, "the STRICT path must be taken for this shape when cupy is present"
    assert got is not None and np.array_equal(got, want)
