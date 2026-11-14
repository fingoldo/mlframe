"""The cupy variant of the interaction pre-rank score streams the candidate columns in blocks and joins them to the same vector."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _device_blocks
from mlframe.feature_selection.filters import _fe_interaction_prerank_kernels as kernels


def _operands(n: int = 4000, p: int = 900, k: int = 5):
    """Standardized float32 operands and a class matrix."""
    rng = np.random.default_rng(3)
    return (rng.standard_normal((n, p)).astype(np.float32), rng.standard_normal((n, p)).astype(np.float32), rng.standard_normal((n, k)).astype(np.float32))


def test_discrete_score_cupy_blocks_equal_unblocked(monkeypatch):
    """Blocks of 100 columns give exactly the unblocked scores, and both agree with the numpy variant to float32 round-off."""
    zv, zv2, yc = _operands()
    whole = kernels.discrete_score_cupy(zv, zv2, yc)
    monkeypatch.setattr(_device_blocks, "first_block", lambda cp_, n_rows, n_cols, itemsize, multiple: 100)
    blocked = kernels.discrete_score_cupy(zv, zv2, yc)
    assert whole.shape == (zv.shape[1],)
    assert np.array_equal(whole, blocked)
    want = kernels.discrete_score_numpy(zv, zv2, yc)
    np.testing.assert_allclose(whole, want, rtol=1e-4, atol=1e-4)
