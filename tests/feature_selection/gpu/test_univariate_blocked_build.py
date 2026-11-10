"""The univariate basis build in column blocks (the whole candidate matrix not fitting) selects and returns what the whole-matrix build does."""

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _orthogonal_univariate_fe as U
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_gpu_resident as G


def _data(n=5000, p=14, seed=0):
    """A frame whose target depends on x0 squared and x1 cubed."""
    r = np.random.default_rng(seed)
    X = pd.DataFrame(r.standard_normal((n, p)), columns=[f"x{i}" for i in range(p)])
    y = ((X["x0"] ** 2 + X["x1"] ** 3 + 0.3 * r.standard_normal(n)) > 1.0).astype(np.int64).to_numpy()
    return X, y


def test_blocked_build_equals_whole_matrix_build(monkeypatch):
    """Same scores frame (names and values) and the same appended columns, with the block size forced to three source columns."""
    X, y = _data()
    whole_X, whole_scores = U.hybrid_orth_mi_fe(X, y, degrees=(2, 3))
    monkeypatch.setattr(G, "_vram_fits", lambda n, k: False)
    import mlframe.feature_selection.filters._device_blocks as B

    monkeypatch.setattr(B, "first_block", lambda cp_, n, k, item, mult: 3)
    blk_X, blk_scores = U.hybrid_orth_mi_fe(X, y, degrees=(2, 3))
    assert list(blk_scores["engineered_col"]) == list(whole_scores["engineered_col"])
    np.testing.assert_allclose(blk_scores["engineered_mi"].to_numpy(), whole_scores["engineered_mi"].to_numpy(), rtol=0, atol=1e-12)
    assert list(blk_X.columns) == list(whole_X.columns)
    np.testing.assert_array_equal(blk_X.to_numpy(), whole_X.to_numpy())
