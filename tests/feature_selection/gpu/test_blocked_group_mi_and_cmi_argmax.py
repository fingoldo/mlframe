"""Group MI and the greedy CMI argmax in column blocks equal their one-pass results."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _device_blocks as B
from mlframe.feature_selection.filters._qbin_block_gpu import group_mi_block_gpu


def test_group_mi_blocks_equal_one_pass(monkeypatch):
    """Block of 4 columns over 11 columns gives the one-pass MI exactly."""
    r = np.random.default_rng(0)
    n = 3000
    cols = [r.standard_normal(n) for _ in range(11)]
    y = (cols[0] + 0.5 * r.standard_normal(n) > 0).astype(np.int64)
    whole = group_mi_block_gpu(cols, y, 10)
    monkeypatch.setattr(B, "first_block", lambda cp_, n_, k_, item, mult: 4)
    blocked = group_mi_block_gpu(cols, y, 10)
    assert whole is not None and blocked is not None
    np.testing.assert_array_equal(blocked, whole)
