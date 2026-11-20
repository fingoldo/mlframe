"""gate_divergence indexes y inside an unchecked njit walk: a target of the wrong length must raise."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._gate_rank_divergence import gate_divergence


def test_gate_divergence_rejects_short_target():
    """A target shorter than the matrix rows is rejected instead of read out of bounds."""
    mat = np.random.default_rng(0).normal(size=(200, 4)).astype(np.float32)
    with pytest.raises(ValueError):
        gate_divergence(mat, np.zeros(10, dtype=np.int64), [0, 1], 4)
