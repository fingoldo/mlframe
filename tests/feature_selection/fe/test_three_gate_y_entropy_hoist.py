"""Passing a fold's precomputed H(y) to the three-gate MI gives the value computed per column."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_three_gate_mi_fe import _mi_from_binned_xy, _y_entropy


@pytest.mark.parametrize("clip_zero", [False, True])
@pytest.mark.parametrize("n_classes", [2, 5])
def test_hoisted_y_entropy_equals_per_column(clip_zero, n_classes):
    """Bit-equal MI for informative, noise and constant columns, binary and multiclass labels."""
    rng = np.random.default_rng(n_classes)
    y = rng.integers(0, n_classes, 3000)
    cols = [(y + rng.integers(0, 2, 3000)) % 10, rng.integers(0, 10, 3000), np.zeros(3000, dtype=np.int64)]
    yh = _y_entropy(y)
    for x in cols:
        assert _mi_from_binned_xy(x, y, clip_zero=clip_zero, y_entropy=yh) == _mi_from_binned_xy(x, y, clip_zero=clip_zero)
