"""The GPU smart log shifts each column by its own minimum, matching the CPU per-column rule."""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._internals import smart_log
from mlframe.feature_selection.filters.feature_engineering import _gpu_smart_log


def test_gpu_smart_log_shift_is_per_column():
    """A positive column next to a negative one must not be shifted (numpy stands in for cupy)."""
    X = np.column_stack([np.array([1.0, 2.0, 3.0]), np.array([-2.0, 0.0, 1.0])])
    got = _gpu_smart_log(np, X)
    for j in range(2):
        np.testing.assert_allclose(got[:, j], smart_log(X[:, j]), rtol=1e-12)
