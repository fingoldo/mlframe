"""A direct call whose joint histogram cannot fit a block's shared memory is refused with a clear error, not a launch failure."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cupy")

from mlframe.feature_selection.filters.info_theory._cmi_cuda import conditional_mi_batched_cuda


def test_oversized_joint_histogram_raises_value_error():
    """40 x 40 x 40 cells need 256,000 bytes of shared memory, far above the 48 KB block limit."""
    rng = np.random.default_rng(0)
    n, p = 500, 3
    Xc = rng.integers(0, 40, size=(p, n)).astype(np.int32)
    y = rng.integers(0, 40, size=n).astype(np.int32)
    z = rng.integers(0, 40, size=n).astype(np.int32)
    with pytest.raises(ValueError, match="shared memory"):
        conditional_mi_batched_cuda(Xc, y, z, 40, 40, 40)


def test_small_joint_histogram_still_runs():
    """A small histogram keeps working."""
    rng = np.random.default_rng(1)
    n, p = 500, 3
    Xc = rng.integers(0, 4, size=(p, n)).astype(np.int32)
    out = conditional_mi_batched_cuda(Xc, rng.integers(0, 4, size=n).astype(np.int32), rng.integers(0, 3, size=n).astype(np.int32), 4, 4, 3)
    assert out.shape == (p,) and np.all(np.isfinite(out))
