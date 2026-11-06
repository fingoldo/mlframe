"""An empty family gives zero null maxima on the device path instead of failing on a zero-width launch.

The CPU twin returns zeros for an empty family; the device raised (audit PARG3-7, wave 3).
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def test_empty_family_returns_zeros_per_permutation():
    """Zero columns and five permutations give five zeros, with no device launch."""
    from mlframe.feature_selection.filters import _family_maxt_cuda as fm

    cols = np.zeros((0, 50), dtype=np.int16)
    perms = np.tile(np.arange(50), (5, 1))
    out = fm.family_null_max_device(cols, np.array([], dtype=np.int64), perms, 2)
    np.testing.assert_array_equal(out, np.zeros(5))
