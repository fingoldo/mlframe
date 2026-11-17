"""Regression test: ``np.array(lazy)`` must return a private copy, not the cached host array."""

import numpy as np

from mlframe.feature_selection.filters._lazy_host_codes import LazyHostCodes


def test_np_array_of_lazy_codes_is_a_copy():
    """Mutating the result of np.array(lazy) must not change what later reads of the lazy array see."""
    lazy = LazyHostCodes(np.arange(6, dtype=np.int64).reshape(3, 2))
    lazy._host = np.arange(6, dtype=np.int64).reshape(3, 2)
    private = np.array(lazy)
    private[0, 0] = 99
    assert lazy.host()[0, 0] == 0
