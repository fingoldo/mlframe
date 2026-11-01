"""The cupy DTW entry points refuse a buffer whose index would overflow the kernels' 32-bit int (MEM3-1, MEM3-2).

The full-matrix and banded kernels compute ``i * stride + j`` in 32-bit ints. Above 2**31 - 1 cells the index wraps and the kernel writes out
of bounds. The check runs before any device work, so it does not need a GPU to be exercised.
"""

import numpy as np
import pytest

from mlframe.signal import dtw


def test_full_matrix_over_the_kernel_index_range_is_refused_before_allocation():
    """A pair of 50k-point series needs about 2.5e9 cells: refused with the limit named."""
    x = np.zeros(50_000, dtype=np.float32)
    y = np.zeros(50_000, dtype=np.float32)
    with pytest.raises(ValueError, match="32-bit kernel index limit"):
        dtw.dtw_cupy_full(x, y)


def test_banded_buffer_over_the_kernel_index_range_is_refused_before_allocation():
    """A wide Sakoe-Chiba window over a long series also overflows the band index: refused the same way."""
    x = np.zeros(50_000, dtype=np.float32)
    y = np.zeros(50_000, dtype=np.float32)
    with pytest.raises(ValueError, match="32-bit kernel index limit"):
        dtw.dtw_cupy_banded(x, y, window=30_000)
