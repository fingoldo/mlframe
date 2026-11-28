"""Cluster 13 regression: the kernel dtype audit resolves C ``long`` for the platform it runs on (audit 2026-11-20, finding 13-F28)."""

import ctypes

import numpy as np

from mlframe.feature_selection._benchmarks import kernel_arg_dtype_audit as audit


def test_long_and_unsigned_long_have_the_platform_size():
    """Both map to a numpy type of ctypes.sizeof(c_long) bytes: 4 on Windows, 8 on LP64 systems."""
    size = ctypes.sizeof(ctypes.c_long)
    assert np.dtype(audit._NUMPY_OF_C["long"]).itemsize == size
    assert np.dtype(audit._NUMPY_OF_C["unsigned long"]).itemsize == size
    assert np.dtype(audit._NUMPY_OF_C["long long"]).itemsize == 8
