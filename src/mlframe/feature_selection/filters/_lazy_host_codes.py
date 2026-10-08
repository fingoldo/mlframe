"""A host view of device-resident integer codes that is only copied back when something actually reads it.

The GPU-resident FE stages keep every candidate's bin codes on the device and used to ALSO copy each one back to the host "as the byte-path fallback" for
sites that might need numpy. In resident mode almost none do, so those copies were pure device-to-host traffic. ``LazyHostCodes`` stands in for that host
copy: shape/size/dtype are known without touching the device, and the first real read (``np.asarray``, indexing, any ndarray method) performs the one
copy and caches it. A site that never reads the host form never pays for it; a site that does gets exactly the bytes the eager copy would have produced.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class LazyHostCodes:
    """Deferred ``cp.asnumpy(dev).astype(dtype)`` of a device code array.

    Construction remembers the device array and the host dtype; nothing is copied until the host view is first read.
    """

    __slots__ = ("_dev", "_dtype", "_host", "shape")

    def __init__(self, dev: Any, dtype: Any = np.int64) -> None:
        self._dev = dev
        self._dtype = np.dtype(dtype)
        self._host: np.ndarray | None = None
        self.shape = tuple(int(s) for s in dev.shape)

    @property
    def dev(self) -> Any:
        """The device array this view mirrors."""
        return self._dev

    @property
    def dtype(self) -> np.dtype:
        """Host dtype of the materialised copy."""
        return self._dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions (known without a copy)."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Number of elements (known without a copy)."""
        return int(np.prod(self.shape, dtype=np.int64))

    def host(self) -> np.ndarray:
        """The host array, copied from the device on first use."""
        if self._host is None:
            import cupy as cp

            self._host = np.asarray(cp.asnumpy(self._dev)).astype(self._dtype, copy=False)
        return self._host

    def max(self) -> Any:
        """Largest code: a scalar reduction on the device unless the host copy already exists."""
        return self._host.max() if self._host is not None else self._dtype.type(self._dev.max().get())

    def min(self) -> Any:
        """Smallest code: a scalar reduction on the device unless the host copy already exists."""
        return self._host.min() if self._host is not None else self._dtype.type(self._dev.min().get())

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        """numpy conversion: materialises the host copy."""
        h = self.host()
        return h if dtype is None else h.astype(dtype, copy=False)

    def __len__(self) -> int:
        """Length of the first axis (known without a copy)."""
        return self.shape[0]

    def __getitem__(self, key: Any) -> Any:
        """Indexing reads the host copy, except a (row-slice, column) pick of a 2-D array (``[:, j]``, ``[::k, j]``), which stays a lazy device view."""
        if self._host is None and len(self.shape) == 2 and isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], slice) and isinstance(key[1], (int, np.integer)):
            return LazyHostCodes(self._dev[key[0], int(key[1])], self._dtype)
        return self.host()[key]

    def __getattr__(self, name: str) -> Any:
        """Any other ndarray attribute (``astype``, ``max``, ``ravel``, ...) is served by the host copy."""
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.host(), name)


def strided(values: Any, stride: int) -> Any:
    """Every ``stride``-th row of a host array or a lazy device-backed one, keeping the latter on the device."""
    if stride <= 1:
        return values
    if isinstance(values, LazyHostCodes):
        return LazyHostCodes(values.dev[::stride], values.dtype)
    return values[::stride]
