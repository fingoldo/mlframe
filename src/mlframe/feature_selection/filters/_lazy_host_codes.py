"""A host view of device-resident integer codes that is only copied back when something actually reads it.

The GPU-resident FE stages keep every candidate's bin codes on the device and used to ALSO copy each one back to the host "as the byte-path fallback" for
sites that might need numpy. In resident mode almost none do, so those copies were pure device-to-host traffic. ``LazyHostCodes`` stands in for that host
copy: shape/size/dtype are known without touching the device, and the first real read (``np.asarray``, indexing, any ndarray method) performs the one
copy and caches it. A site that never reads the host form never pays for it; a site that does gets exactly the bytes the eager copy would have produced.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

# Below this many elements the copy is too small for the two range reductions to pay for themselves.
_NARROW_MIN_ELEMENTS = 4096


class LazyHostCodes:
    """Deferred ``cp.asnumpy(dev).astype(dtype)`` of a device code array.

    Construction remembers the device array and the host dtype; nothing is copied until the host view is first read.
    """

    __slots__ = ("_cast", "_codes", "_dev", "_dtype", "_host", "_parent", "shape")

    def __init__(self, dev: Any, dtype: Any = np.int64) -> None:
        self._dev = dev
        self._dtype = np.dtype(dtype)
        self._host: np.ndarray | None = None
        self._cast: "dict[np.dtype, np.ndarray] | None" = None
        self._codes: "dict[tuple, np.ndarray] | None" = None  # host quantile codes already computed from this array, by (nbins, dtype) and column
        self._parent: "tuple[LazyHostCodes, int] | None" = None  # (array, column) this one is a one-column view of
        self.shape = tuple(int(s) for s in dev.shape)

    @property
    def dev(self) -> Any:
        """The device array this view mirrors."""
        return self._dev

    @property
    def dtype(self) -> np.dtype:
        """Host dtype of the materialised copy."""
        return cast("np.dtype[Any]", self._dtype)

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
            self._host = self._narrow_copy()
            if self._host is None:
                import cupy as cp

                self._host = np.asarray(cp.asnumpy(self._dev)).astype(self._dtype, copy=False)
        return self._host

    def codes_cached(self, nbins: int, dtype: Any) -> "np.ndarray | None":
        """A copy of the host quantile codes of this array at ``(nbins, dtype)`` when they were already computed for it, or for every one of its columns; else None.

        The codes of one column are the same whether it is binned alone or inside a block (each column is binned independently), so a stage that binned a one-column view
        spares the stage that bins the whole block the second device-to-host copy. The cache belongs to this array object and goes with it.
        """
        store = self._codes
        if not store:
            return None
        tag = (int(nbins), np.dtype(dtype))
        whole = store.get((tag, None))
        if whole is not None:
            return whole.copy()
        if len(self.shape) == 2 and self.shape[1] > 0 and all((tag, j) in store for j in range(self.shape[1])):
            joined: np.ndarray = np.concatenate([store[(tag, j)] for j in range(self.shape[1])], axis=1)
            return joined
        return None

    def store_codes(self, nbins: int, dtype: Any, codes: np.ndarray) -> None:
        """Remember the host quantile codes just computed from this array, and from its parent's column when it is a one-column view of one."""
        tag = (int(nbins), np.dtype(dtype))
        if self._codes is None:
            self._codes = {}
        self._codes[(tag, None)] = codes.copy()
        if self._parent is not None and codes.ndim == 2 and codes.shape[1] == 1:
            parent, col = self._parent
            if parent._codes is None:
                parent._codes = {}
            parent._codes[(tag, col)] = codes.copy()

    def host_as(self, dtype: Any) -> np.ndarray:
        """The array as a host array of ``dtype``, converted once and kept.

        A float array read at a narrower float dtype is cast on the device first, so only the narrow bytes cross (half for double -> single precision); every reader of the same
        dtype shares the one copy. The device cast flushes results below the smallest normal float32 (about 1.2e-38) to zero where the host cast keeps a subnormal; no
        consumer of a criterion-dtype column can tell the two apart.
        """
        want = np.dtype(dtype)
        if self._host is not None:
            return self._host.astype(want, copy=False)
        if want == self._dtype:
            return self.host()
        if self._cast is None:
            self._cast = {}
        got = self._cast.get(want)
        if got is None:
            dev = self._dev
            if want.kind == "f" and dev.dtype.kind == "f" and want.itemsize < dev.dtype.itemsize:
                import cupy as cp

                got = np.asarray(cp.asnumpy(dev.astype(want)))
            else:
                got = self.host().astype(want, copy=False)
            self._cast[want] = got
        return got

    def _narrow_copy(self) -> "np.ndarray | None":
        """The host array from ONE transfer of int8 codes with the array's min and max appended, or None when the values do not fit int8 or the array is not worth it.

        Codes are small ordinals but often stored as int64. The range travels in the same transfer, so the copy stays a single device-to-host operation; the host checks it
        and widens back to the requested dtype, which leaves the array identical to the dense copy. A range that does not fit costs a second, dense copy.
        """
        dev = self._dev
        if self._dtype.kind not in "iu" or dev.dtype.kind not in "iu" or dev.dtype.itemsize <= 1 or int(dev.size) < _NARROW_MIN_ELEMENTS:
            return None
        import cupy as cp

        flat = dev.reshape(-1)
        tail = cp.stack([flat.min(), flat.max()]).astype(cp.int64).view(cp.int8)
        packed = np.asarray(cp.asnumpy(cp.concatenate([flat.astype(cp.int8), tail])))
        lo, hi = packed[-16:].view(np.int64)
        if lo < -128 or hi > 127:
            return None
        return packed[:-16].reshape(dev.shape).astype(self._dtype, copy=False)

    def max(self) -> Any:
        """Largest code: a scalar reduction on the device unless the host copy already exists."""
        return self._host.max() if self._host is not None else self._dtype.type(self._dev.max().get())

    def min(self) -> Any:
        """Smallest code: a scalar reduction on the device unless the host copy already exists."""
        return self._host.min() if self._host is not None else self._dtype.type(self._dev.min().get())

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        """numpy conversion: materialises the host copy; a request for a narrower float dtype takes the shared narrow copy (``host_as``), so ``np.asarray(lazy, dtype=...)`` never moves double precision."""
        if dtype is None:
            return self.host()
        return self.host_as(dtype)

    def __len__(self) -> int:
        """Length of the first axis (known without a copy)."""
        return self.shape[0]

    def __getitem__(self, key: Any) -> Any:
        """Indexing reads the host copy, except a (row-slice, column) pick of a 2-D array (``[:, j]``, ``[::k, j]``), which stays a lazy device view."""
        if self._host is None and len(self.shape) == 2 and isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], slice) and isinstance(key[1], (int, np.integer)):
            return LazyHostCodes(self._dev[key[0], int(key[1])], self._dtype)
        if self._host is None and len(self.shape) == 2 and isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], slice) and isinstance(key[1], slice):
            child = LazyHostCodes(self._dev[key[0], key[1]], self._dtype)  # a row / column range stays a lazy device view as well
            first, stop, step = key[1].indices(self.shape[1])
            if key[0] == slice(None) and step == 1 and stop - first == 1:
                child._parent = (self, first)  # the codes computed for this column are then known to the whole array
            return child
        return self.host()[key]

    def __getattr__(self, name: str) -> Any:
        """Any other ndarray attribute (``astype``, ``max``, ``ravel``, ...) is served by the host copy."""
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.host(), name)


def length_of(values: Any) -> int:
    """First-axis length of an array, a lazy device-backed array or any sequence, without materialising a lazy one (``np.asarray(x).shape[0]`` copies it whole)."""
    shape = getattr(values, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[0])
    return int(np.asarray(values).shape[0])  # length-ok: no shape attribute means a plain sequence


def size_of(values: Any) -> int:
    """Element count of an array, a lazy device-backed array or any sequence, without materialising a lazy one."""
    shape = getattr(values, "shape", None)
    if shape is not None:
        return int(np.prod(shape, dtype=np.int64))
    return int(np.asarray(values).size)  # length-ok: no shape attribute means a plain sequence


def host_as(values: Any, dtype: Any) -> np.ndarray:
    """``values`` as a host array of ``dtype``: a lazy device-backed array goes through its cached, possibly narrowed copy (``LazyHostCodes.host_as``), anything else through ``np.asarray``."""
    if isinstance(values, LazyHostCodes):
        return values.host_as(dtype)
    return np.asarray(values, dtype=dtype)


def strided(values: Any, stride: int) -> Any:
    """Every ``stride``-th row of a host array or a lazy device-backed one, keeping the latter on the device."""
    if stride <= 1:
        return values
    if isinstance(values, LazyHostCodes):
        return LazyHostCodes(values.dev[::stride], values.dtype)
    return values[::stride]
