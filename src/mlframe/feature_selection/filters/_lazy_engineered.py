"""Engineered columns of a large frame that are replayed from their recipe when read, instead of being stored as full-length float arrays.

At 1000000 rows an engineered column costs 4 MB as float32. The FE families replay every winner of their decision on the row subsample at the full row count and append it to the frame, and a
fit with every family on ends with 714 such columns: 2.86 GB, held from the cascades to the end of the fit (the census of a 1M x 1000 fit at 150 s: 2856 MB in one frame of 714 float32 columns,
65 percent of the growth of the process). Most of them never reach the selection; the ones that do are read a few times (the discretisation pass, the pair stage's operands, the gates).

:class:`LazyEngineeredColumn` is a pandas extension array that holds the recipe and the frame it is replayed on, and materialises the float32 values when something reads them (``np.asarray``,
``to_numpy``, ``take``, ``astype``): the same values the eager replay stored, because the recipe is a closed-form function of the raw columns (the contract of ``fe_decide_on_subsample``). A
small cache of the newest materialised columns, shared by all of them and bounded in bytes, answers the repeated reads of one column; everything else is recomputed on demand. A column that was
sliced or concatenated is held as plain values (the subsample gathers of the decision stages are small).

Used only for pandas frames of at least ``MLFRAME_FE_LAZY_REPLAY_MIN_ROWS`` rows (default 250000; ``0`` turns the mechanism off).
"""

from __future__ import annotations

import itertools
import threading
from collections import OrderedDict
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype, register_extension_dtype

from mlframe.utils.env_flags import env_int

#: Rows from which the full-length replay of a winner is deferred (0: never deferred).
DEFAULT_MIN_ROWS = 250_000
#: Bytes of materialised columns kept for repeated reads (shared by every lazy column of the process).
CACHE_BYTES = 512 * 1024 * 1024


def lazy_replay_min_rows() -> int:
    """Rows from which a replayed winner stays lazy, read on every call (``MLFRAME_FE_LAZY_REPLAY_MIN_ROWS``; 0 turns the mechanism off)."""
    return int(env_int("MLFRAME_FE_LAZY_REPLAY_MIN_ROWS", DEFAULT_MIN_ROWS, minimum=0))


class _ColumnCache:
    """An LRU of materialised columns bounded by their total bytes; safe for the FE lanes (threads)."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = int(max_bytes)
        self._items: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def __getstate__(self) -> "dict[str, Any]":
        """The bound only: a cache that was pickled is empty again when loaded (its columns are recomputable and its lock is not picklable)."""
        return {"max_bytes": self.max_bytes}

    def __setstate__(self, state: "dict[str, Any]") -> None:
        """Restore an empty cache of the saved bound."""
        self.__init__(state["max_bytes"])  # type: ignore[misc]

    def get(self, key: int) -> "Optional[np.ndarray]":
        """The cached column (made the newest) or ``None``."""
        with self._lock:
            hit = self._items.get(key)
            if hit is not None:
                self._items.move_to_end(key)
            return hit

    def put(self, key: int, values: np.ndarray) -> None:
        """Remember ``values``; the oldest columns go when the total exceeds the bound (a single column above the bound is not kept)."""
        if values.nbytes > self.max_bytes:
            return
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= old.nbytes
            self._items[key] = values
            self._bytes += values.nbytes
            while self._bytes > self.max_bytes and self._items:
                _, dropped = self._items.popitem(last=False)
                self._bytes -= dropped.nbytes

    def clear(self) -> None:
        """Forget every column."""
        with self._lock:
            self._items.clear()
            self._bytes = 0


_CACHE = _ColumnCache(CACHE_BYTES)
#: Counters of the process (columns created, replays run, reads answered by the cache, bytes replayed): what the profile of a fit reads to see how often a lazy column is computed.
STATS = {"created": 0, "replayed": 0, "cache_hits": 0, "replayed_bytes": 0}
_SERIAL = itertools.count()  # a key per column that is never reused (the ids of collected objects are)


#: The sentinel pandas passes for "no ``na_value`` was given" (``Series.to_numpy`` hands it over as it is).
_NO_DEFAULT = getattr(pd.api.extensions, "no_default", object())


def _na_value_is_unset(na_value: Any) -> bool:
    """Whether ``to_numpy`` was given no replacement for the missing values (nothing, pandas' own sentinel, ``pd.NA`` or NaN)."""
    if na_value is None or na_value is _NO_DEFAULT or na_value is pd.NA:
        return True
    if isinstance(na_value, float):
        return bool(np.isnan(na_value))
    return False


@register_extension_dtype
class LazyEngineeredDtype(ExtensionDtype):
    """The dtype of :class:`LazyEngineeredColumn`: a numeric float32 column."""

    name = "lazy_engineered_float32"
    type = np.float32
    kind = "f"
    _is_numeric = True
    na_value = np.nan

    @classmethod
    def construct_array_type(cls) -> "type[LazyEngineeredColumn]":
        """The array class of this dtype."""
        return LazyEngineeredColumn

    def _get_common_dtype(self, dtypes: Any) -> "Optional[np.dtype]":
        """The dtype of a frame block mixing lazy columns with numpy floats and integers (``DataFrame.to_numpy()``): their common numpy dtype, so the mix is not turned into ``object``."""
        numpy_parts = [d for d in dtypes if isinstance(d, np.dtype)]
        others = [d for d in dtypes if not isinstance(d, np.dtype) and not isinstance(d, LazyEngineeredDtype)]
        if others or any(d.kind not in "fiub" for d in numpy_parts):
            return None
        return np.result_type(np.float32, *numpy_parts) if numpy_parts else np.dtype(np.float32)


class LazyEngineeredColumn(ExtensionArray):
    """A float32 column that is computed by ``replay()`` when it is read, or plain values after a slice; see the module docstring."""

    _dtype = LazyEngineeredDtype()

    def __init__(self, replay: "Optional[Callable[[], np.ndarray]]", length: int, values: "Optional[np.ndarray]" = None) -> None:
        self._replay = replay
        self._length = int(length)
        self._values = values  # set for a held (sliced / concatenated) column: ``replay`` is then None
        self._key = next(_SERIAL)
        if replay is not None:
            STATS["created"] += 1

    # ---- construction -------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @classmethod
    def from_values(cls, values: np.ndarray) -> "LazyEngineeredColumn":
        """A column that holds ``values`` (float32) itself."""
        arr = np.ascontiguousarray(values, dtype=np.float32)
        return cls(None, int(arr.shape[0]), arr)

    @classmethod
    def _from_sequence(cls, scalars: Any, *, dtype: Any = None, copy: bool = False) -> "LazyEngineeredColumn":
        """The array pandas builds from a sequence of values."""
        return cls.from_values(np.asarray(scalars, dtype=np.float32))

    @classmethod
    def _from_factorized(cls, values: Any, original: Any) -> "LazyEngineeredColumn":
        """The inverse of ``factorize`` (only the held values are ever factorised)."""
        return cls.from_values(np.asarray(values, dtype=np.float32))

    @classmethod
    def _concat_same_type(cls, to_concat: Any) -> "LazyEngineeredColumn":
        """Concatenation along rows: the values of every part, held."""
        return cls.from_values(np.concatenate([np.asarray(part) for part in to_concat]))

    # ---- values -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def materialize(self) -> np.ndarray:
        """The float32 values of the column (read-only): computed on the first read and kept in the shared cache while it is among the newest."""
        if self._values is not None:
            return self._values
        hit = _CACHE.get(self._key)
        if hit is not None:
            STATS["cache_hits"] += 1
            return hit
        assert self._replay is not None
        computed = np.ascontiguousarray(self._replay(), dtype=np.float32).reshape(-1)
        if computed.shape[0] != self._length:
            raise ValueError(f"a lazy engineered column replayed to {computed.shape[0]} rows, expected {self._length}")
        computed.flags.writeable = False
        STATS["replayed"] += 1
        STATS["replayed_bytes"] += int(computed.nbytes)
        _CACHE.put(self._key, computed)
        return computed

    def __array__(self, dtype: Any = None, copy: "bool | None" = None) -> np.ndarray:
        """``np.asarray(column)``: the values, cast when a dtype is asked for."""
        values = self.materialize()
        if dtype is not None and np.dtype(dtype) != values.dtype:
            return values.astype(dtype)
        return values.copy() if copy else values

    def to_numpy(self, dtype: Any = None, copy: bool = False, na_value: Any = None) -> np.ndarray:
        """The values as an ndarray (``na_value`` replaces the NaN when given)."""
        unset = _na_value_is_unset(na_value)
        out = self.__array__(dtype, copy=copy if unset else True)
        return out if unset else np.where(np.isnan(out), na_value, out)

    # ---- the ExtensionArray contract ----------------------------------------------------------------------------------------------------------------------------------------------------

    @property
    def dtype(self) -> LazyEngineeredDtype:
        """The dtype object."""
        return self._dtype

    @property
    def nbytes(self) -> int:
        """Bytes held by the column itself: nothing for a lazy column (its values live in the shared cache), the values of a held one."""
        return int(self._values.nbytes) if self._values is not None else 0

    def __len__(self) -> int:
        """The number of rows."""
        return self._length

    def __getitem__(self, item: Any) -> Any:
        """A scalar for an integer, the column itself for a slice of all its rows (``concat`` slices every block that way, and must not compute it), a held column for anything else."""
        if isinstance(item, slice) and item.indices(self._length) == (0, self._length, 1):
            return self
        values = self.materialize()
        if isinstance(item, (int, np.integer)):
            return values[item]
        taken = values[item]
        return LazyEngineeredColumn.from_values(taken)

    def isna(self) -> np.ndarray:
        """The NaN mask of the values."""
        return np.isnan(self.materialize())

    def take(self, indices: Any, *, allow_fill: bool = False, fill_value: Any = None) -> "LazyEngineeredColumn":
        """The rows at ``indices`` as a held column (``-1`` is filled with ``fill_value`` or NaN when ``allow_fill``)."""
        idx = np.asarray(indices, dtype=np.int64)
        values = self.materialize()
        if allow_fill:
            fill = np.nan if fill_value is None or fill_value is pd.NA else fill_value
            out = np.where(idx < 0, fill, values[np.where(idx < 0, 0, idx)]).astype(np.float32)
        else:
            out = values[idx]
        return LazyEngineeredColumn.from_values(out)

    def copy(self) -> "LazyEngineeredColumn":
        """The same column (a lazy column is immutable, so sharing it is a copy)."""
        return self if self._values is None else LazyEngineeredColumn.from_values(self._values.copy())

    def astype(self, dtype: Any, copy: bool = True) -> Any:
        """The values in another dtype (this dtype keeps the column)."""
        if isinstance(dtype, LazyEngineeredDtype) or dtype == "lazy_engineered_float32":
            return self
        return np.asarray(self.materialize(), dtype=dtype) if not isinstance(dtype, pd.api.extensions.ExtensionDtype) else pd.array(self.materialize(), dtype=dtype)

    def __eq__(self, other: Any) -> Any:
        """Element-wise equality with a scalar or an array-like."""
        return self.materialize() == np.asarray(other)

    def _reduce(self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs: Any) -> Any:
        """``Series.<name>()`` of the values (``mean``, ``std``, ``min``, ...)."""
        result = getattr(pd.Series(self.materialize()), name)(skipna=skipna, **kwargs)
        # Intentional float64: the pandas result of a reduction is a double
        return pd.array([result], dtype="float64") if keepdims else result

    def _values_for_factorize(self) -> "tuple[np.ndarray, float]":
        """The values and the NaN sentinel for ``factorize``."""
        return self.materialize(), np.nan

    def _formatter(self, boxed: bool = False) -> Callable[[Any], str]:
        """How a value prints."""
        return lambda x: f"{x:.6g}"

    def __reduce__(self) -> "tuple[Any, ...]":
        """Pickle the values (the closure of a lazy column holds the frame it is replayed on)."""
        return (LazyEngineeredColumn.from_values, (self.materialize(),))

    def __repr__(self) -> str:
        """A short description that does not compute the values."""
        return f"<LazyEngineeredColumn rows={self._length} {'held' if self._values is not None else 'lazy'}>"
