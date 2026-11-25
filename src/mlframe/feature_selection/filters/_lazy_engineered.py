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

import functools
import itertools
import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype, register_extension_dtype

from mlframe.utils.env_flags import env_int
from mlframe.utils.log_throttle import log_throttle

from ._cache_release import register_cache_releaser

logger = logging.getLogger(__name__)

#: Rows from which the full-length replay of a winner is deferred (0: never deferred).
DEFAULT_MIN_ROWS = 250_000
#: Bytes of materialised columns kept for repeated reads (shared by every lazy column of the process): room for every engineered column of a 500000 x 1000 fit (800 columns of 2 MB), given back under memory pressure.
CACHE_BYTES = 3 * 1024 * 1024 * 1024
#: Bytes of the replayed first rows of lazy columns (the heads the decision stages probe a recipe on).
HEAD_CACHE_BYTES = 64 * 1024 * 1024
#: A cache holds what the machine has available above this reserve when a column is put into it (floor ``CACHE_FLOOR_BYTES``): the caches hold only what can be recomputed, and a fit at the edge of the
#: RAM (1M x 1000 on 16 GB) has to give it back before the code matrix is copied. The reserve is the FE stages' own (``_fe_min_free_ram_bytes``, 3 GiB). Replaying a column costs seconds at the full
#: row count: with a cache of 512 MB a 500000 x 1000 fit replayed each of its 800 engineered columns three times (4.8 GB of replays, 54 s of the critical path).
CACHE_RESERVE_BYTES = 3 * 1024 * 1024 * 1024
CACHE_FLOOR_BYTES = 64 * 1024 * 1024
#: A head slice of at most this many rows is replayed on the head alone; a longer one computes the column.
HEAD_MAX_ROWS = 65_536
#: Bytes of the rows of lazy columns gathered for the decision subsample (one gather per column and draw, whatever the number of families that read it).
GATHER_CACHE_BYTES = 128 * 1024 * 1024
#: A row gather of at most this share of the rows is answered from (and kept in) the gather cache; a larger one is a plain take.
GATHER_MAX_SHARE = 0.25


def lazy_replay_min_rows() -> int:
    """Rows from which a replayed winner stays lazy, read on every call (``MLFRAME_FE_LAZY_REPLAY_MIN_ROWS``; 0 turns the mechanism off)."""
    return int(env_int("MLFRAME_FE_LAZY_REPLAY_MIN_ROWS", DEFAULT_MIN_ROWS, minimum=0))


_AVAILABLE_TTL_S = 0.5
_AVAILABLE: "list[float]" = [-1e9, 0.0]  # [monotonic time of the reading, available bytes]


def _available_bytes() -> "int | None":
    """The machine's available RAM, read at most every ``_AVAILABLE_TTL_S`` (``psutil.virtual_memory`` takes 2.4 ms on Windows and the caches ask at every ``put``: 96 ms for the 40 columns of one family); ``None`` when unreadable."""
    now = time.monotonic()
    if now - _AVAILABLE[0] < _AVAILABLE_TTL_S:
        return int(_AVAILABLE[1])
    try:
        import psutil

        value = int(psutil.virtual_memory().available)
    except Exception as exc:  # psutil missing or unreadable: the caches keep their own bounds
        log_throttle(logger, "_lazy_engineered.available_bytes", logging.WARNING, "available RAM unreadable (%s: %s); the lazy-column caches keep their own bound", type(exc).__name__, exc)
        return None
    _AVAILABLE[0], _AVAILABLE[1] = now, float(value)
    return value


class _ColumnCache:
    """An LRU of materialised columns bounded by their total bytes; safe for the FE lanes (threads)."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = int(max_bytes)
        self._items: "OrderedDict[Any, np.ndarray]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def __getstate__(self) -> "dict[str, Any]":
        """The bound only: a cache that was pickled is empty again when loaded (its columns are recomputable and its lock is not picklable)."""
        return {"max_bytes": self.max_bytes}

    def __setstate__(self, state: "dict[str, Any]") -> None:
        """Restore an empty cache of the saved bound."""
        self.__init__(state["max_bytes"])  # type: ignore[misc]

    def get(self, key: Any) -> "Optional[np.ndarray]":
        """The cached column (made the newest) or ``None``."""
        with self._lock:
            hit = self._items.get(key)
            if hit is not None:
                self._items.move_to_end(key)
            return hit

    def bound(self) -> int:
        """The bytes the cache may hold now: its own bound, or what the machine has available above ``CACHE_RESERVE_BYTES`` (never below the floor) when it is tight."""
        available = _available_bytes()
        return self.max_bytes if available is None else min(self.max_bytes, max(CACHE_FLOOR_BYTES, available - CACHE_RESERVE_BYTES))

    def put(self, key: Any, values: np.ndarray) -> None:
        """Remember ``values``; the oldest columns go when the total exceeds the bound (a single column above the bound is not kept)."""
        limit = self.bound()
        if values.nbytes > limit:
            return
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= old.nbytes
            self._items[key] = values
            self._bytes += values.nbytes
            while self._bytes > limit and self._items:
                _, dropped = self._items.popitem(last=False)
                self._bytes -= dropped.nbytes

    def clear(self) -> int:
        """Forget every column; the bytes that were held."""
        with self._lock:
            freed = self._bytes
            self._items.clear()
            self._bytes = 0
            return freed


_CACHE = _ColumnCache(CACHE_BYTES)
_HEAD_CACHE = _ColumnCache(HEAD_CACHE_BYTES)
_GATHER_CACHE = _ColumnCache(GATHER_CACHE_BYTES)


@functools.lru_cache(maxsize=1)
def _prefetch_workers() -> int:
    """Threads of the prefetch pool: the physical cores, at most 8 (read once, the count of cores does not change)."""
    try:
        import psutil

        count = psutil.cpu_count(logical=False) or 2
    except Exception:  # psutil missing: the logical count
        count = os.cpu_count() or 2
    return max(1, min(8, int(count)))


def prefetch_columns(X: Any, names: Iterable[str], workers: "int | None" = None, rows: "np.ndarray | None" = None) -> int:
    """Compute, on a thread pool, the values of the lazy engineered columns of ``X`` among ``names`` that are not in the cache yet; the number computed.

    With ``rows`` the work is the gather of those rows of each column (what ``X.iloc[rows]`` asks of them), kept in the gather cache: the columns whose gather is cached are left alone.

    A stage that reads hundreds of engineered columns one after another replays them one at a time on one thread (the accuracy gate: 800 columns at 500000 rows, 12 s of the critical path with the other cores
    idle). The replays run in ``nogil`` numba kernels and numpy, so they overlap; the reader then finds the values in the cache. At most as many columns as half the value cache holds are computed ahead of
    the reader (a prefetch that pushed out its own head would compute everything twice). A replay that fails is left for the reader, which meets the same error on its own call."""
    from concurrent.futures import ThreadPoolExecutor

    idx = None if rows is None else np.asarray(rows, dtype=np.int64)
    gather_key = None if idx is None else (idx.shape[0], hash(idx.tobytes()))
    todo = []
    for name in names:
        try:
            array = X[name].array
        except (KeyError, TypeError, AttributeError):
            continue
        if not isinstance(array, LazyEngineeredColumn) or array._values is not None or _CACHE.get(array._key) is not None:
            continue
        if gather_key is None or _GATHER_CACHE.get((array._key, *gather_key)) is None:
            todo.append(array)
    if not todo:
        return 0
    workers = _prefetch_workers() if workers is None else max(1, min(8, int(workers)))
    room = len(todo) if idx is not None else max(1, int(_CACHE.bound() // 2 // max(1, todo[0]._length * 4)))
    done = 0

    def replay_one(column: "LazyEngineeredColumn") -> bool:
        """One replay (or gather); False when it failed (the reader will meet the error itself)."""
        try:
            if idx is None:
                column.materialize()
            else:
                column.take(idx)
            return True
        except Exception as exc:
            log_throttle(logger, "_lazy_engineered.prefetch", logging.INFO, "prefetch of a lazy column failed (%s: %s); the reader computes it", type(exc).__name__, exc)
            return False

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lazy_prefetch") as pool:
        for start in range(0, len(todo), room):
            done += sum(pool.map(replay_one, todo[start : start + room]))
    return done


def release_caches() -> int:
    """Drop the materialised columns, the replayed heads and the gathered rows (all recomputable); the bytes freed. Registered with the FE stages' cache releaser: it runs when the free RAM is low."""
    return _CACHE.clear() + _HEAD_CACHE.clear() + _GATHER_CACHE.clear()


register_cache_releaser(release_caches)

#: Counters of the process (columns created, replays run, reads answered by the cache, bytes replayed): what the profile of a fit reads to see how often a lazy column is computed.
STATS = {"created": 0, "replayed": 0, "cache_hits": 0, "replayed_bytes": 0, "head_replays": 0, "gather_hits": 0}
_STATS_LOCK = threading.Lock()  # the prefetch pool replays columns on several threads: a bare ``+=`` on a dict value loses increments


def _count(name: str, amount: int = 1) -> None:
    """Add ``amount`` to the process counter ``name``."""
    with _STATS_LOCK:
        STATS[name] += amount


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

    def __init__(
        self,
        replay: "Optional[Callable[[], np.ndarray]]",
        length: int,
        values: "Optional[np.ndarray]" = None,
        head_replay: "Optional[Callable[[int], np.ndarray]]" = None,
    ) -> None:
        self._replay = replay
        self._head_replay = head_replay  # replays the first ``rows`` rows alone (the recipe is row-wise: the transform of any batch is what the full replay holds for its rows)
        self._length = int(length)
        self._values = values  # set for a held (sliced / concatenated) column: ``replay`` is then None
        self._key = next(_SERIAL)
        if replay is not None:
            _count("created")

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
            _count("cache_hits")
            return hit
        assert self._replay is not None
        computed = np.ascontiguousarray(self._replay(), dtype=np.float32).reshape(-1)
        if computed.shape[0] != self._length:
            raise ValueError(f"a lazy engineered column replayed to {computed.shape[0]} rows, expected {self._length}")
        computed.flags.writeable = False
        _count("replayed")
        _count("replayed_bytes", int(computed.nbytes))
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
        """A scalar for an integer, the column itself for a slice of all its rows (``concat`` slices every block that way, and must not compute it), the first rows of the column for a slice that starts at
        its top (replayed on those rows alone), a held column for anything else."""
        if isinstance(item, slice):
            start, stop, step = item.indices(self._length)
            if (start, step) == (0, 1):
                return self if stop == self._length else self._head(stop)
        values = self.materialize()
        if isinstance(item, (int, np.integer)):
            return values[item]
        taken = values[item]
        return LazyEngineeredColumn.from_values(taken)

    def _head(self, rows: int) -> "LazyEngineeredColumn":
        """The first ``rows`` rows as a held column.

        A frame slice ``X.iloc[:2048]`` (the probe of a recipe on the head of the data) slices every column of the frame; answering each lazy one by computing the whole column replayed the engineered
        columns of every earlier family at the full row count again and again (the cache holds a fraction of them). A column whose values are known is cut; otherwise its recipe is replayed on the head alone, once per
        length (a small cache), which recurses through the columns of its own frame the same way."""
        values = self._values if self._values is not None else _CACHE.get(self._key)
        if values is None and self._head_replay is not None and rows <= HEAD_MAX_ROWS:
            head = _HEAD_CACHE.get((self._key, rows))
            if head is None:
                head = np.ascontiguousarray(self._head_replay(rows), dtype=np.float32).reshape(-1)
                if head.shape[0] != rows:
                    raise ValueError(f"a lazy engineered column replayed to {head.shape[0]} head rows, expected {rows}")
                head.flags.writeable = False
                _count("head_replays")
                _HEAD_CACHE.put((self._key, rows), head)
            return LazyEngineeredColumn.from_values(head)
        if values is None:
            values = self.materialize()
        return LazyEngineeredColumn.from_values(values[:rows].copy())

    def isna(self) -> np.ndarray:
        """The NaN mask of the values."""
        return np.isnan(self.materialize())

    def take(self, indices: Any, *, allow_fill: bool = False, fill_value: Any = None) -> "LazyEngineeredColumn":
        """The rows at ``indices`` as a held column (``-1`` is filled with ``fill_value`` or NaN when ``allow_fill``)."""
        idx = np.asarray(indices, dtype=np.int64)
        if 0 < idx.shape[0] <= GATHER_MAX_SHARE * self._length and not (allow_fill and bool((idx < 0).any())):  # pandas asks for a fill even when no index is -1
            return self._gather(idx)
        values = self.materialize()
        if allow_fill:
            fill = np.nan if fill_value is None or fill_value is pd.NA else fill_value
            out = np.where(idx < 0, fill, values[np.where(idx < 0, 0, idx)]).astype(np.float32)
        else:
            out = values[idx]
        return LazyEngineeredColumn.from_values(out)

    def _gather(self, idx: np.ndarray) -> "LazyEngineeredColumn":
        """The rows ``idx`` as a held column, computed once per column and index: every family of a fit that decides on the same subsample gathers the same rows of the same engineered columns, and each
        gather used to replay the column at the full row count again (the value cache holds a fraction of the columns of a fit)."""
        key = (self._key, idx.shape[0], hash(idx.tobytes()))
        out = _GATHER_CACHE.get(key)
        if out is not None:
            _count("gather_hits")
        else:
            out = np.ascontiguousarray(self.materialize()[idx], dtype=np.float32)
            out.flags.writeable = False
            _GATHER_CACHE.put(key, out)
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
