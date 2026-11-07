"""Append-only store of engineered columns: one ``(columns, rows)`` float32 block per chunk, written once and read back by name or as a block.

The FE stages used to append their candidates by rebuilding the frame (``pd.concat`` / ``DataFrame(dict)``), which copies every existing column each time. Here a column is copied into the arena a
single time; growth adds a chunk and never moves what is stored. Above ``memmap_bytes`` the chunks are file-backed (``np.memmap``), so the store can exceed the RAM. ``append`` is thread-safe
(the FE cascades run concurrently): the name-to-slot table is updated under a lock and each column is written into its own slot.
"""

from __future__ import annotations

import os
import tempfile
import threading
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

_DEFAULT_CHUNK_COLUMNS = 64
_DEFAULT_MEMMAP_BYTES = 2 * 1024**3


def _remove_quietly(path: str) -> None:
    """Delete a memmap file; one still mapped on Windows is left to the temp cleaner."""
    try:
        os.remove(path)
    except OSError:
        pass


class ColumnArena:
    """Named 1-D columns of ``n_rows`` values stored contiguously per column, in chunks of ``chunk_columns`` columns."""

    def __init__(self, n_rows: int, dtype: Any = np.float32, chunk_columns: int = _DEFAULT_CHUNK_COLUMNS, memmap_bytes: int = _DEFAULT_MEMMAP_BYTES, memmap_dir: "str | None" = None) -> None:
        self.n_rows = int(n_rows)
        self.dtype = np.dtype(dtype)
        self.chunk_columns = max(1, int(chunk_columns))
        self.memmap_bytes = int(memmap_bytes)
        self._memmap_dir = memmap_dir
        self._chunks: list[np.ndarray] = []
        self._slot: dict[str, int] = {}
        self._lock = threading.Lock()
        self._files: list[str] = []

    def __len__(self) -> int:
        return len(self._slot)

    def __contains__(self, name: object) -> bool:
        return name in self._slot

    @property
    def names(self) -> list[str]:
        """Column names in insertion order."""
        return list(self._slot)

    @property
    def nbytes(self) -> int:
        """Bytes held by the allocated chunks (in RAM or on disk)."""
        return sum(int(c.nbytes) for c in self._chunks)

    def _new_chunk(self) -> np.ndarray:
        """A fresh ``(chunk_columns, n_rows)`` chunk: in RAM while the store is small, a temporary memmap file once the next chunk would pass ``memmap_bytes``."""
        shape = (self.chunk_columns, self.n_rows)
        if self.nbytes + int(np.prod(shape)) * self.dtype.itemsize > self.memmap_bytes:
            fd, path = tempfile.mkstemp(prefix="mlframe_arena_", suffix=".bin", dir=self._memmap_dir)
            os.close(fd)
            self._files.append(path)
            return np.memmap(path, dtype=self.dtype, mode="w+", shape=shape)
        return np.empty(shape, dtype=self.dtype)

    def append(self, name: str, values: np.ndarray) -> int:
        """Store ``values`` under ``name`` (cast to the arena dtype in the one copy) and return its slot; a name already present keeps its first column and returns that slot."""
        arr = np.asarray(values)
        if arr.shape != (self.n_rows,):
            raise ValueError(f"column {name!r} has shape {arr.shape}, the arena holds columns of {self.n_rows} rows")
        with self._lock:
            slot = self._slot.get(name)
            if slot is not None:
                return slot
            slot = len(self._slot)
            if slot // self.chunk_columns >= len(self._chunks):
                self._chunks.append(self._new_chunk())
            self._slot[name] = slot
            target = self._chunks[slot // self.chunk_columns][slot % self.chunk_columns]
        target[:] = arr  # outside the lock: the slot is this call's alone
        return slot

    def extend(self, columns: "dict[str, np.ndarray]") -> list[str]:
        """Append every column of ``columns`` and return the names that were new."""
        new = [n for n in columns if n not in self._slot]
        for name in columns:
            self.append(name, columns[name])
        return new

    def column(self, name: str) -> np.ndarray:
        """The stored column as a view (no copy)."""
        slot = self._slot[name]
        row: np.ndarray = self._chunks[slot // self.chunk_columns][slot % self.chunk_columns]
        return row

    def block(self, names: Sequence[str]) -> np.ndarray:
        """``(len(names), n_rows)`` C-contiguous array of the named columns (a copy; the columns of one chunk are gathered with one fancy index)."""
        out = np.empty((len(names), self.n_rows), dtype=self.dtype)
        for k, name in enumerate(names):
            out[k] = self.column(name)
        return out

    def blocks(self, names: "Iterable[str] | None" = None, budget_bytes: int = 256 * 1024 * 1024) -> "Iterator[tuple[list[str], np.ndarray]]":
        """Blocks of the named (default: all) columns within ``budget_bytes``, the same shape ``fe_column_blocks`` yields for a frame."""
        todo = list(self._slot if names is None else names)
        step = max(1, int(budget_bytes) // max(1, self.n_rows * self.dtype.itemsize))
        for c0 in range(0, len(todo), step):
            part = todo[c0 : c0 + step]
            yield part, self.block(part)

    def as_frame(self, names: "Sequence[str] | None" = None, index: Any = None) -> Any:
        """A pandas frame of the named (default: all) columns, built from one stacked block; the one place the arena is copied into a frame."""
        import pandas as pd

        cols = list(self._slot if names is None else names)
        return pd.DataFrame(self.block(cols).T, columns=cols, index=index if index is not None else pd.RangeIndex(self.n_rows), copy=False)

    def close(self) -> None:
        """Release the chunks and delete the memmap files."""
        self._chunks.clear()
        self._slot.clear()
        for path in self._files:
            _remove_quietly(path)
        self._files.clear()

    def __enter__(self) -> "ColumnArena":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["ColumnArena"]
