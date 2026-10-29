"""Chunk codes that stay on the device until the pair family gate asks for a few of their columns.

A pair-search chunk bins an (n, K) block of candidate columns on the device. The pair family gate reads the codes of every configuration of a pair, but only for
the pairs that clear the cheap table floor, and only those pairs' columns. Copying the whole block back for every chunk moved the full (n, K) matrix per chunk
(12 GB over the pair-search audit) to serve a handful of reads. ``ChunkCodes`` keeps the device array and the registry's "host filled" flag; a ``LazyCodeColumn`` is
one column of it, and ``resolve_code_columns`` copies the columns a gate needs in one transfer per chunk.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np


class ChunkCodes:
    """The codes of one pair-search chunk: a host buffer that may still be unfilled, and the device array it is filled from."""

    __slots__ = ("_dev", "_filled", "dtype", "host", "shape")

    ndim = 2  # a chunk's codes are always an (n, K) block

    def __init__(self, host: np.ndarray, dev: Any, filled: list) -> None:
        self.host = host
        self._dev = dev
        self._filled = filled
        self.shape = tuple(host.shape)
        self.dtype = host.dtype

    @classmethod
    def capture(cls, host: np.ndarray) -> "ChunkCodes | np.ndarray":
        """A ``ChunkCodes`` for a host buffer whose fill is still deferred, otherwise ``host`` itself (already filled, nothing to defer)."""
        from .._gpu_resident_fe import capture_deferred_codes, ensure_host_codes_filled

        if os.environ.get("MLFRAME_PAIR_LAZY_CODES", "1") == "0":
            ensure_host_codes_filled(host)
            return host
        got = capture_deferred_codes(host)
        if got is None or got[1][0]:
            return host
        return cls(host, got[0], got[1])

    def host_ready(self) -> bool:
        """Whether the host buffer holds the codes (filled by some reader)."""
        return bool(self._filled[0])

    def full_host(self) -> np.ndarray:
        """The whole host block, copied from the device first if no reader has filled it."""
        if not self._filled[0]:
            self._dev.get(out=self.host)
            self._filled[0] = True
        return self.host

    def columns(self, idx: Sequence[int]) -> np.ndarray:
        """Host array (n, len(idx)) of the codes at columns ``idx``; copies only those columns when the host buffer is unfilled."""
        if self._filled[0]:
            return np.ascontiguousarray(self.host[:, list(idx)])
        import cupy as cp

        return np.ascontiguousarray(cp.asnumpy(self._dev[:, cp.asarray(list(idx), dtype=cp.int64)]))

    def __getitem__(self, key: Any) -> Any:
        """Any other indexing reads the filled host block."""
        return self.full_host()[key]

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        """numpy conversion: the filled host block."""
        h = self.full_host()
        return h if dtype is None else h.astype(dtype, copy=False)


class LazyCodeColumn:
    """One (n, 1) column of a ``ChunkCodes``; materialised by ``resolve_code_columns`` or, for a stray reader, by ``np.asarray``."""

    __slots__ = ("chunk", "dtype", "j", "shape")

    def __init__(self, chunk: ChunkCodes, j: int) -> None:
        self.chunk = chunk
        self.j = int(j)
        self.shape = (chunk.shape[0], 1)
        self.dtype = chunk.dtype

    @property
    def ndim(self) -> int:
        """Always 2."""
        return 2

    @property
    def dev(self) -> Any:
        """The (n, 1) device view of this column, or None when the chunk has no device copy; the device family gate scores batches that carry one without copying them back."""
        d = self.chunk._dev
        return None if d is None else d[:, self.j : self.j + 1]

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        """numpy conversion: a one-column copy."""
        a = self.chunk.columns([self.j])
        return a if dtype is None else a.astype(dtype, copy=False)


def code_column(chunk_codes: Any, ci: int) -> Any:
    """The (n, 1) codes of buffer column ``ci``: a lazy reference for a deferred chunk, the host slice otherwise."""
    if isinstance(chunk_codes, ChunkCodes):
        return LazyCodeColumn(chunk_codes, ci)
    return np.asarray(chunk_codes[:, ci : ci + 1]).reshape(-1, 1)


def resolve_code_columns(items: Sequence[Any]) -> "list[Any]":
    """``items`` with each ``LazyCodeColumn`` replaced by its host array, in order; every other item is passed through untouched; the lazy columns of one chunk are copied in a single transfer."""
    out: "list[Any]" = [None] * len(items)
    by_chunk: "dict[int, tuple[ChunkCodes, list[tuple[int, int]]]]" = {}
    for pos, it in enumerate(items):
        if isinstance(it, LazyCodeColumn):
            by_chunk.setdefault(id(it.chunk), (it.chunk, []))[1].append((pos, it.j))
        else:
            out[pos] = it  # an array or a LazyHostCodes: the gate takes both, and a LazyHostCodes keeps its device array for the device path
    for chunk, picks in by_chunk.values():
        block = chunk.columns([j for _, j in picks])
        for k, (pos, _) in enumerate(picks):
            out[pos] = np.ascontiguousarray(block[:, k : k + 1])
    return out
