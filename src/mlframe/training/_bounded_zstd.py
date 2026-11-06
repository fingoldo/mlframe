"""Output-size guards for zstd-compressed pickle bundles, so a tiny hostile file cannot expand into an unbounded allocation."""

from __future__ import annotations

import os
from typing import Any

from mlframe._optional_imports import import_optional
import logging

logger = logging.getLogger(__name__)

MAX_DECOMPRESSED_ENV = "MLFRAME_MAX_DECOMPRESSED_BYTES"
_CAP_CEILING_BYTES = 64 * 1024**3


def _default_max_decompressed_bytes() -> int:
    """Half of physical RAM, at most 64 GiB: a bundle that cannot fit in memory must not get a buffer the host cannot back."""
    try:
        import psutil

        total = int(psutil.virtual_memory().total)
    except Exception:  # pragma: no cover - psutil is a hard dependency; keep the ceiling if the probe fails
        return _CAP_CEILING_BYTES
    return min(_CAP_CEILING_BYTES, total // 2)


def max_decompressed_bytes() -> int:
    """Decompressed-size ceiling in bytes: ``MLFRAME_MAX_DECOMPRESSED_BYTES`` when it parses to a positive integer, else half of RAM (at most 64 GiB). 0 or a negative value disables the cap."""
    raw = os.environ.get(MAX_DECOMPRESSED_ENV)
    if raw is None or not raw.strip():
        return _default_max_decompressed_bytes()
    try:
        value = int(raw)
    except ValueError as _exc:
        logger.warning("max_decompressed_bytes fell back to its default after %s", type(_exc).__name__)
        return _default_max_decompressed_bytes()
    return value if value > 0 else 0


class DecompressedSizeError(OSError):
    """The decompressed stream exceeded the configured ceiling."""


class BoundedReader:
    """Read-through wrapper over a zstd stream reader that raises ``DecompressedSizeError`` once more than ``limit`` bytes were produced.
    Exposes the ``read``/``readline``/``readinto`` trio the pickle unpicklers use. A ``limit`` of 0 disables the cap."""

    def __init__(self, raw: Any, limit: int) -> None:
        self._raw = raw
        self._limit = limit
        self._count = 0

    def _account(self, n: int) -> None:
        """Add ``n`` produced bytes and enforce the ceiling."""
        self._count += n
        if self._limit and self._count > self._limit:
            raise DecompressedSizeError(f"decompressed stream exceeds {self._limit} bytes ({MAX_DECOMPRESSED_ENV})")

    def read(self, size: int = -1) -> bytes:
        """Read up to ``size`` bytes (all remaining when negative), counting them."""
        if size is None or size < 0:
            if self._limit:
                data = self._raw.read(self._limit - self._count + 1)
            else:
                data = self._raw.read()
        else:
            data = self._raw.read(size)
        self._account(len(data))
        return bytes(data)

    def readline(self, size: int = -1) -> bytes:
        """Read one line, counting it."""
        data = self._raw.readline(size)
        self._account(len(data))
        return bytes(data)

    def readinto(self, buf: Any) -> int:
        """Fill ``buf`` from the stream, counting the bytes produced."""
        n = int(self._raw.readinto(buf))
        self._account(n)
        return n


def decompress_bounded(data: bytes) -> bytes:
    """Decompress a single zstd frame, refusing frames whose output would exceed the ceiling."""
    zstd = import_optional("zstandard", "db", "reading compressed model bundles")
    limit = max_decompressed_bytes()
    dctx = zstd.ZstdDecompressor()
    if not limit:
        return bytes(dctx.decompress(data))
    declared = zstd.frame_content_size(data)
    if declared >= 0 and declared != zstd.CONTENTSIZE_UNKNOWN and declared > limit:
        raise DecompressedSizeError(f"zstd frame declares {declared} decompressed bytes, over the {limit} ceiling ({MAX_DECOMPRESSED_ENV})")
    return bytes(dctx.decompress(data, max_output_size=limit))
