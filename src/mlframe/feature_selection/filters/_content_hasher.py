"""One content hasher for in-process memo keys over large arrays: xxh3-128 when ``xxhash`` is installed (several GB/s), else blake2b-128 (about 1 GB/s).

Both objects take ``update(buffer)`` and give ``digest()`` / ``hexdigest()``; the keys live only inside one process, so the algorithm may differ between machines.
"""

from __future__ import annotations

import hashlib
from typing import Any


def new_content_hasher() -> Any:
    """A fresh 128-bit incremental hasher (xxh3 if available, else blake2b)."""
    try:
        import xxhash

        return xxhash.xxh3_128()
    except ImportError:  # best-effort: blake2b is an equally strong 128-bit hash, only slower
        return hashlib.blake2b(digest_size=16)
