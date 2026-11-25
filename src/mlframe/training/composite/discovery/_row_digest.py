"""Collision-safe digest of a row-index array, shared by the fold caches that key on it."""

from __future__ import annotations

import numpy as np


def rows_digest(rows: np.ndarray) -> bytes:
    """Collision-safe cache-key digest of a row-index array (a 64-bit ``hash()`` could return another fold's entry; audit 34-F27)."""
    import hashlib

    return hashlib.blake2b(np.ascontiguousarray(rows).view(np.uint8).data, digest_size=16).digest()
