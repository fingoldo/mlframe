"""Content-keyed memo for the univariate basis expansion, shared by every scorer of one fit.

Carved out of the package ``__init__`` for file size; the package re-exports every name defined here.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable

import numpy as np
import pandas as pd

from .._cache_release import register_cache_releaser

logger = logging.getLogger(__name__)

_BASIS_MEMO_LOCK = threading.Lock()
_BASIS_MEMO: "dict[tuple, pd.DataFrame]" = {}  # content key -> the generated columns; one fit asks for the same expansion of the same columns from many scorers
_BASIS_MEMO_MAX_ENTRIES = 6
_BASIS_MEMO_MAX_BYTES = 768 * 1024 * 1024


def _release_basis_memo() -> int:
    """Drop the memoised basis expansions (registered for release under memory pressure); returns the bytes freed."""
    with _BASIS_MEMO_LOCK:
        freed = sum(int(df.memory_usage(index=False).sum()) for df in _BASIS_MEMO.values())
        _BASIS_MEMO.clear()
    return freed


register_cache_releaser(_release_basis_memo)


def _basis_memo_key(X: pd.DataFrame, cols: Any, degrees: Any, basis: str, dedup: bool, thr: float, y: Any, routing: str) -> "tuple | None":
    """Content key of one call: the numeric source columns it can read, ``y`` and every parameter (``None`` when the inputs cannot be hashed)."""
    import hashlib

    new_hash: Callable[..., Any]
    try:
        import xxhash

        new_hash = xxhash.xxh3_128  # several times faster than blake2b on the ~1 GB a 100000 x 600 frame hashes per call
    except ImportError:

        def new_hash(data: Any = b"") -> Any:
            """blake2b stand-in when xxhash is not installed."""
            return hashlib.blake2b(data, digest_size=16)

    try:
        names = list(cols) if cols is not None else [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        h = new_hash()
        for c in names:
            h.update(str(c).encode())
            arr = np.ascontiguousarray(X[c].to_numpy())
            if arr.dtype == object or arr.dtype.kind not in "biufc":
                # A categorical / string / object column has no byte view (it made the whole key fail, so a call whose ``cols`` listed one never reached the memo): hash its values.
                h.update(np.ascontiguousarray(pd.util.hash_pandas_object(X[c], index=False).to_numpy()).view(np.uint8).data)
            else:
                h.update(arr.view(np.uint8).data)
        hy = None
        if y is not None:
            hy = new_hash(np.ascontiguousarray(np.asarray(y)).view(np.uint8).data).hexdigest()
        return (h.hexdigest(), tuple(map(str, names)), tuple(int(d) for d in degrees), str(basis), bool(dedup), float(thr), hy, str(routing), len(X))
    except Exception as exc:
        logger.debug("basis memo key failed (%s); computing without the memo", exc)
        return None
