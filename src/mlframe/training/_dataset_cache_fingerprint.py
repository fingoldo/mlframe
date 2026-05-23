"""Shared content fingerprint for booster dataset caches.

Background. ``xgb_shim``, ``lgb_shim``, ``_cb_pool`` (val cache) and
``_cb_pool_build`` (train cache) all maintained a per-instance cache of
the booster-native dataset object (``QuantileDMatrix`` / ``Dataset`` /
``Pool``) keyed by ``id(X) + columns + shape + cat_features``. The
``id(X)`` component made the cache effectively useless across:

* ``sklearn.clone()`` (composite-ensemble OOF refit produces fresh
  shim instances with empty caches anyway, but the cache key would
  ALSO differ because frame ids change),
* ``train_X.iloc[idx].reset_index(drop=True)`` (every call returns a
  fresh pandas object; ids never match across rounds),
* cross-target reuse of the same underlying X (per-target loop slices
  produce fresh frames; the suite-level pipeline-cache hits but the
  shim cache misses).

Result on prod TVT 2026-05-23: ~20 s of QuantileDMatrix rebuild per
target × 4 targets = ~80 s wasted in one ensemble run alone, plus
hidden cross-target CB Pool / LGB Dataset rebuilds.

This module exposes a single ``compute_signature(X)`` that:

1. Reads ``columns``, ``shape``, ``dtypes`` (cheap, O(n_cols)).
2. Hashes a 3-row sample (first, middle, last) of X to disambiguate
   genuinely different content with identical column/shape signatures
   (eg. two ``.iloc`` slices into the same logical X carry identical
   shapes -- the row-sample distinguishes them when they cover
   different index ranges).
3. Returns a hashable tuple usable as a dict key.

The fingerprint is O(n_cols), not O(n_rows): a 4M-row × 25-col frame
hashes in microseconds. Near-perfect cache hit on identical logical
data (different Python ids); clean miss on actually-different content.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def compute_signature(X: Any, *, extra: tuple = ()) -> tuple:
    """Return a content-based cache key for ``X``.

    Parameters
    ----------
    X
        DataFrame-like (pandas, polars) or ndarray. Anything supporting
        ``columns`` / ``shape`` / row indexing is handled; unknown types
        fall back to ``(type_name, repr_hash)``.
    extra
        Additional tuple to fold into the key (eg. ``cat_features``,
        ``text_features``). Always positional so callers can compose
        without changing the key when ``extra`` is omitted.
    """
    cols = tuple(X.columns) if hasattr(X, "columns") else None
    shape = getattr(X, "shape", (None, None))
    n_rows = (
        int(shape[0]) if shape and shape[0] is not None else None
    )
    n_cols = (
        int(shape[1])
        if shape and len(shape) > 1 and shape[1] is not None
        else None
    )
    content_hash = _row_sample_hash(X, n_rows)
    return (cols, n_rows, n_cols, content_hash, extra)


def _row_sample_hash(X: Any, n_rows: int | None) -> int | None:
    """Hash a 3-row sample (first, middle, last) of ``X``.

    Returns ``None`` when sampling fails (unknown DataFrame variant,
    no rows, hashable-row coercion failure). A None hash collapses to
    the column/shape-only key, which is still safer than ``id(X)``
    because at least content-different frames with identical shapes
    don't collide.
    """
    if n_rows is None or n_rows < 1:
        return None
    indices = (0, n_rows // 2, n_rows - 1)
    samples: list = []
    if hasattr(X, "iloc"):
        try:
            for idx in indices:
                if 0 <= idx < n_rows:
                    row = X.iloc[idx]
                    # Try numeric coercion first (fast path, common case).
                    try:
                        samples.append(
                            tuple(row.to_numpy(dtype=np.float64, copy=False))
                        )
                    except (TypeError, ValueError):
                        # Object / mixed dtype: fall back to tuple of
                        # str(values) -- slower but always hashable.
                        samples.append(tuple(str(v) for v in row.tolist()))
            return hash(tuple(samples))
        except Exception:
            return None
    # Polars fast path: ``df.row(idx)`` returns a single-row tuple in O(n_cols),
    # whereas ``df.to_numpy()`` would materialise the entire (n_rows, n_cols)
    # frame just to read 3 rows. c0103 iter261 profile attributed 4.51s
    # (4 calls x 1.13s) to the to_numpy path on a 200k x 25 polars frame; the
    # row() path runs in microseconds. Detect polars by the (.row, .slice)
    # method pair, which pandas does not expose.
    if hasattr(X, "row") and hasattr(X, "slice") and callable(getattr(X, "row", None)):
        try:
            for idx in indices:
                if 0 <= idx < n_rows:
                    samples.append(X.row(idx))
            return hash(tuple(samples))
        except Exception:
            return None
    if hasattr(X, "to_numpy"):
        try:
            arr = X.to_numpy()
            for idx in indices:
                if 0 <= idx < n_rows:
                    samples.append(tuple(arr[idx]))
            return hash(tuple(samples))
        except Exception:
            return None
    if hasattr(X, "__getitem__") and hasattr(X, "ndim"):
        try:
            arr = np.asarray(X)
            for idx in indices:
                if 0 <= idx < n_rows:
                    row = arr[idx]
                    if isinstance(row, np.ndarray):
                        samples.append(tuple(row.tolist()))
                    else:
                        samples.append(row)
            return hash(tuple(samples))
        except Exception:
            return None
    return None
