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

Observed in prod: ~20 s of QuantileDMatrix rebuild per
target × 4 targets = ~80 s wasted in one ensemble run alone, plus
hidden cross-target CB Pool / LGB Dataset rebuilds.

This module exposes a single ``compute_signature(X)`` that:

1. Reads ``columns``, ``shape``, ``dtypes`` (cheap, O(n_cols)).
2. Hashes an evenly-strided row sample (count scales with ``sqrt(n_rows)``,
   capped at 64) of X to disambiguate genuinely different content with
   identical column/shape signatures (eg. two ``.iloc`` slices into the
   same logical X carry identical shapes -- the row-sample distinguishes
   them when they cover different index ranges). A fixed 3-row sample
   (first/middle/last) previously collided whenever two structurally-
   similar-but-content-different frames happened to agree at exactly
   those 3 positions -- realistic in the composite-target-discovery /
   ensemble-refit regime this cache targets, where many near-identical
   row-wise-extension frames share a base X.
3. Returns a hashable tuple usable as a dict key.

The fingerprint stays sub-linear in n_rows (capped at 64 sampled rows
regardless of frame size): a 4M-row × 25-col frame still hashes in
microseconds. Near-perfect cache hit on identical logical data
(different Python ids); much lower collision odds on actually-different
content than the old fixed-3-row scheme.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_MAX_SAMPLE_ROWS = 64


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
    cols = tuple(str(c) for c in X.columns) if hasattr(X, "columns") else None
    shape = getattr(X, "shape", (None, None))
    n_rows = int(shape[0]) if shape and shape[0] is not None else None
    n_cols = int(shape[1]) if shape and len(shape) > 1 and shape[1] is not None else None
    content_hash: object = _row_sample_hash(X, n_rows)
    if content_hash is None:
        # The row sample could not be hashed, so this key would say nothing about CONTENT: two different frames of the
        # same columns and shape would share it and one would be served the other's cached dataset. A fresh sentinel
        # makes the key match nothing - a cache miss, which is the safe answer to "cannot tell".
        content_hash = ("unhashable", object())
    # dtypes, as step 1 of the module docstring always said: a tier transition that recasts the same columns
    # float64 -> float32 (or int -> category) otherwise produces an identical key, and the booster caches hand back
    # a dataset built from the old dtypes.
    return (cols, n_rows, n_cols, _dtype_signature(X), content_hash, extra)


_POLARS_DTYPE_CANON = {
    "Float64": "f8", "Float32": "f4", "Int64": "i8", "Int32": "i4", "Int16": "i2", "Int8": "i1",
    "UInt64": "u8", "UInt32": "u4", "UInt16": "u2", "UInt8": "u1", "Boolean": "b1",
    "String": "O", "Utf8": "O", "Object": "O", "Categorical": "category", "Enum": "category",
}


def _canonical_dtype(d: Any) -> str:
    """A container-agnostic dtype name, so the same logical column keys the same from pandas, polars and numpy."""
    name = str(d)
    if name in _POLARS_DTYPE_CANON:
        return _POLARS_DTYPE_CANON[name]
    base = name.split("(", 1)[0]  # polars parametrised types: Datetime(time_unit='us'), Enum(categories=[...])
    if base in _POLARS_DTYPE_CANON:
        return _POLARS_DTYPE_CANON[base]
    if name == "category":
        return "category"
    if name in ("str", "string", "object") or name.startswith("string["):  # pandas string dtypes (pandas 3 default "str")
        return "O"
    try:
        nd = np.dtype(d)
        return "O" if nd.kind in "OUS" else f"{nd.kind}{nd.itemsize}"
    except TypeError:
        return name


def _dtype_signature(X: Any):
    """Per-column canonical dtypes (pandas / polars), the array dtype for an ndarray, or None when neither is known."""
    try:
        dtypes = getattr(X, "dtypes", None)
        if dtypes is None:
            dtype = getattr(X, "dtype", None)  # ndarray
            return _canonical_dtype(dtype) if dtype is not None else None
        if hasattr(dtypes, "tolist"):  # pandas Series of dtypes
            return tuple(_canonical_dtype(d) for d in dtypes.tolist())
        if isinstance(dtypes, (list, tuple)):  # polars
            return tuple(_canonical_dtype(d) for d in dtypes)
        return _canonical_dtype(dtypes)
    except Exception:  # a dtype probe must never break the key; it only sharpens it
        return None


def _canonicalise_row(row_values: Any) -> tuple:
    """Coerce a row of values to a canonical, container-agnostic
    Python tuple. Same logical values from pandas vs polars vs numpy
    MUST produce equal output; otherwise the module-level cache
    misses across the booster shims even though X is the same logical
    frame (observed in prod: pl-vs-pd dtype-handling asymmetry
    silently invalidated the XGB DMatrix cache across composite
    targets, costing ~60 s per ensemble run).
    """
    out: list = []
    for v in row_values:
        if v is None:
            out.append("__null__")
            continue
        try:
            if isinstance(v, float) and v != v:
                out.append("__nan__")
                continue
        except Exception as e:  # nosec B110 - best-effort path
            logger.debug("NaN-sentinel check failed for value %r: %s", v, e)
        # numpy scalars: extract Python value to drop dtype carriage.
        if hasattr(v, "item") and callable(v.item):
            try:
                py = v.item()
                if isinstance(py, float) and py != py:
                    out.append("__nan__")
                    continue
                out.append(py)
                continue
            except (ValueError, TypeError):
                pass
        out.append(v)
    return tuple(out)


def _sample_indices(n_rows: int) -> tuple:
    """Evenly-strided row indices covering ``[0, n_rows-1]``, count scaling with
    ``sqrt(n_rows)`` (min 3, capped at :data:`_MAX_SAMPLE_ROWS`) so the fingerprint stays
    sub-linear while sampling far more of the frame than a fixed first/middle/last triple."""
    n_samples = min(_MAX_SAMPLE_ROWS, max(3, int(math.sqrt(n_rows))))
    if n_samples >= n_rows:
        return tuple(range(n_rows))
    # round(i * (n_rows-1) / (n_samples-1)) evenly spans [0, n_rows-1]; dict.fromkeys
    # dedupes any indices that round to the same position while preserving order.
    return tuple(dict.fromkeys(round(i * (n_rows - 1) / (n_samples - 1)) for i in range(n_samples)))


def _row_sample_hash(X: Any, n_rows: int | None) -> int | None:
    """Hash an evenly-strided row sample of ``X`` (see :func:`_sample_indices`).

    Returns ``None`` when sampling fails (unknown DataFrame variant,
    no rows, hashable-row coercion failure). A None hash collapses to
    the column/shape-only key, which is still safer than ``id(X)``
    because at least content-different frames with identical shapes
    don't collide.

    Critical invariant: pandas and polars containers carrying
    LOGICALLY-EQUAL data must produce IDENTICAL hashes -- every branch
    feeds rows through :func:`_canonicalise_row` so dtype carriage
    (numpy scalar vs Python float, pl.Categorical vs pd.category) is
    stripped before hashing.
    """
    if n_rows is None or n_rows < 1:
        return None
    indices = _sample_indices(n_rows)
    samples: list = []
    # Polars fast path FIRST: many polars frames also expose ``iloc``
    # via newer compat layers, but we want the O(n_cols) ``row()`` call
    # not the n_rows-materialising ``iloc``. Detect polars by the
    # (.row, .slice) method pair, which pandas does not expose.
    if hasattr(X, "row") and hasattr(X, "slice") and callable(getattr(X, "row", None)):
        try:
            samples = [_canonicalise_row(X.row(idx)) for idx in indices if 0 <= idx < n_rows]
            return hash(tuple(samples))
        except Exception as exc:
            logger.debug("_row_sample_hash: polars .row() sampling failed, cache key falls back to id()-based: %s", exc)
            return None
    if hasattr(X, "iloc"):
        try:
            for idx in indices:
                if 0 <= idx < n_rows:
                    row = X.iloc[idx]
                    try:
                        vals = row.tolist()
                    except AttributeError:
                        vals = list(row)
                    samples.append(_canonicalise_row(vals))
            return hash(tuple(samples))
        except Exception as exc:
            logger.debug("_row_sample_hash: pandas .iloc sampling failed, cache key falls back to id()-based: %s", exc)
            return None
    if hasattr(X, "to_numpy"):
        try:
            arr = X.to_numpy()
            for idx in indices:
                if 0 <= idx < n_rows:
                    row = arr[idx]
                    if hasattr(row, "tolist"):
                        row = row.tolist()
                    samples.append(_canonicalise_row(row))
            return hash(tuple(samples))
        except Exception as exc:
            logger.debug("_row_sample_hash: to_numpy() sampling failed, cache key falls back to id()-based: %s", exc)
            return None
    if hasattr(X, "__getitem__") and hasattr(X, "ndim"):
        try:
            arr = np.asarray(X)
            for idx in indices:
                if 0 <= idx < n_rows:
                    row = arr[idx]
                    if isinstance(row, np.ndarray):
                        samples.append(_canonicalise_row(row.tolist()))
                    else:
                        samples.append(_canonicalise_row((row,)))
            return hash(tuple(samples))
        except Exception as exc:
            logger.debug("_row_sample_hash: np.asarray() sampling failed, cache key falls back to id()-based: %s", exc)
            return None
    return None
