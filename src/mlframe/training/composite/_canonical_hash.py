"""Library-version-independent canonical encoding of frame columns for the discovery-cache key.

``data_signature`` is a cache KEY written by one run and looked up by the next, possibly under a different pandas / polars /
numpy. Anything the key reads from a library's representation (``str(dtype)``, ``hash_pandas_object``, polars ``hash_rows``,
the inferred datetime resolution, the platform default int width) moves the key without the data moving, and every cached
entry goes permanently cold. This module turns columns into bytes that depend only on the logical data.

Canonicalisation rules (the contract; changing any of them moves every key):

* Logical type tokens: ``bool``, ``int`` (every signed/unsigned width and pandas nullable / arrow ints), ``float16`` /
  ``float32`` / ``float64`` (widths stay distinct: float32 genuinely holds different values), ``string`` (numpy object
  holding str, ``string[python]``, ``string[pyarrow]``, pandas default ``str``, polars String), ``category`` (pandas
  Categorical, polars Categorical / Enum), ``datetime[<tz>|naive]``, ``timedelta``, ``date``, ``time``, ``decimal``,
  ``object`` (pandas object column holding non-str values), ``other`` (nested / exotic types).
* Nullable vs numpy dtypes of the same logical type share a token; a pandas nullable column without nulls keys exactly like
  its numpy twin.
* Nulls: float NaN, pandas NA / NaT / None and polars null are all "null". They are recorded in a validity bitmap and the
  value slot is zeroed, so NaN-vs-NA storage never reaches the key. ``-0.0`` folds to ``0.0``.
* Numerics: fixed-width little-endian (ints as 64-bit, floats at their own width).
* Temporals: integer nanoseconds (UTC for tz-aware datetimes), whatever the stored resolution; ``date`` as days.
* Text: UTF-8 with a presence byte and an 8-byte length prefix per value; categoricals by their string value.
* Stats scalars: floats as ``float.hex`` (exact), ints / strings via their Python text.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._composite_utils import is_polars_df as _is_polars_df

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_NAT = np.iinfo(np.int64).min
_UNIT_TO_NS = {"ns": 1, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000}
_FLOAT_TOKENS = {"float16": np.dtype("<f2"), "float32": np.dtype("<f4"), "float64": np.dtype("<f8")}
_TEXT_TOKENS = ("string", "category", "object", "decimal", "other")


def _is_temporal(token: str) -> bool:
    """True for the date / time / datetime / timedelta logical type tokens."""
    return token.startswith("datetime") or token in ("timedelta", "date", "time")


# ----------------------------------------------------------------------------------------------
# Logical type tokens
# ----------------------------------------------------------------------------------------------


def _numpy_numeric_token(dt: np.dtype) -> Optional[str]:
    """Token for a numpy bool/int/float dtype, None for anything else."""
    kind = dt.kind
    if kind == "b":
        return "bool"
    if kind in ("i", "u"):
        return "int"
    if kind == "f":
        return f"float{dt.itemsize * 8}"
    return None


def pandas_logical_type(s: pd.Series, probe: Optional[pd.Series] = None) -> str:
    """Canonical logical type of a pandas column. *probe* (a small slice) decides whether a numpy object column holds strings;
    only a bounded slice is inspected so the token stays O(1) on huge frames."""
    dt = s.dtype
    if isinstance(dt, pd.CategoricalDtype):
        return "category"
    if isinstance(dt, pd.StringDtype):
        return "string"
    if isinstance(dt, pd.DatetimeTZDtype):
        return f"datetime[{dt.tz}]"
    if isinstance(dt, pd.ArrowDtype):
        import pyarrow as pa

        pat = dt.pyarrow_dtype
        if pa.types.is_string(pat) or pa.types.is_large_string(pat) or getattr(pa.types, "is_string_view", lambda _t: False)(pat):
            return "string"
        if pa.types.is_boolean(pat) or pa.types.is_integer(pat) or pa.types.is_floating(pat):
            tok = _numpy_numeric_token(np.dtype(dt.numpy_dtype))
            return tok if tok is not None else "other"
        return "other"
    if isinstance(dt, np.dtype):
        tok = _numpy_numeric_token(dt)
        if tok is not None:
            return tok
        if dt.kind == "M":
            return "datetime[naive]"
        if dt.kind == "m":
            return "timedelta"
        if dt.kind in ("U", "S"):
            return "string"
        if dt.kind == "O":
            vals = (probe if probe is not None else s).dropna().tolist()
            return "string" if all(isinstance(v, str) for v in vals) else "object"
        return "other"
    # pandas masked extension dtypes (Int64, UInt8, Float64, boolean) carry the numpy dtype they mirror.
    np_dt = getattr(dt, "numpy_dtype", None)
    if np_dt is not None:
        tok = _numpy_numeric_token(np.dtype(np_dt))
        if tok is not None:
            return tok
    return "other"


def polars_logical_type(dt: Any) -> str:
    """Canonical logical type of a polars dtype (never ``str(dtype)``, whose repr changes across polars versions)."""
    if dt == pl.Boolean:
        return "bool"
    if dt.is_integer():
        return "int"
    if dt == pl.Float32:
        return "float32"
    if dt == pl.Float64:
        return "float64"
    if dt == pl.String:
        return "string"
    if isinstance(dt, (pl.Categorical, pl.Enum)) or dt in (pl.Categorical, pl.Enum):
        return "category"
    if isinstance(dt, pl.Datetime):
        return f"datetime[{dt.time_zone}]" if dt.time_zone else "datetime[naive]"
    if isinstance(dt, pl.Duration) or dt == pl.Duration:
        return "timedelta"
    if dt == pl.Date:
        return "date"
    if dt == pl.Time:
        return "time"
    if isinstance(dt, pl.Decimal) or dt == pl.Decimal:
        return "decimal"
    return "other"


def _numeric_np_dtype(token: str, src: np.dtype | None = None) -> np.dtype:
    """Canonical little-endian numpy dtype a numeric token is encoded as (uint64 kept unsigned so it does not wrap)."""
    if token == "bool":
        return np.dtype(np.bool_)
    if token == "int":
        # uint64 cannot go through int64 without wrapping; its bytes equal int64's for every value both can hold.
        return np.dtype("<u8") if src is not None and src.kind == "u" and src.itemsize == 8 else np.dtype("<i8")
    return _FLOAT_TOKENS[token]


# ----------------------------------------------------------------------------------------------
# Scalar / slice encoders
# ----------------------------------------------------------------------------------------------


def canonical_scalar(v: Any) -> str:
    """Stable text form of a stats scalar: floats exact via ``float.hex``, ints and strings via their Python text."""
    if v is None:
        return "~"
    if isinstance(v, (bool, np.bool_)):
        return f"i{int(v)}"
    if isinstance(v, (int, np.integer)):
        return f"i{int(v)}"
    if isinstance(v, (float, np.floating)):
        f = float(v) + 0.0
        return "~" if f != f else f"f{f.hex()}"
    s = v if isinstance(v, str) else str(v)
    return f"s{len(s)}:{s}"


def format_stats(mn: Any, mx: Any, n_null: int) -> bytes:
    """Canonical bytes for a column's (min, max, null count) summary."""
    return f"min={canonical_scalar(mn)};max={canonical_scalar(mx)};null={int(n_null)}".encode()


def _encode_numeric(vals: np.ndarray, mask: np.ndarray, np_dt: np.dtype) -> bytes:
    """Null bitmap plus the values cast to ``np_dt`` with nulls zeroed and -0.0 folded into 0.0."""
    out = np.array(vals, dtype=np_dt, copy=True)
    if mask.any():
        out[mask] = 0
    if np_dt.kind == "f":
        out = out + np_dt.type(0.0)
    return b"N" + np.packbits(mask).tobytes() + b"|" + np.ascontiguousarray(out).tobytes()


def _encode_ints(vals: Sequence[int], mask: np.ndarray, factor: int) -> bytes:
    # Python ints: scaling a seconds-resolution value to ns can exceed int64, so widen to 16 bytes instead of wrapping.
    """Null bitmap plus each value scaled by ``factor`` as a 16-byte signed integer (wide enough not to overflow at ns)."""
    parts = [b"T", np.packbits(mask).tobytes(), b"|"]
    for v, m in zip(vals, mask):
        parts.append((0 if m else int(v) * factor).to_bytes(16, "little", signed=True))
    return b"".join(parts)


def _encode_text(vals: Sequence[Any], mask: np.ndarray) -> bytes:
    """Length-prefixed UTF-8 of each value's string form, with a distinct marker for null."""
    parts = [b"S"]
    for v, m in zip(vals, mask):
        if m:
            parts.append(b"\x00")
        else:
            b = (v if isinstance(v, str) else str(v)).encode("utf-8")
            parts.append(b"\x01" + len(b).to_bytes(8, "little") + b)
    return b"".join(parts)


def _pandas_temporal_i8(s: pd.Series) -> Tuple[np.ndarray, int]:
    """(int64 values in the stored unit, ns factor) for a pandas datetime/timedelta column; a view, no copy."""
    arr = s.array
    return np.asarray(arr.asi8), _UNIT_TO_NS[getattr(arr, "unit", "ns")]


def encode_pandas_slice(small: pd.Series, token: str) -> bytes:
    """Canonical bytes of a SMALL pandas slice (head/tail window or sampled rows)."""
    if token in ("bool", "int") or token in _FLOAT_TOKENS:
        dt = small.dtype
        mask = np.asarray(small.isna().to_numpy(), dtype=bool)
        if isinstance(dt, np.dtype):
            vals = small.to_numpy()
        else:
            vals = small.to_numpy(dtype=_numeric_np_dtype(token).newbyteorder("="), na_value=0)
        src = dt if isinstance(dt, np.dtype) else np.dtype(getattr(dt, "numpy_dtype", np.int64))
        return _encode_numeric(vals, mask, _numeric_np_dtype(token, src))
    if _is_temporal(token):
        i8, factor = _pandas_temporal_i8(small)
        return _encode_ints(i8.tolist(), i8 == _NAT, factor)
    mask = np.asarray(small.isna().to_numpy(), dtype=bool)
    return _encode_text(small.tolist(), mask)


def encode_polars_slice(small: Any, token: str) -> bytes:
    """Canonical bytes of a SMALL polars slice; byte-identical to ``encode_pandas_slice`` for the same logical data."""
    if token in ("bool", "int") or token in _FLOAT_TOKENS:
        mask = small.is_null().to_numpy()
        if token in _FLOAT_TOKENS:
            mask = mask | small.is_nan().fill_null(False).to_numpy()
            vals = small.fill_null(0.0).to_numpy()
        else:
            vals = small.fill_null(False if token == "bool" else 0).to_numpy()
        src = np.dtype("<u8") if small.dtype == pl.UInt64 else None
        return _encode_numeric(vals, np.asarray(mask, dtype=bool), _numeric_np_dtype(token, src))
    if _is_temporal(token):
        mask = np.asarray(small.is_null().to_numpy(), dtype=bool)
        return _encode_ints(small.cast(pl.Int64).fill_null(0).to_list(), mask, _polars_ns_factor(small.dtype))
    mask = np.asarray(small.is_null().to_numpy(), dtype=bool)
    if token == "category":
        small = small.cast(pl.String)
    return _encode_text(small.to_list(), mask)


def _polars_ns_factor(dt: Any) -> int:
    """Multiplier that converts a polars temporal dtype's physical integers to the canonical resolution."""
    if dt == pl.Date:
        return 1
    if dt == pl.Time:
        return 1  # polars Time is always ns since midnight.
    unit = getattr(dt, "time_unit", None)
    return _UNIT_TO_NS[unit if unit is not None else "ns"]


# ----------------------------------------------------------------------------------------------
# Whole-column stats
# ----------------------------------------------------------------------------------------------

try:
    import numba as _numba
except ImportError:  # pragma: no cover
    _numba = None

_STATS_NUMBA_MIN_N: int = 50_000

if _numba is not None:

    @_numba.njit(cache=True, fastmath=False)
    def _float_stats_nan_null_kernel(arr):
        """Single pass (min, max, NaN count, valid count). Only NaN is null: inf is a value, matching polars' min/max after
        ``fill_nan(None)`` so both frame types agree."""
        n_null = 0
        n_valid = 0
        mn = np.inf
        mx = -np.inf
        for i in range(arr.shape[0]):
            v = arr[i]
            if v != v:
                n_null += 1
            else:
                n_valid += 1
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
        return mn, mx, n_null, n_valid


def _float_stats(arr: np.ndarray) -> bytes:
    """Canonical (min, max, null count) bytes of a float array, NaN counted as null (numba kernel for large arrays)."""
    if _numba is not None and arr.size >= _STATS_NUMBA_MIN_N and arr.dtype in (np.float32, np.float64):
        mn, mx, n_null, n_valid = _float_stats_nan_null_kernel(arr)
    else:
        nan = np.isnan(arr)
        n_null = int(nan.sum())
        valid = arr[~nan] if n_null else arr
        n_valid = valid.size
        mn, mx = (np.min(valid), np.max(valid)) if n_valid else (None, None)
    if not n_valid:
        return format_stats(None, None, n_null)
    return format_stats(float(mn), float(mx), n_null)


def pandas_column_stats(s: pd.Series, token: str) -> bytes:
    """Whole-column (min, max, null count) of a pandas column in canonical form. One vectorised pass per stat, no Python row
    loop: this is the only full-column work in the signature."""
    n = len(s)
    dt = s.dtype
    if token in _FLOAT_TOKENS:
        if isinstance(dt, np.dtype):
            return _float_stats(s.to_numpy())
        return _float_stats(s.to_numpy(dtype=_FLOAT_TOKENS[token].newbyteorder("="), na_value=np.nan))
    if token in ("bool", "int"):
        try:
            if isinstance(dt, np.dtype):
                arr = s.to_numpy()
                return format_stats(int(np.min(arr)), int(np.max(arr)), 0)
            mask = np.asarray(s.isna().to_numpy(), dtype=bool)
            n_null = int(mask.sum())
            if n_null == n:
                return format_stats(None, None, n_null)
            src = np.dtype(getattr(dt, "numpy_dtype", np.int64))
            arr = s.to_numpy(dtype=_numeric_np_dtype(token, src).newbyteorder("="), na_value=0)
            valid = arr[~mask] if n_null else arr
            return format_stats(int(np.min(valid)), int(np.max(valid)), n_null)
        except Exception as exc:
            logger.debug("cache: int/bool column min/max digest failed, using opaque marker: %s", exc)
            return b"int_opaque"
    if _is_temporal(token):
        try:
            i8, factor = _pandas_temporal_i8(s)
            valid_mask = i8 != _NAT
            n_null = int(n - np.count_nonzero(valid_mask))
            if n_null == n:
                return format_stats(None, None, n_null)
            valid = i8[valid_mask] if n_null else i8
            return format_stats(int(np.min(valid)) * factor, int(np.max(valid)) * factor, n_null)
        except Exception as exc:
            logger.debug("cache: temporal column min/max digest failed, using opaque marker: %s", exc)
            return b"temporal_opaque"
    n_null = int(s.isna().sum())
    if n_null == n:
        return format_stats(None, None, n_null)
    try:
        if token == "category":
            codes = np.asarray(s.cat.codes.to_numpy())
            used = np.bincount(codes[codes >= 0], minlength=len(s.cat.categories)) > 0
            labels = [str(v) for v in s.cat.categories[used].tolist()]
            return format_stats(min(labels), max(labels), n_null)
        # numpy object reductions do not reliably skip None (pandas 3 raises comparing str to None), so drop nulls first
        # there; extension string arrays skip them natively and are left alone to avoid the copy.
        base = s.dropna() if (n_null and isinstance(s.dtype, np.dtype)) else s
        mn, mx = base.min(skipna=True), base.max(skipna=True)
        return format_stats(mn if isinstance(mn, str) else str(mn), mx if isinstance(mx, str) else str(mx), n_null)
    except Exception as exc:
        # Mixed non-comparable objects: keep the null count, drop min/max rather than raising on a cache key.
        logger.debug("cache: text column min/max digest failed, keeping null count only: %s", exc)
        return b"opaque;" + format_stats(None, None, n_null)


def polars_column_stats(df: Any, cols: Sequence[str], tokens: Dict[str, str]) -> Dict[str, bytes]:
    """Whole-column stats for every column in ONE polars ``select`` (min / max / null count), canonicalised to the same bytes
    ``pandas_column_stats`` produces for the same logical data."""
    exprs: List[Any] = []
    plan: List[Tuple[str, bool]] = []
    for c in cols:
        tok = tokens[c]
        e = pl.col(c)
        has_minmax = True
        if tok in _FLOAT_TOKENS:
            e = e.fill_nan(None)
        elif _is_temporal(tok):
            e = e.cast(pl.Int64)
        elif tok == "category":
            e = e.cast(pl.String)
        elif tok not in ("bool", "int", "string"):
            has_minmax = False
        if has_minmax:
            exprs += [e.min().alias(f"_mn_{c}"), e.max().alias(f"_mx_{c}")]
        exprs.append(e.null_count().alias(f"_nc_{c}"))
        plan.append((c, has_minmax))
    row = df.select(exprs).row(0, named=True)
    out: Dict[str, bytes] = {}
    height = df.height
    for c, has_minmax in plan:
        tok = tokens[c]
        n_null = int(row[f"_nc_{c}"] or 0)
        if not has_minmax:
            out[c] = b"opaque;" + format_stats(None, None, n_null)
            continue
        mn, mx = row[f"_mn_{c}"], row[f"_mx_{c}"]
        if mn is None or n_null == height:
            out[c] = format_stats(None, None, n_null)
        elif tok in _FLOAT_TOKENS:
            out[c] = format_stats(float(mn), float(mx), n_null)
        elif tok in ("bool", "int"):
            out[c] = format_stats(int(mn), int(mx), n_null)
        elif _is_temporal(tok):
            f = _polars_ns_factor(df.schema[c])
            out[c] = format_stats(int(mn) * f, int(mx) * f, n_null)
        else:
            out[c] = format_stats(str(mn), str(mx), n_null)
    return out


# ----------------------------------------------------------------------------------------------
# Row-order fingerprint
# ----------------------------------------------------------------------------------------------


def row_order_fingerprint(df: Any, n_rows: int) -> Optional[bytes]:
    """Canonical bytes of the head ``n_rows`` rows plus a distinct tail window of every column, or None for an unsupported
    frame type. Column-major over the window: any permutation of distinguishable rows inside a window changes some column."""
    import hashlib

    if _is_polars_df(df):
        height = df.height
        windows = [df.slice(0, min(height, n_rows))]
        if height > n_rows:
            n_tail = min(height - n_rows, n_rows)
            windows.append(df.slice(height - n_tail, n_tail))
        encode = lambda w, j: encode_polars_slice(w.to_series(j), polars_logical_type(w.dtypes[j]))  # noqa: E731
    elif isinstance(df, pd.DataFrame):
        height = len(df)
        windows = [df.iloc[: min(height, n_rows)]]
        if height > n_rows:
            n_tail = min(height - n_rows, n_rows)
            windows.append(df.iloc[height - n_tail :])
        def encode(w, j):
            """Canonical bytes of column ``j`` of the sampled pandas slice ``w``."""
            col = w.iloc[:, j]
            return encode_pandas_slice(col, pandas_logical_type(col, col))
    else:
        return None
    if height == 0:
        return b""
    h = hashlib.blake2b(digest_size=8)
    for k, w in enumerate(windows):
        h.update(b"|win%d|" % k)
        for j in range(w.shape[1]):
            h.update(encode(w, j))
    return h.hexdigest().encode("ascii")
