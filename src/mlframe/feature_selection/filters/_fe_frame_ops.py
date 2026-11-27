"""Format-agnostic frame operations for the MRMR FE seam.

The FE families historically took a pandas ``X`` and skipped outright on polars (``isinstance(X, pd.DataFrame)`` guards),
so a polars-native suite ran with most of the FE arsenal silently disabled. These helpers let the shared FE seam
(``fe_decide_on_subsample`` + the per-family append sites) operate on pandas / polars / ndarray uniformly, WITHOUT a
whole-frame ``to_pandas()`` copy: row-subsampling and column-appending are done natively per framework, and only the small
decision subsample is materialised.

Row subsampling returns a pandas frame because the family DECISION bodies are still pandas-native internally; that copy is
bounded by the subsample size (~30k rows), NOT the full frame, and is shared across a whole FE step. Full-n column APPEND
stays in the source framework (polars ``hstack`` / pandas ``concat``), so a 100+ GB frame is never duplicated.
"""
from __future__ import annotations

import logging
import threading
import weakref
from typing import Any, Iterable, Iterator

logger = logging.getLogger(__name__)

import numpy as np
from mlframe.utils.log_throttle import log_throttle
from mlframe.feature_selection.filters._fe_family_timing import fe_timed

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is a hard dep in practice
    pd = None
try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]


def is_pandas(X: Any) -> bool:
    """True iff ``X`` is a pandas DataFrame (safe when pandas itself failed to import)."""
    return pd is not None and isinstance(X, pd.DataFrame)


def is_polars(X: Any) -> bool:
    """True iff ``X`` is a polars DataFrame (safe when polars itself failed to import)."""
    return pl is not None and isinstance(X, pl.DataFrame)


def fe_columns(X: Any) -> list[str]:
    """Column names for a pandas / polars frame (list of str)."""
    if is_pandas(X):
        return [str(c) for c in X.columns]
    if is_polars(X):
        return list(X.columns)
    raise TypeError(f"fe_columns: unsupported frame type {type(X)!r}")


def _column_lookup_errors() -> tuple:
    """What a direct column lookup of a pandas frame raises when it cannot answer (evaluated only when a lookup failed, so pandas is not imported by this module)."""
    import pandas as pd

    return (LookupError, TypeError, ValueError, AttributeError, pd.errors.InvalidIndexError)


def _pandas_column_array(X: Any, c: Any) -> Any:
    """Backing array of pandas column ``c`` (an ndarray or an extension array) without building a Series, or ``None`` when the column is absent, its name is duplicated or the frame
    has no direct accessor. ``X[c]`` constructs a Series on every call (30 us with pandas 3, 40 us with ``.to_numpy()``); the FE families read columns about 90000 times in a 100000 x 100 fit."""
    try:
        i = X.columns.get_loc(c)
        if type(i) is not int:
            return None
        return X._get_column_array(i)
    except _column_lookup_errors():  # best-effort: absent or unhashable name, a non-unique index or a frame without the accessor; the caller then reads ``X[c]``
        return None


#: Rows per tile of :func:`fe_stack_columns`: 1024 rows of a few hundred float32 columns stay inside the second-level cache.
_STACK_TILE_ROWS = 1024


def fe_stack_columns(columns: "list[np.ndarray]", dtype: Any, out: "np.ndarray | None" = None) -> np.ndarray:
    """The ``(n, k)`` C-ordered matrix whose column ``j`` is ``columns[j]`` cast to ``dtype`` (``np.column_stack`` of the casts), filled in row tiles.

    Writing one whole column after another walks the destination with a stride of ``k`` values, a cache line per element: 483 ms for 170 columns of 100000 float32 rows, 77 ms in
    tiles of 1024 rows (each tile of the destination stays in cache while every column's slice of it is written). ``out`` (an ``(n, k)`` array of ``dtype``) is filled when given."""
    k = len(columns)
    n = int(columns[0].shape[0]) if k else 0
    mat = np.empty((n, k), dtype=dtype) if out is None else out
    if k == 1:
        mat[:, 0] = columns[0]
        return mat
    for r0 in range(0, n, _STACK_TILE_ROWS):
        r1 = min(n, r0 + _STACK_TILE_ROWS)
        for j in range(k):
            mat[r0:r1, j] = columns[j][r0:r1]
    return mat


def fe_stack_rows(columns: "list[np.ndarray]", dtype: Any) -> np.ndarray:
    """The ``(k, n)`` C-ordered matrix whose row ``j`` is ``columns[j]`` cast to ``dtype`` (each row a contiguous copy of one column), filled in tiles of rows of the source.

    The columns of a frame built from an ``(n, k)`` array are strided views of it, so copying one whole column after another reads a cache line per element (1.1 ms per column of 100000
    float32 rows, 102 columns: 121 ms); in tiles every column's slice of the same source rows is read while they are in cache."""
    k = len(columns)
    n = int(columns[0].shape[0]) if k else 0
    mat = np.empty((k, n), dtype=dtype)
    for r0 in range(0, n, _STACK_TILE_ROWS):
        r1 = min(n, r0 + _STACK_TILE_ROWS)
        for j in range(k):
            mat[j, r0:r1] = columns[j][r0:r1]
    return mat


def fe_std_below(a: np.ndarray, tol: float = 1e-12) -> bool:
    """``float(np.std(a)) < tol`` for a small ``tol`` (the "this column is constant" test of the FE families), without the standard deviation when the range settles it.

    With one value at the minimum and one at the maximum the standard deviation is at least ``range / sqrt(2 n)``, so a range above ``1e6 * tol`` cannot go with a standard deviation
    below ``tol`` for any column that fits in memory: two reductions instead of the four passes and two temporaries of ``np.std`` (0.14 ms -> 0.03 ms at 100000 rows; the families ask
    this several thousand times a fit). A NaN in ``a`` makes the range test fail, so the answer is then ``np.std``'s (False)."""
    if a.size and float(np.max(a)) - float(np.min(a)) > 1e6 * tol:
        return False
    return bool(float(np.std(a)) < tol)


def fe_column_values(X: Any, c: Any) -> np.ndarray:
    """``X[c].to_numpy()`` of a pandas frame: the column's values as a read-only view of the frame's block when it is numpy-backed."""
    arr = _pandas_column_array(X, c) if is_pandas(X) else None
    if type(arr) is np.ndarray:
        out: np.ndarray = arr.view()
        out.flags.writeable = False
        return out
    return np.asarray(X[c].to_numpy())


def fe_column_is_numeric(X: Any, c: Any) -> bool:
    """``c in X.columns and pd.api.types.is_numeric_dtype(X[c])`` of a pandas frame, read from the column's dtype alone."""
    arr = _pandas_column_array(X, c)
    if arr is None:
        if c not in X.columns:
            return False
        s = X[c]
        return bool(getattr(s, "ndim", 1) == 1 and pd.api.types.is_numeric_dtype(s))
    if type(arr) is np.ndarray:
        return arr.dtype.kind in "biufc"
    return bool(pd.api.types.is_numeric_dtype(arr.dtype))


def fe_is_numeric_col(X: Any, c: str) -> bool:
    """Whether column ``c`` of ``X`` is a numeric (incl. bool) scalar column, for pandas or polars, without raising on
    categorical / string columns. Duplicate-named pandas columns (``X[c]`` -> 2D) are treated as non-numeric (ambiguous)."""
    if is_pandas(X):
        return fe_column_is_numeric(X, c)
    if is_polars(X):
        if c not in X.columns:
            return False
        dt = X.schema[c]
        return True if dt.is_numeric() else dt == pl.Boolean
    raise TypeError(f"fe_is_numeric_col: unsupported frame type {type(X)!r}")


def _prefetch_lazy_rows(X: Any, idx: np.ndarray) -> None:
    """Gather the rows ``idx`` of the pending lazy engineered columns of the pandas frame ``X`` on a thread pool, before the frame is cut: the cut reads them one after another and replays each at the full row count."""
    from . import _lazy_engineered as lazy

    if not lazy.STATS["created"]:
        return
    names = [name for name, dtype in X.dtypes.items() if isinstance(dtype, lazy.LazyEngineeredDtype)]
    if names:
        lazy.prefetch_columns(X, names, rows=idx)


def fe_subsample_to_pandas(X: Any, idx: np.ndarray) -> Any:
    """Row-subsample ``X`` at integer positions ``idx`` and return a PANDAS frame (index reset).

    Only the subsample is materialised, so on a 100+ GB frame this copies ~len(idx) rows, never the whole frame - and it
    runs on CPU (host), where the frame lives. polars is gathered natively then bridged to pandas (the family decision
    bodies are pandas-native); pandas uses ``.iloc``. Numeric precision is NOT re-cast here: FE precision is governed by
    the existing ``MLFRAME_CRIT_DTYPE_RELAXED`` knob (``_crit_np_dtype()`` - f32-relaxed default), which the family bodies
    already consult at their ``.to_numpy`` boundary; a subsample-level cast would only lose precision without saving work
    while those bodies upcast to f64 (see the bench + Open-work-items note on the matrix-native plane).
    """
    if is_pandas(X):
        _prefetch_lazy_rows(X, idx)
        return X.iloc[idx].reset_index(drop=True)
    if is_polars(X):
        # X[idx] gathers rows natively; to_pandas on the small subsample is the cheap correct bridge for the pandas-native
        # family decision. Numeric columns become numpy-backed (nulls -> NaN), matching the densified pandas path.
        return X[np.asarray(idx)].to_pandas()
    raise TypeError(f"fe_subsample_to_pandas: unsupported frame type {type(X)!r}")


class _FrameColumns(dict):
    """``{name: 1-D array}`` of columns taken from a pandas frame, remembering the frame: appending them to another pandas frame concatenates the frame's own columns (copy-on-write sharing) instead of copying the read-only views."""

    source: Any = None


def _is_lazy_engineered(arr: Any) -> bool:
    """Whether ``arr`` is an engineered column replayed on demand (it moves between frames as it is, so the values are not computed by the move)."""
    from ._lazy_engineered import LazyEngineeredColumn

    return isinstance(arr, LazyEngineeredColumn)


def fe_extract_columns(X: Any, names: Iterable[str]) -> dict[str, np.ndarray]:
    """Extract the named columns of ``X`` (pandas / polars) as ``{name: 1d numpy array}`` - per-column views, no
    whole-frame copy. Used to move engineered columns off an augmented frame for a native re-append onto another frame."""
    out = _FrameColumns()
    out.source = X if is_pandas(X) else None
    for nm in names:
        if is_pandas(X):
            arr = _pandas_column_array(X, nm)
            out[nm] = arr if _is_lazy_engineered(arr) else fe_column_values(X, nm)
        elif is_polars(X):
            out[nm] = X[nm].to_numpy()
        else:
            raise TypeError(f"fe_extract_columns: unsupported frame type {type(X)!r}")
    return out


# Eager polars->pandas materialisation is bounded to frames under this size (CLAUDE.md eager-conversion rule); a larger
# polars frame must not be whole-frame-copied to pandas, so the few OOF / cross-row FE families that genuinely need the
# full frame (no closed-form subsample-replay) skip above it rather than double peak RAM on a 100+ GB frame.
FE_EAGER_MATERIALIZE_MAX_BYTES = 2 * 1024**3


def fe_n_rows(X: Any) -> int:
    """Row count of a pandas / polars frame."""
    if is_pandas(X):
        return int(X.shape[0])
    if is_polars(X):
        return int(X.height)
    raise TypeError(f"fe_n_rows: unsupported frame type {type(X)!r}")


def fe_column_blocks(X: Any, names: "Iterable[str] | None" = None, budget_bytes: int = 256 * 1024 * 1024, dtype: Any = np.float32) -> "Iterator[tuple[list[str], np.ndarray]]":
    """Blocks of numeric columns of a pandas / polars frame as C-contiguous ``(columns, rows)`` arrays of ``dtype``, each block within ``budget_bytes`` (at least one column).

    The one reading seam for a frame that may not fit in memory: nothing but the current block is ever held, a column goes frame -> block with a single cast-and-copy (a polars float32
    column is a zero-copy view first), and the rows of a block are contiguous per column, which is the layout the device kernels and the code coders read. ``names`` defaults to every numeric column."""
    cols = [c for c in (fe_columns(X) if names is None else names) if fe_is_numeric_col(X, c)]
    n = fe_n_rows(X)
    per_col = max(1, n) * np.dtype(dtype).itemsize
    step = max(1, int(budget_bytes) // per_col)
    for c0 in range(0, len(cols), step):
        part = cols[c0 : c0 + step]
        block = np.empty((len(part), n), dtype=dtype)
        for k, c in enumerate(part):
            block[k] = X[c].to_numpy()
        yield part, block


def fe_numeric_float32_view(X: Any) -> Any:
    """Pandas copy of the numeric columns of a polars frame, stored float32, built one column block at a time.

    Half the memory of ``to_pandas`` on a wide numeric frame, because the double copy of the whole frame never exists:
    only one block is alive besides the result. Non-numeric columns are left out. Used by the families that must read
    a large polars frame but need only its numeric columns (see audit row 28).
    """
    num = [c for c in X.columns if X[c].dtype.is_numeric()]
    mat = np.empty((len(num), X.height), dtype=np.float32)  # (columns, rows): each column contiguous, the frame below is a transposed view of it
    for k, c in enumerate(num):
        mat[k] = X[c].to_numpy()
    return pd.DataFrame(mat.T, columns=num, index=pd.RangeIndex(X.height), copy=False)


def fe_polars_exceeds(X: Any, max_bytes: int = FE_EAGER_MATERIALIZE_MAX_BYTES) -> bool:
    """True iff ``X`` is a polars frame whose in-memory size exceeds ``max_bytes`` (so an eager ``to_pandas`` would be a
    large whole-frame copy). pandas / ndarray return False (already pandas, or their family avoids the copy)."""
    if is_polars(X):
        try:
            return int(X.estimated_size()) > int(max_bytes)
        except Exception as e:
            log_throttle(logger, "_fe_frame_ops.fe_polars_exceeds", logging.WARNING, "fe_polars_exceeds: estimated_size() failed, treating frame as within budget: %s", e)
            return False
    return False


def fe_to_pandas(X: Any) -> Any:
    """Return a pandas frame for ``X`` (identity for pandas, ``.to_pandas()`` for polars).

    ONLY for the rare full-n FE fallback paths (subsample disabled / partial recipe coverage / unexpected family shape),
    which historically re-ran the pandas-native family on the whole frame anyway. On the normal subsampled path this is
    never called, so a 100+ GB frame is not converted in steady state.
    """
    if is_pandas(X):
        return X
    if is_polars(X):
        return X.to_pandas()
    return X


def fe_drop_columns(X: Any, names: Iterable[str]) -> Any:
    """``X`` without the columns ``names`` (those present), for a pandas or a polars frame; ``X`` itself when there is nothing to drop."""
    present = set(fe_columns(X))
    drop = [c for c in names if c in present]
    if not drop:
        return X
    return X.drop(columns=drop) if is_pandas(X) else X.drop(drop)


def fe_frame_from_columns(cols: dict, index: Any) -> "pd.DataFrame":
    """A pandas frame over the column arrays of ``cols`` (``name -> 1-D array``, in dict order) WITHOUT copying them: the frame holds the arrays themselves (one block per column).

    ``pd.DataFrame(dict)`` consolidates the columns into blocks, which copies every array once and, with mixed dtypes, about twice more (an extra-basis frame of 30000 x 5100 legs peaked at 3.9 GB for a 0.94 GB result).
    The caller hands the arrays over: they must not be modified through other references afterwards. A read-only array (a view of another frame's block) is copied, so the frame never shares memory it cannot write."""
    return pd.DataFrame({name: (arr if getattr(arr, "flags", None) is None or arr.flags.writeable else arr.copy()) for name, arr in cols.items()}, index=index, copy=False)


_APPENDED_FROM: "dict[int, tuple[weakref.ref, weakref.ref, tuple, Any]]" = {}
_APPENDED_LOCK = threading.Lock()


def _note_appended(child: Any, parent: Any, names: Iterable[str], added: Any = None) -> None:
    """Remember that the pandas frame ``child`` is ``parent`` with the columns ``names`` appended at the end, and the small frame ``added`` that holds exactly those columns (the frame and its parent held weakly,
    ``added`` as long as the entry lives; the entry goes when ``child`` does).

    ``added`` lets a reader take rows of the appended columns without touching ``child``: the first column selection of a freshly concatenated frame makes pandas rebuild its block bookkeeping,
    which costs a tenth of a second at 1000 columns and more with hundreds of extension blocks."""
    key = id(child)

    def _drop(_ref: Any, key: int = key) -> None:
        """Forget the entry of a collected frame."""
        with _APPENDED_LOCK:
            _APPENDED_FROM.pop(key, None)

    try:
        entry = (weakref.ref(child, _drop), weakref.ref(parent), tuple(names), added)
    except TypeError:  # not weakly referenceable: no lineage
        logger.debug("_note_appended: %s cannot be weakly referenced; no appended-column lineage is recorded", type(child).__name__)
        return
    with _APPENDED_LOCK:
        _APPENDED_FROM[key] = entry


def appended_from(child: Any) -> "tuple[Any, tuple, Any] | None":
    """``(parent frame, appended column names, frame of the appended columns or None)`` when ``child`` was made by :func:`fe_append_columns` from a parent that is still alive, else ``None``."""
    with _APPENDED_LOCK:
        entry = _APPENDED_FROM.get(id(child))
    if entry is None or entry[0]() is not child:
        return None
    parent = entry[1]()
    return None if parent is None else (parent, entry[2], entry[3])


@fe_timed("stage:append_columns")
def fe_append_columns(X: Any, cols: dict[str, np.ndarray]) -> Any:
    """Append engineered numpy columns ``{name: 1d-array}`` to ``X`` in ITS OWN framework (no whole-frame conversion).

    pandas -> ``concat`` (aligned on X.index); polars -> ``with_columns`` (native, zero-copy per Series); ndarray ->
    horizontal stack. On pandas without copy-on-write the ``concat`` copies every existing column too, not just the new
    ones. A shallow-copy-and-assign variant shares the base but did not win end to end (fragmentation; see
    ``_benchmarks/bench_fe_append_columns.py`` and ``_benchmarks/ab_fe_append_columns_e2e.py``).
    """
    if not cols:
        return X
    if is_pandas(X):
        # no consolidation: ``pd.DataFrame(dict)`` copies the columns into blocks (at 1000000 rows the engineered columns of the replay, several hundred of 8 MB, took 2.6 GB twice over); the
        # frame of ``fe_frame_from_columns`` holds the arrays themselves and the concat shares them, so the columns exist once
        src = getattr(cols, "source", None)
        if src is not None and src.index.equals(X.index) and not src.columns.duplicated().any():
            # the columns are views of ``src``'s blocks (read-only): the frame's own columns are shared under copy-on-write, where wrapping the views would copy each one
            added = src[list(cols)]
        else:
            added = fe_frame_from_columns(cols, X.index)
        out = pd.concat([X, added], axis=1)
        _note_appended(out, X, list(cols), added)
        return out
    if is_polars(X):
        return X.with_columns([pl.Series(name, np.asarray(vals)) for name, vals in cols.items()])
    if isinstance(X, np.ndarray):
        extra = np.column_stack([np.asarray(v) for v in cols.values()])
        return np.hstack([X, extra])
    raise TypeError(f"fe_append_columns: unsupported frame type {type(X)!r}")
