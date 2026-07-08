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

from typing import Any

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is a hard dep in practice
    pd = None
try:
    import polars as pl
except Exception:
    pl = None  # type: ignore[assignment]


def is_pandas(X: Any) -> bool:
    return pd is not None and isinstance(X, pd.DataFrame)


def is_polars(X: Any) -> bool:
    return pl is not None and isinstance(X, pl.DataFrame)


def fe_columns(X: Any) -> list[str]:
    """Column names for a pandas / polars frame (list of str)."""
    if is_pandas(X):
        return [str(c) for c in X.columns]
    if is_polars(X):
        return list(X.columns)
    raise TypeError(f"fe_columns: unsupported frame type {type(X)!r}")


def fe_is_numeric_col(X: Any, c: str) -> bool:
    """Whether column ``c`` of ``X`` is a numeric (incl. bool) scalar column, for pandas or polars, without raising on
    categorical / string columns. Duplicate-named pandas columns (``X[c]`` -> 2D) are treated as non-numeric (ambiguous)."""
    if is_pandas(X):
        if c not in X.columns:
            return False
        s = X[c]
        if getattr(s, "ndim", 1) != 1:
            return False
        return bool(pd.api.types.is_numeric_dtype(s))
    if is_polars(X):
        if c not in X.columns:
            return False
        dt = X.schema[c]
        return bool(dt.is_numeric()) or dt == pl.Boolean
    raise TypeError(f"fe_is_numeric_col: unsupported frame type {type(X)!r}")


def fe_subsample_to_pandas(X: Any, idx: np.ndarray):
    """Row-subsample ``X`` at integer positions ``idx`` and return a PANDAS frame (index reset).

    Only the subsample is materialised, so on a 100+ GB frame this copies ~len(idx) rows, never the whole frame -- and it
    runs on CPU (host), where the frame lives. polars is gathered natively then bridged to pandas (the family decision
    bodies are pandas-native); pandas uses ``.iloc``. Numeric precision is NOT re-cast here: FE precision is governed by
    the existing ``MLFRAME_CRIT_DTYPE_RELAXED`` knob (``_crit_np_dtype()`` -- f32-relaxed default), which the family bodies
    already consult at their ``.to_numpy`` boundary; a subsample-level cast would only lose precision without saving work
    while those bodies upcast to f64 (see the bench + Open-work-items note on the matrix-native plane).
    """
    if is_pandas(X):
        return X.iloc[idx].reset_index(drop=True)
    if is_polars(X):
        # X[idx] gathers rows natively; to_pandas on the small subsample is the cheap correct bridge for the pandas-native
        # family decision. Numeric columns become numpy-backed (nulls -> NaN), matching the densified pandas path.
        return X[np.asarray(idx)].to_pandas()
    raise TypeError(f"fe_subsample_to_pandas: unsupported frame type {type(X)!r}")


def fe_extract_columns(X: Any, names) -> dict[str, np.ndarray]:
    """Extract the named columns of ``X`` (pandas / polars) as ``{name: 1d numpy array}`` -- per-column views, no
    whole-frame copy. Used to move engineered columns off an augmented frame for a native re-append onto another frame."""
    out: dict[str, np.ndarray] = {}
    for nm in names:
        if is_pandas(X):
            out[nm] = X[nm].to_numpy()
        elif is_polars(X):
            out[nm] = X[nm].to_numpy()
        else:
            raise TypeError(f"fe_extract_columns: unsupported frame type {type(X)!r}")
    return out


# Eager polars->pandas materialisation is bounded to frames under this size (CLAUDE.md eager-conversion rule); a larger
# polars frame must not be whole-frame-copied to pandas, so the few OOF / cross-row FE families that genuinely need the
# full frame (no closed-form subsample-replay) skip above it rather than double peak RAM on a 100+ GB frame.
FE_EAGER_MATERIALIZE_MAX_BYTES = 2 * 1024**3


def fe_polars_exceeds(X: Any, max_bytes: int = FE_EAGER_MATERIALIZE_MAX_BYTES) -> bool:
    """True iff ``X`` is a polars frame whose in-memory size exceeds ``max_bytes`` (so an eager ``to_pandas`` would be a
    large whole-frame copy). pandas / ndarray return False (already pandas, or their family avoids the copy)."""
    if is_polars(X):
        try:
            return int(X.estimated_size()) > int(max_bytes)
        except Exception:
            return False
    return False


def fe_to_pandas(X: Any):
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


def fe_append_columns(X: Any, cols: dict[str, np.ndarray]) -> Any:
    """Append engineered numpy columns ``{name: 1d-array}`` to ``X`` in ITS OWN framework (no whole-frame conversion).

    pandas -> ``concat`` (aligned on X.index); polars -> ``with_columns`` (native, zero-copy per Series); ndarray ->
    horizontal stack. Only the new columns are materialised; the base frame is not duplicated.
    """
    if not cols:
        return X
    if is_pandas(X):
        add = pd.DataFrame(cols, index=X.index)
        return pd.concat([X, add], axis=1)
    if is_polars(X):
        return X.with_columns([pl.Series(name, np.asarray(vals)) for name, vals in cols.items()])
    if isinstance(X, np.ndarray):
        extra = np.column_stack([np.asarray(v) for v in cols.values()])
        return np.hstack([X, extra])
    raise TypeError(f"fe_append_columns: unsupported frame type {type(X)!r}")
