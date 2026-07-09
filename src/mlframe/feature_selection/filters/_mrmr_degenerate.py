"""Degenerate-frame robustness audit + diagnostic surface for MRMR.

MRMR already tolerates pathological input columns without crashing (an all-NaN /
constant column lands at MI ~ 0 and is silently dropped by the relevance gate; an
exact-duplicate / perfectly-collinear column is dropped by the conditional-MI
redundancy gate). What was MISSING is a DIAGNOSTIC SURFACE telling the user WHICH
columns were degenerate and WHY -- mirroring the sibling selectors' diagnostic
attributes (RFECV's suspicious-column logs, ShapProxied's cluster reports).

``audit_degenerate_columns`` runs a cheap O(p) scan (one variance / NaN check per
column + an O(p) hashed-content pass for exact duplicates + a correlation pass over
the numeric columns for perfect collinearity) and returns a dict
``column -> reason`` where reason is one of::

    "all_nan"  -- every value is NaN/null
    "constant"  -- zero variance (one distinct non-null value)
    "duplicate_of:<col>" -- byte-identical to an earlier column
    "collinear_with:<col>"-- |Pearson| == 1 with an earlier numeric column (perfect
                            linear dependence, e.g. 2*x+3)

The scan is PURELY DIAGNOSTIC: it does NOT remove columns or alter which features
MRMR selects (the existing relevance / redundancy gates already handle that, byte
-identically). The result is stored on ``MRMR.degenerate_columns_`` after fit so a
downstream report / UI can show what the frame contained.

The y-NaN/inf clean ValueError parity with the sibling selectors lives in
``MRMR.fit`` directly (it must raise BEFORE any work); this module only handles the
column-side diagnostic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Perfect-collinearity tolerance. |Pearson| within this of 1.0 counts as a perfect
# linear dependence (covers float round-off in an exact 2*x+3 relationship).
_COLLINEAR_TOL = 1e-9


def _column_arrays(X):
    """Yield (name, values_ndarray) for each column of X (DataFrame / ndarray / polars).

    Returns names as the caller's column labels for a DataFrame, else integer positions.
    """
    if isinstance(X, pd.DataFrame):
        for name in X.columns:
            yield name, X[name].to_numpy()
        return
    # polars
    if hasattr(X, "columns") and hasattr(X, "to_numpy") and not isinstance(X, np.ndarray):
        try:
            arr = X.to_numpy()
            cols = list(X.columns)
            for i, name in enumerate(cols):
                yield name, arr[:, i]
            return
        except Exception:  # nosec B110 - best-effort path
            pass
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    for i in range(arr.shape[1]):
        yield i, arr[:, i]


def _is_all_nan(values: np.ndarray) -> bool:
    """True iff every entry is missing (NaN for numeric/complex dtypes, ``pd.isna`` for object dtype); empty arrays are not considered all-NaN."""
    if values.dtype.kind in "fc":
        return bool(values.size) and bool(np.all(np.isnan(values)))
    # object / other: treat None / NaN-like as missing
    try:
        return bool(values.size) and bool(pd.isna(values).all())
    except Exception:
        return False


def _is_constant(values: np.ndarray) -> bool:
    """True iff at most one distinct NON-NULL value (zero variance). All-null is
    handled separately as ``all_nan``; a single non-null distinct value is constant."""
    try:
        if values.dtype.kind in "fc":
            finite = values[~np.isnan(values)]
            if finite.size == 0:
                return False  # all-nan -> not "constant" (reported as all_nan)
            return bool(np.ptp(finite) == 0)
        # object / int / bool / datetime
        ser = pd.Series(values)
        nun = ser.nunique(dropna=True)
        return nun <= 1 and not ser.isna().all()
    except Exception:
        return False


def _content_key(values: np.ndarray):
    """A hashable, NaN-aware fingerprint of a column's content for exact-duplicate
    detection. ``pandas.util.hash_array`` is dtype-agnostic and maps NaN to a single
    stable sentinel, so two columns hash equal iff they are element-wise identical
    (NaN positions included)."""
    try:
        return pd.util.hash_array(np.asarray(values)).tobytes()
    except Exception:
        try:
            return np.asarray(values).tobytes()
        except Exception:
            return None


def audit_degenerate_columns(X) -> dict:
    """Cheap O(p) degenerate-column scan. Returns ``{column: reason}``.

    Order of precedence per column: all_nan > constant > duplicate_of > collinear_with.
    A column flagged for an earlier reason is NOT also tested for a later one (an
    all-NaN column is reported as ``all_nan``, never as a duplicate of another all-NaN
    column). Duplicate / collinear are reported relative to the FIRST (earliest)
    matching column.
    """
    degenerate: dict = {}
    seen_content: dict = {}  # content_key -> first column name
    numeric_cols: list = []  # (name, standardized values) for collinearity pass

    for name, values in _column_arrays(X):
        if _is_all_nan(values):
            degenerate[name] = "all_nan"
            continue
        if _is_constant(values):
            degenerate[name] = "constant"
            continue
        key = _content_key(values)
        if key is not None:
            if key in seen_content:
                degenerate[name] = f"duplicate_of:{seen_content[key]}"
                continue
            seen_content[key] = name
        # collect for the collinearity pass (numeric, non-degenerate only)
        if values.dtype.kind in "fiu":
            v = values.astype(np.float64)
            finite = np.isfinite(v)
            if finite.sum() >= 2:
                numeric_cols.append((name, v, finite))

    # Perfect-collinearity pass -- VECTORISED so the whole correlation matrix is one
    # BLAS GEMM (O(n*p^2) in optimised C) instead of a Python O(p^2) loop of np.corrcoef
    # calls (which on p=200 / n=20k cost ~8 s; the GEMM is ~tens of ms). NaNs are filled
    # with the per-column mean (so they contribute zero deviation) -- exact for an honest
    # complete linear relationship, which is the only case |corr| reaches 1.0. Only the
    # numeric, non-degenerate columns participate; the first column of each collinear
    # group is the reference.
    live = [(n, v, f) for (n, v, f) in numeric_cols if n not in degenerate]
    if len(live) >= 2:
        names = [n for (n, _, _) in live]
        n_rows = live[0][1].shape[0]
        M = np.empty((n_rows, len(live)), dtype=np.float64)
        for k, (_, v, fin) in enumerate(live):
            col = v.copy()
            if not fin.all():
                col_mean = float(np.nanmean(col)) if fin.any() else 0.0
                col = np.where(fin, col, col_mean)
            M[:, k] = col
        # Standardise; zero-variance columns (shouldn't reach here -- caught as constant)
        # are guarded by a non-zero std floor so they cannot spuriously read |corr|=1.
        with np.errstate(invalid="ignore"):  # a non-finite-derived col_mean can make the centre subtract NaN; the std floor below handles it
            M -= M.mean(axis=0, keepdims=True)
            stds = np.sqrt((M * M).sum(axis=0))
        good = stds > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            M = np.where(good, M / np.where(stds == 0, 1.0, stds), 0.0)
        corr = M.T @ M  # unit-norm columns -> Gram matrix == correlation matrix
        np.fill_diagonal(corr, 0.0)
        abs_corr = np.abs(corr)
        for j in range(len(live)):
            if not good[j]:
                continue
            # earliest i < j with |corr| ~ 1 and i itself not already flagged collinear
            row = abs_corr[j, :j]
            hits = np.where(np.abs(row - 1.0) <= _COLLINEAR_TOL)[0]
            for i in hits:
                if good[i] and names[i] not in degenerate:
                    degenerate[names[j]] = f"collinear_with:{names[i]}"
                    break

    return degenerate
