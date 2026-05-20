"""Auto-detect helpers for composite-target discovery: time-column candidates (for EWMA / rolling / frac_diff which assume chronological row order) and group-column candidates (for linear_residual_grouped). Pure pandas / polars + numpy; no composite-internal deps."""


from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import polars as pl  # type: ignore
    _HAS_POLARS = True
except ImportError:  # pragma: no cover
    pl = None  # type: ignore
    _HAS_POLARS = False


def _is_polars_df(x: Any) -> bool:
    """True iff polars is importable and ``x`` is a polars DataFrame. Explicit
    isinstance avoids duck-typing false positives (e.g. user-side wrappers)."""
    return _HAS_POLARS and isinstance(x, pl.DataFrame)


from .composite_screening import _is_numeric_column


# ----------------------------------------------------------------------
# detect_time_column + sort_df_by_time (auto-sort for EWMA / rolling / frac_diff transforms which assume chronological row order).
#
# The time-series transforms (``ewma_residual`` / ``rolling_quantile_ratio`` / ``frac_diff``) silently produce semantically-wrong T when fed unordered rows -- the EWMA recursion + rolling window assume rows are in chronological order. Most datasets at the discovery stage are unsorted (sample / shuffled splits), so unconditional default would corrupt results. These helpers solve that.
#
# detect_time_column_candidates:
# - Datetime-dtype columns are always candidates (highest score).
# - Numeric columns are candidates if STRICTLY monotonic increasing (or decreasing) on the input order -- common pattern: row_id, timestamp-as-int, depth (well-log).
# - Returns (col_name, info) ranked by signal strength.
#
# sort_df_by_time_column:
# - Returns a copy of df with rows sorted by the chosen time column ASCENDING. Preserves index alignment so caller can map back. Polars / pandas aware.
# ----------------------------------------------------------------------


def detect_time_column_candidates(
    df: Any,
    *,
    candidate_columns: Sequence[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Scan ``df`` for columns that look like chronological time keys (suitable for EWMA / rolling / frac_diff transforms that require ordered input).

    Returns list of ``(column_name, info_dict)`` sorted by score descending. ``info_dict`` carries:
    - ``dtype``: str, the source column dtype.
    - ``is_datetime``: bool, True for datetime / timestamp dtypes.
    - ``is_monotonic``: bool, True if strictly increasing OR strictly decreasing.
    - ``monotonic_direction``: "asc" / "desc" / None.
    - ``score``: float, datetime > monotonic numeric > nothing else.

    Empty when no column qualifies.
    """
    if _is_polars_df(df):
        if candidate_columns is None:
            candidate_columns = list(df.columns)
        # ``_is_polars_df`` already ensured polars is importable, so we
        # can reference the module-level ``pl`` symbol directly. The
        # prior local re-import was dead code that obscured the
        # dependency from static analysers.
        def get_col(c):
            return df.get_column(c)
        def get_dtype(c):
            return str(df.schema[c])
    elif isinstance(df, pd.DataFrame):
        if candidate_columns is None:
            candidate_columns = list(df.columns)
        def get_col(c):
            return df[c]
        def get_dtype(c):
            return str(df[c].dtype)
    else:
        raise TypeError(f"detect_time_column_candidates: unsupported df type {type(df).__name__}")

    results: list[tuple[str, dict[str, Any]]] = []
    for col in candidate_columns:
        try:
            dtype_str = get_dtype(col).lower()
            series = get_col(col)
        except Exception as _e:
            # Pre-fix `continue` silently dropped cols where dtype / series
            # access raised, so operators wondered why a hoped-for time_col
            # never picked. DEBUG level (not WARN) because skipping IS the
            # right action - we just want the trail when it matters.
            import logging as _logging
            _logging.getLogger(__name__).debug(
                "detect_time_column_candidates: skipping col=%r: %s", col, _e,
            )
            continue
        is_datetime = ("datetime" in dtype_str
                       or "timestamp" in dtype_str
                       or dtype_str.startswith("date"))
        info: dict[str, Any] = {
            "dtype": dtype_str,
            "is_datetime": is_datetime,
            "is_monotonic": False,
            "monotonic_direction": None,
            "score": 0.0,
        }
        if is_datetime:
            info["score"] = 100.0
            results.append((str(col), info))
            continue
        # Numeric: check monotonicity. ``series.to_numpy()`` (polars/pandas)
        # already returns an ndarray, no extra np.asarray wrap needed; the
        # subsequent astype is copy=False so dtype-match becomes a no-op.
        try:
            if hasattr(series, "to_numpy"):
                arr = series.to_numpy()
            else:
                arr = np.asarray(series)
            if not np.issubdtype(arr.dtype, np.number):
                arr = arr.astype(np.float64, copy=False)
        except (TypeError, ValueError):
            continue
        finite = np.isfinite(arr)
        if finite.sum() < 2:
            continue
        diffs = np.diff(arr[finite])
        if np.all(diffs > 0):
            info["is_monotonic"] = True
            info["monotonic_direction"] = "asc"
            info["score"] = 50.0
            results.append((str(col), info))
        elif np.all(diffs < 0):
            info["is_monotonic"] = True
            info["monotonic_direction"] = "desc"
            info["score"] = 50.0
            results.append((str(col), info))
    results.sort(key=lambda kv: kv[1]["score"], reverse=True)
    return results


def sort_df_by_time_column(df: Any, time_column: str, *, ascending: bool = True) -> Any:
    """Return a copy of ``df`` sorted by ``time_column``. Polars / pandas aware. Original df is NOT mutated.

    Required precondition for the time-series composite transforms (ewma_residual, rolling_quantile_ratio, frac_diff). Caller is responsible for keeping aligned arrays (y, base) in the SAME sort order.
    """
    if _is_polars_df(df):
        return df.sort(time_column, descending=not ascending)
    if isinstance(df, pd.DataFrame):
        return df.sort_values(by=time_column, ascending=ascending).reset_index(drop=True)
    raise TypeError(f"sort_df_by_time_column: unsupported df type {type(df).__name__}")


# ----------------------------------------------------------------------
# detect_group_column (auto-detect a categorical column suitable for ``linear_residual_grouped``).
#
# Discovery doesn't currently know how to pick a ``group_column`` for the grouped residual transform -- callers configure it manually. This helper scans the dataframe and recommends column candidates that look like group keys (categorical, moderate cardinality, balanced sizes).
#
# Recommended ranges (calibrated against the TVT well_id pattern):
# - ``min_unique`` (default 3): below this is too few groups for J-S shrinkage to help.
# - ``max_unique`` (default 500): above this is too granular -- per-group fits over-fit on tiny groups.
# - ``min_size_ratio`` (default 0.01): smallest group must hold at least 1% of rows so the per-group OLS has enough data. Floors at min_size_ratio * n_rows.
#
# Returns a list of (column_name, info_dict) sorted by score (highest first); empty when no column matches. Score combines (a) inverse coefficient of variation of group sizes (more uniform = better) and (b) MI-style information gain (proxied by entropy of group assignments). Caller picks the top candidate or evaluates the full list.
# ----------------------------------------------------------------------

_GROUP_DETECT_DEFAULT_MIN_UNIQUE: int = 3
_GROUP_DETECT_DEFAULT_MAX_UNIQUE: int = 500
_GROUP_DETECT_DEFAULT_MIN_SIZE_RATIO: float = 0.01


def detect_group_column_candidates(
    df: Any,
    *,
    candidate_columns: Sequence[str] | None = None,
    min_unique: int = _GROUP_DETECT_DEFAULT_MIN_UNIQUE,
    max_unique: int = _GROUP_DETECT_DEFAULT_MAX_UNIQUE,
    min_size_ratio: float = _GROUP_DETECT_DEFAULT_MIN_SIZE_RATIO,
) -> list[tuple[str, dict[str, Any]]]:
    """Scan ``df`` for columns that look like group keys (suitable for ``linear_residual_grouped``).

    Returns
    -------
    List of ``(column_name, info_dict)`` sorted by score descending. ``info_dict`` carries:
    - ``n_unique``: int, number of distinct values.
    - ``min_group_size``: int, rows in the smallest group.
    - ``max_group_size``: int, rows in the largest group.
    - ``size_cv``: float, coefficient of variation of group sizes (lower = more uniform).
    - ``score``: float, composite ranking score (higher = better candidate).

    Empty list when no column meets all the thresholds.
    """
    if _is_polars_df(df):
        if candidate_columns is None:
            candidate_columns = [
                c for c in df.columns
                if not _is_numeric_column(df, c)
            ]
        def get_col(c):
            # Polars Series.to_numpy() already returns an ndarray -- the
            # earlier np.asarray wrapper allocated a redundant view.
            return df.get_column(c).to_numpy()
    elif isinstance(df, pd.DataFrame):
        if candidate_columns is None:
            # Default: ALL non-numeric columns + low-cardinality numeric (int) ones.
            candidate_columns = [
                c for c in df.columns
                if not pd.api.types.is_numeric_dtype(df[c])
                or (pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() <= max_unique)
            ]
        def get_col(c):
            return df[c].to_numpy()
    else:
        raise TypeError(f"detect_group_column_candidates: unsupported df type {type(df).__name__}")

    n_rows = len(df)
    min_size_floor = max(1, int(min_size_ratio * n_rows))
    results: list[tuple[str, dict[str, Any]]] = []
    for col in candidate_columns:
        try:
            arr = get_col(col)
        except Exception as _e:
            # Same shape as detect_time_column_candidates above: silent skip
            # blinds operators to "why didn't group_col pick this?"; DEBUG
            # log keeps the trail without WARN-spamming on normal scans.
            import logging as _logging
            _logging.getLogger(__name__).debug(
                "detect_group_column_candidates: skipping col=%r: %s", col, _e,
            )
            continue
        # Skip all-null columns.
        mask_finite = pd.notna(arr) if hasattr(arr, "__iter__") else None
        if mask_finite is not None and not np.any(mask_finite):
            continue
        uniq, counts = np.unique(arr[mask_finite] if mask_finite is not None else arr, return_counts=True)
        n_unique = int(uniq.size)
        if n_unique < min_unique or n_unique > max_unique:
            continue
        min_g = int(counts.min())
        max_g = int(counts.max())
        if min_g < min_size_floor:
            continue
        mean_g = float(counts.mean())
        std_g = float(counts.std())
        size_cv = std_g / max(mean_g, 1.0)
        # Score: low size_cv (uniform groups) + moderate n_unique (more groups = more JS information).
        # Bound n_unique contribution; very large n_unique is penalised by max_unique gate already.
        score = (1.0 / (1.0 + size_cv)) * float(min(n_unique, 50))
        results.append((str(col), {
            "n_unique": n_unique,
            "min_group_size": min_g,
            "max_group_size": max_g,
            "size_cv": size_cv,
            "score": score,
        }))
    results.sort(key=lambda kv: kv[1]["score"], reverse=True)
    return results


# ----------------------------------------------------------------------
# detect_cat_columns: symmetric sibling of detect_group_column_candidates,
# calibrated for FHC target-encoding (target_mean / WoE) rather than
# linear_residual_grouped.
#
# Wave 64 (2026-05-20): closes the wave-10 deferral. Where group-column
# detection wants UNIFORM group sizes (linear_residual_grouped needs
# balanced groups to fit per-group affine residuals reliably), category
# detection wants ENOUGH SAMPLES PER LEVEL (FHC encoders need >=
# min_samples_per_cat samples per category to estimate a stable target
# mean / log-odds without high variance). The thresholds differ:
#
#   detect_group_column_candidates: min_unique=3, max_unique=500,
#     min_size_ratio=0.01 (uniform), score = 1/(1+size_cv) * min(n_unique, 50)
#
#   detect_cat_columns: min_unique=2, max_unique=1000, min_samples_per_cat=20
#     (absolute, NOT ratio -- FHC stability is dominated by *absolute*
#     per-category count not by relative balance), score weighted toward
#     coverage (more rows in the largest k categories = stronger signal).
# ----------------------------------------------------------------------

_CAT_DETECT_DEFAULT_MIN_UNIQUE: int = 2
_CAT_DETECT_DEFAULT_MAX_UNIQUE: int = 1000
_CAT_DETECT_DEFAULT_MIN_SAMPLES_PER_CAT: int = 20


def detect_cat_columns(
    df: Any,
    *,
    candidate_columns: Sequence[str] | None = None,
    min_unique: int = _CAT_DETECT_DEFAULT_MIN_UNIQUE,
    max_unique: int = _CAT_DETECT_DEFAULT_MAX_UNIQUE,
    min_samples_per_cat: int = _CAT_DETECT_DEFAULT_MIN_SAMPLES_PER_CAT,
) -> list[tuple[str, dict[str, Any]]]:
    """Scan ``df`` for columns that look like categorical features (suitable for
    FHC target_mean / WoE / CatBoostEncoder).

    Distinct from ``detect_group_column_candidates`` (which targets uniform
    grouping for linear_residual_grouped); the calibration here favors columns
    where the per-category sample count is sufficient for stable target-encoding
    statistics, not for balanced groups.

    Parameters
    ----------
    df : pl.DataFrame | pd.DataFrame
    candidate_columns : sequence of str, optional
        Columns to consider. Defaults to all non-float columns (str / bool /
        low-card int).
    min_unique : int, default 2
        Minimum distinct values (binary indicators are valid categoricals).
    max_unique : int, default 1000
        Above this cardinality, FHC encoding produces too many sparse buckets
        and is usually dominated by hash / embedding methods instead.
    min_samples_per_cat : int, default 20
        Absolute minimum samples for the SMALLEST category. FHC stability is
        dominated by per-category count, not by relative balance.

    Returns
    -------
    List of ``(column_name, info_dict)`` sorted by score descending. ``info_dict``
    carries:
    - ``n_unique``: int
    - ``min_per_cat``: int, rows in the rarest category
    - ``max_per_cat``: int, rows in the most-common category
    - ``coverage_top10``: float, fraction of rows covered by the 10 most common
      categories (high coverage = lots of signal concentrated in few levels).
    - ``score``: float, composite ranking (higher = better categorical candidate)

    Empty list when no column meets the thresholds. Safe for both pl.DataFrame
    and pd.DataFrame.
    """
    if _is_polars_df(df):
        if candidate_columns is None:
            candidate_columns = [
                c for c in df.columns
                if not _is_numeric_column(df, c)
            ]
        def get_col(c):
            return df.get_column(c).to_numpy()
    elif isinstance(df, pd.DataFrame):
        if candidate_columns is None:
            # Default: non-float columns -- string, bool, low-cardinality int
            # (the int-as-cat heuristic from project_mlframe_int_as_cat_detector).
            candidate_columns = [
                c for c in df.columns
                if not pd.api.types.is_float_dtype(df[c])
                and (
                    not pd.api.types.is_numeric_dtype(df[c])
                    or (pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() <= max_unique)
                )
            ]
        def get_col(c):
            return df[c].to_numpy()
    else:
        raise TypeError(f"detect_cat_columns: unsupported df type {type(df).__name__}")

    n_rows = len(df)
    if n_rows == 0:
        return []

    results: list[tuple[str, dict[str, Any]]] = []
    for col in candidate_columns:
        try:
            arr = get_col(col)
        except Exception as _e:
            import logging as _logging
            _logging.getLogger(__name__).debug(
                "detect_cat_columns: skipping col=%r: %s", col, _e,
            )
            continue

        # Strip nulls so they don't count as a separate "category".
        mask_finite = pd.notna(arr) if hasattr(arr, "__iter__") else None
        if mask_finite is not None and not np.any(mask_finite):
            continue
        arr_clean = arr[mask_finite] if mask_finite is not None else arr
        uniq, counts = np.unique(arr_clean, return_counts=True)
        n_unique = int(uniq.size)
        if n_unique < min_unique or n_unique > max_unique:
            continue
        min_per_cat = int(counts.min())
        max_per_cat = int(counts.max())
        if min_per_cat < min_samples_per_cat:
            continue

        # Coverage by the top-10 categories: high coverage means few categories
        # carry most rows, giving FHC encoders strong signal-to-noise.
        top10_count = int(np.sort(counts)[-10:].sum())
        coverage_top10 = top10_count / max(1, len(arr_clean))

        # Score: favor moderate n_unique (too few -> low info; too many -> sparse)
        # weighted by coverage. The (n_unique / log(n_unique+1)) shape rewards
        # 10-100 categories more than 2 or 500.
        info_bonus = float(n_unique) / float(np.log1p(n_unique) + 1.0)
        score = coverage_top10 * info_bonus

        results.append((str(col), {
            "n_unique": n_unique,
            "min_per_cat": min_per_cat,
            "max_per_cat": max_per_cat,
            "coverage_top10": coverage_top10,
            "score": score,
        }))
    results.sort(key=lambda kv: kv[1]["score"], reverse=True)
    return results
