"""NaN, infinity, and constant-column processing."""

from __future__ import annotations

import logging
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import polars as pl

# .utils re-exports from this module; defer the filter_existing import
# to call sites (line 110, 114) to break the cycle.

logger = logging.getLogger(__name__)

try:
    import polars.selectors as cs
except ImportError:  # pragma: no cover
    cs = None


def _process_special_values(
    df: pl.DataFrame | pd.DataFrame,
    expr_func: Callable[[], pl.Expr] | None = None,
    fill_func_name: str | None = None,
    kind: str = "",
    fill_value: float | None = None,
    drop_columns: bool = False,
    verbose: int = 1,
) -> pl.DataFrame | pd.DataFrame:
    """
    Generic handler for NaNs, nulls, infinities, or constant columns in numeric columns.

    Args:
        df: Input DataFrame (Polars or pandas)
        expr_func: Function returning Polars expression for detecting issues
        fill_func_name: Method name for filling values (fill_nan, fill_null, replace)
        kind: Description of issue type (for logging)
        fill_value: Value to fill with
        drop_columns: Whether to drop problematic columns
        verbose: Verbosity level

    Returns:
        Cleaned DataFrame
    """
    is_polars = isinstance(df, pl.DataFrame)

    if verbose:
        logger.info(f"{'Identifying' if drop_columns else 'Counting'} {kind}{'s' if not kind.endswith('columns') else ''}...")

    if is_polars:
        # Polars: use provided expr_func
        qos_df = df.select(expr_func())

        if drop_columns:
            # For constant detection, we get boolean indicators
            if len(qos_df) > 0:
                constant_cols = [col for col, is_const in zip(qos_df.columns, qos_df.row(0)) if is_const]
            else:
                constant_cols = []
            errors_df = pl.DataFrame({"column": constant_cols, "nerrors": [1] * len(constant_cols)}, schema={"column": pl.String, "nerrors": pl.Int64})
        else:
            if len(qos_df) > 0:
                errors_df = pl.DataFrame({"column": qos_df.columns, "nerrors": qos_df.row(0)}).filter(pl.col("nerrors") > 0).sort("nerrors", descending=True)
            else:
                errors_df = pl.DataFrame({"column": [], "nerrors": []}, schema={"column": pl.String, "nerrors": pl.Int64})
        nrows, ncols = df.shape
    else:
        # Pandas: different handling
        if drop_columns:
            # For constant column detection
            if "numeric" in kind:
                # Numeric constants: min == max would miss all-NaN columns
                # (``NaN == NaN`` is False under IEEE-754 semantics, so the
                # comparison silently treats an all-NaN column as non-constant
                # and leaves it in the frame). Use ``nunique(dropna=False)``
                # which counts NaN as its own bucket: pure-NaN -> 1, mixed
                # NaN+value -> 2+, all-equal numeric -> 1. Matches the polars
                # branch which treats all-NaN columns as constant.
                constant_cols = [
                    col for col in df.select_dtypes(include="number").columns
                    if df[col].nunique(dropna=False) <= 1
                ]
            else:
                # Categorical constants: n_unique == 1
                constant_cols = [col for col in df.select_dtypes(exclude="number").columns if df[col].nunique() == 1]
            errors_df = pd.DataFrame({"column": constant_cols, "nerrors": [1] * len(constant_cols)})
        else:
            # For NaN/null/inf detection - use vectorized operations
            numeric_df = df.select_dtypes(include="number")
            if "NaN" in kind:
                nerrors_series = numeric_df.isna().sum()
            elif "null" in kind:
                nerrors_series = numeric_df.isnull().sum()
            elif "infinite" in kind:
                nerrors_series = np.isinf(numeric_df).sum()
            else:
                nerrors_series = pd.Series(dtype=int)

            # Filter to non-zero and convert to DataFrame
            nerrors_series = nerrors_series[nerrors_series > 0].sort_values(ascending=False)
            errors_df = pd.DataFrame({"column": nerrors_series.index.tolist(), "nerrors": nerrors_series.values})
        nrows, ncols = df.shape

    # Log and handle errors
    if len(errors_df) > 0:
        if verbose:
            logger.info(f"Found {len(errors_df)} {kind} out of {ncols} columns:")
            logger.info("\n%s", errors_df)

        if drop_columns:
            cols_to_drop = errors_df["column"].to_list() if is_polars else errors_df["column"].tolist()
            if cols_to_drop:
                from .utils import filter_existing  # lazy: breaks cycle with .utils
                # Only drop columns that actually exist in the dataframe
                if is_polars:
                    existing_cols = filter_existing(df, cols_to_drop)
                    if existing_cols:
                        df = df.drop(existing_cols)
                else:
                    existing_cols = filter_existing(df, cols_to_drop)
                    if existing_cols:
                        df = df.drop(columns=existing_cols)
            if verbose:
                logger.info(f"Dropped {len(cols_to_drop)} {kind}.")
        elif fill_value is not None:
            if is_polars:
                if fill_func_name is None:
                    raise ValueError("fill_func_name is required for Polars DataFrames when fill_value is provided")
                if fill_func_name == "replace":
                    # For infinities: only ±inf are replaced. Gate to cs.float()
                    # rather than cs.numeric() because integer columns cannot
                    # hold ±inf (storage-wise impossible) AND polars 1.x raises
                    # ``InvalidOperationError: conversion from f64 to i32
                    # failed ... for [inf, -inf]`` if the replace touches an
                    # integer column - it tries to cast the search values
                    # down to the column dtype, which is undefined for inf.
                    # NaN is preserved here (.replace does not touch NaN);
                    # callers that also want NaN filled should run
                    # ``process_nans`` first.
                    df = df.with_columns(cs.float().replace([float("inf"), float("-inf")], fill_value))
                else:
                    df = df.with_columns(getattr(cs.numeric(), fill_func_name)(fill_value))
            else:
                # Restrict pandas fill to numeric columns -- mirrors the polars cs.numeric() gate.
                # Unrestricted df.fillna(0.0) raises on Categorical columns
                # ("Cannot setitem on a Categorical with a new category").
                num_cols = df.select_dtypes(include="number").columns
                if "NaN" in kind or "null" in kind:
                    df[num_cols] = df[num_cols].fillna(fill_value)
                elif "infinite" in kind:
                    df[num_cols] = df[num_cols].replace([float("inf"), float("-inf")], fill_value)
            if verbose:
                from ._ram_helpers import log_ram_usage  # local import: ._ram_helpers <-> .utils <-> ._nan_processing cycle

                logger.info(f"{kind.capitalize()}s filled with {fill_value} value.")
                log_ram_usage()

    return df


def process_nans(df: pl.DataFrame | pd.DataFrame, fill_value: float = 0.0, verbose: int = 1) -> pl.DataFrame | pd.DataFrame:
    """
    Process NaN values in numeric columns by filling them with a specified value.

    Args:
        df: Polars or pandas DataFrame to process
        fill_value: Value to replace NaNs with (default: 0.0)
        verbose: Verbosity level (default: 1)

    Returns:
        DataFrame with NaN values filled
    """
    return _process_special_values(
        df=df,
        expr_func=lambda: (cs.numeric().is_nan().sum() if isinstance(df, pl.DataFrame) else None),
        fill_func_name="fill_nan",
        kind="NaN",
        fill_value=fill_value,
        verbose=verbose,
    )


def process_nulls(df: pl.DataFrame | pd.DataFrame, fill_value: float = 0.0, verbose: int = 1) -> pl.DataFrame | pd.DataFrame:
    """
    Process NULL values in numeric columns by filling them with a specified value.

    Args:
        df: Polars or pandas DataFrame to process
        fill_value: Value to replace NULLs with (default: 0.0)
        verbose: Verbosity level (default: 1)

    Returns:
        DataFrame with NULL values filled
    """
    return _process_special_values(
        df=df,
        expr_func=lambda: (cs.numeric().is_null().sum() if isinstance(df, pl.DataFrame) else None),
        fill_func_name="fill_null",
        kind="null",
        fill_value=fill_value,
        verbose=verbose,
    )


def process_infinities(df: pl.DataFrame | pd.DataFrame, fill_value: float = 0.0, verbose: int = 1) -> pl.DataFrame | pd.DataFrame:
    """
    Process infinite values in numeric columns by replacing them with a specified value.

    Args:
        df: Polars or pandas DataFrame to process
        fill_value: Value to replace infinities with (default: 0.0)
        verbose: Verbosity level (default: 1)

    Returns:
        DataFrame with infinite values replaced
    """
    return _process_special_values(
        df=df,
        expr_func=lambda: (cs.numeric().is_infinite().sum() if isinstance(df, pl.DataFrame) else None),
        fill_func_name="replace",
        kind="infinite",
        fill_value=fill_value,
        verbose=verbose,
    )


def get_numeric_columns(df: pl.DataFrame | pd.DataFrame) -> list:
    """
    Get list of numeric column names from DataFrame schema without scanning data.

    Args:
        df: Polars or pandas DataFrame

    Returns:
        List of numeric column names

    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [1.0, 2.0]})
    >>> sorted(get_numeric_columns(df))
    ['a', 'c']
    """
    if isinstance(df, pl.DataFrame):
        return [name for name, dtype in df.schema.items() if dtype.is_numeric()]
    else:
        return df.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(df: pl.DataFrame | pd.DataFrame, include_string: bool = True) -> list:
    """
    Get list of categorical column names from DataFrame schema without scanning data.

    Args:
        df: Polars or pandas DataFrame
        include_string: Whether to include string columns (default True)

    Returns:
        List of categorical column names
    """
    if isinstance(df, pl.DataFrame):
        # Function-local import: strategies imports from utils, so a top-level
        # import here would form a strategies->utils->strategies cycle at module load.
        from .strategies import get_polars_cat_columns
        if include_string:
            return get_polars_cat_columns(df)
        else:
            # 2026-04-28: also include pl.Enum (instance-level dtype, doesn't
            # compare ``== pl.Categorical``). Without it, CB confidence
            # model receives pl.Enum cat columns as numeric features and
            # raises ``Unsupported data type Enum(...) for a numerical
            # feature column``. Surfaced default-seed c0043 / c0049 / c0050
            # (hgb / pl.Enum cat columns + confidence_analysis_cfg=True).
            return [
                name for name, dtype in df.schema.items()
                if dtype == pl.Categorical
                or (hasattr(pl, "Enum") and isinstance(dtype, pl.Enum))
            ]
    else:
        # Function-local import (see note above) -- breaks strategies↔utils cycle.
        from .strategies import PANDAS_CATEGORICAL_SELECT_DTYPES
        if include_string:
            return df.select_dtypes(
                include=list(PANDAS_CATEGORICAL_SELECT_DTYPES)
            ).columns.tolist()
        else:
            return df.select_dtypes(include=["category"]).columns.tolist()


def remove_constant_columns(df: pl.DataFrame | pd.DataFrame, verbose: int = 1) -> pl.DataFrame | pd.DataFrame:
    """
    Remove constant columns from DataFrame.

    Args:
        df: Input DataFrame
        verbose: Verbosity level

    Returns:
        DataFrame with constant columns removed

    Notes:
        - Numeric columns: where min == max
        - Non-numeric columns: where n_unique == 1

    Duplicates ``pyutilz.data.polarslib.drop_constant_columns`` (canonical
    implementation upstream). Kept in-tree for the mlframe-internal pandas
    code path and the polars eq_missing variant; consolidation requires an
    upstream pyutilz PR + version pin, deferred to that follow-up.
    """
    is_polars = isinstance(df, pl.DataFrame)

    if is_polars:
        # Process numeric columns (min == max). For an all-NULL numeric column
        # both min() and max() return None, and plain `None == None` in polars
        # is `null` (not True) -> the column slips past `min == max`. Use
        # `eq_missing` which treats null==null as True, collapsing "constant"
        # and "all-null" into one check without an OR branch. Same cost as
        # `min == max` alone (~1ms on 600k x 6; eq_missing is actually
        # marginally cheaper than the | null_count variant). Discovered
        # 2026-04-24 on fuzz c0117 (pandas->parquet->polars + inject_degenerate
        # -> robust_scale crashed on NoneType-NoneType).
        df = _process_special_values(
            df=df,
            expr_func=lambda: cs.numeric().min().eq_missing(cs.numeric().max()),
            kind="constant numeric columns",
            drop_columns=True,
            verbose=verbose,
        )

        # Process non-numeric columns (n_unique == 1)
        df = _process_special_values(
            df=df,
            expr_func=lambda: (cs.by_dtype(pl.String, pl.Categorical).n_unique() == 1),
            kind="constant non-numeric columns",
            drop_columns=True,
            verbose=verbose,
        )
    else:
        # Pandas: match Polars semantics (min==max for numeric; n_unique==1 for others).
        # Per-column loop measured faster than df.agg(['min','max']) -- see audit 2026-04-14.
        numeric_cols = df.select_dtypes(include="number").columns
        constant_num_cols = [col for col in numeric_cols if df[col].min() == df[col].max()]

        # All-NaN numeric columns: min==max yields NaN==NaN -> False, so they are NOT
        # flagged by the loop above. Treat them as constant too (no information).
        all_nan_num = [c for c in numeric_cols if c not in constant_num_cols and df[c].isna().all()]
        constant_num_cols = constant_num_cols + all_nan_num

        non_numeric_cols = df.select_dtypes(exclude="number").columns
        constant_cat_cols = []
        for col in non_numeric_cols:
            try:
                if df[col].nunique(dropna=False) <= 1:
                    constant_cat_cols.append(col)
            except TypeError:
                # Unhashable values (e.g. list/np.ndarray embeddings) -- can't be constant-checked.
                continue

        constant_cols = list(constant_num_cols) + list(constant_cat_cols)

        if constant_cols and verbose:
            logger.info(f"Removing {len(constant_cols)} constant columns: {constant_cols}")

        if constant_cols:
            df = df.drop(columns=constant_cols)

    return df


def batch_scan_constants_and_inf_polars(
    df: pl.DataFrame,
    detect_constant_numeric: bool = True,
    detect_constant_nonnumeric: bool = True,
    detect_inf: bool = True,
) -> dict:
    """Single-pass polars scan for constant + infinite-bearing column detection.

    Replaces three separate ``df.select(...)`` passes (one each for
    ``cs.numeric().is_infinite().sum()``, ``cs.numeric().min().eq_missing(max())``
    and ``cs.by_dtype(String, Categorical).n_unique() == 1``) with a single
    ``df.select(...)`` that bundles every aggregation. polars' query planner
    can fuse the scans into one data sweep, which materially helps when the
    frame is wide (post-polars-ds-pipeline + polynomial features can push
    column counts into the thousands). c0140 iter291 attributed 60.7s
    cumulative across the three sequential ``_process_special_values``
    calls in ``preprocess_dataframe``; the combined query is ~1.15x faster
    on synthetic 200k x 205 frames and proportionally larger on real-world
    wide post-pipeline frames where the constant-overhead per scan
    dominates.

    Returns a dict with keys:
      * ``constant_numeric``: list[str] of numeric column names with min==max
        (treating null==null as constant, so all-null numeric columns count).
        Only populated when ``detect_constant_numeric=True``.
      * ``constant_nonnumeric``: list[str] of string/categorical columns with
        ``n_unique == 1``. Only populated when ``detect_constant_nonnumeric=True``.
      * ``inf_counts``: dict[str, int] mapping numeric column name -> count of
        +/-inf rows. Only populated when ``detect_inf=True``. Filtered to
        columns with count > 0.

    Empty (zero-detection-flag) requests still return the dict skeleton so
    callers can unconditionally read the keys.
    """
    out: dict = {"constant_numeric": [], "constant_nonnumeric": [], "inf_counts": {}}
    if not isinstance(df, pl.DataFrame) or cs is None:
        return out
    exprs = []
    # Prefix-based parsing on result row: unique per kind so a real column
    # named ``foo`` cannot collide with our scan output. The prefixes use an
    # ``__mlf__`` marker that cannot clash with any sane real column name.
    _PFX_INF = "__mlf_inf__"
    _PFX_CN = "__mlf_cn__"
    _PFX_CS = "__mlf_cs__"
    # iter621 (perf): cache ``cs.numeric()`` so it's constructed once
    # instead of up to 3 times across the inf+min+max expressions. The
    # downstream selector ops (is_infinite, sum, min, max, eq_missing)
    # all derive from the cached selector. Saves ~0.5us per call site
    # in selector dispatch overhead. Bench c0013-shape 100k x 30 frame:
    # 23.2ms -> 23.0ms (1.01x; selector-build is a tiny slice of total
    # cost, the polars internal sweep dominates). Shipped anyway because
    # there's no downside and the code is shorter.
    num_sel = cs.numeric() if (detect_inf or detect_constant_numeric) else None
    if detect_inf:
        exprs.append(num_sel.is_infinite().sum().name.prefix(_PFX_INF))
    if detect_constant_numeric:
        exprs.append(num_sel.min().eq_missing(num_sel.max()).name.prefix(_PFX_CN))
    if detect_constant_nonnumeric:
        # bench-attempt-rejected (iter621, 2026-05-31): tried replacing
        # ``n_unique() == 1`` with ``min().eq_missing(max())`` to avoid
        # the hashset build for the constant-string detection. Bench at
        # 100k x 30 (15 num + 15 low-card cats / strings): 0.77x
        # REGRESSION on the low-cardinality categorical case (n_unique
        # on a Categorical with ~8 unique values is a dict-lookup-fast
        # path; min/max requires scanning all 100k values to find the
        # lex extremes). Only wins on high-cardinality string cols
        # (1.30x). The Categorical fast-path beats the min/max scan on
        # the common cat_feature_count axis (low cardinality), so kept
        # n_unique here.
        exprs.append(
            (cs.by_dtype(pl.String, pl.Categorical).n_unique() == 1).name.prefix(_PFX_CS)
        )
    if not exprs:
        return out
    result = df.select(*exprs)
    if result.height == 0:
        return out
    row = result.row(0, named=True)
    for key, value in row.items():
        if key.startswith(_PFX_INF):
            col = key[len(_PFX_INF):]
            if value is not None and value > 0:
                out["inf_counts"][col] = int(value)
        elif key.startswith(_PFX_CN):
            col = key[len(_PFX_CN):]
            if bool(value):
                out["constant_numeric"].append(col)
        elif key.startswith(_PFX_CS):
            col = key[len(_PFX_CS):]
            if bool(value):
                out["constant_nonnumeric"].append(col)
    return out


__all__ = [
    "process_nans",
    "process_nulls",
    "process_infinities",
    "get_numeric_columns",
    "get_categorical_columns",
    "remove_constant_columns",
    # filter_existing lives in .utils (lazy-imported here to break a cycle); not a
    # module-level attribute of _nan_processing, so it must not be in this __all__.
    "batch_scan_constants_and_inf_polars",
]
