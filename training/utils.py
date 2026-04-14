"""
Utility functions for mlframe training pipeline.

Functions for RAM management, file I/O, dataframe conversions, and data cleaning.
"""

# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------


import os
from textwrap import shorten
from typing import Union, Optional, Callable

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import polars.selectors as cs
from pyutilz.system import clean_ram
from mlframe.helpers import get_own_ram_usage


def log_ram_usage() -> None:
    """Log current RAM usage."""
    logger.info(f"Done. RAM usage: {get_own_ram_usage():.1f}GB.")


def clean_ram_and_gpu(verbose: bool = False) -> None:
    """
    Clean both CPU RAM and GPU memory.

    Combines pyutilz.clean_ram() with GPU memory cleanup.
    Call this after model training to free memory before training next model.

    Args:
        verbose: If True, log memory stats after cleanup
    """
    import gc

    # Clean CPU RAM first
    clean_ram()

    # Clean GPU memory if PyTorch CUDA is available
    try:
        import torch

        if torch.cuda.is_available():
            # Synchronize all CUDA streams before cleanup
            torch.cuda.synchronize()
            # Empty the CUDA memory cache
            torch.cuda.empty_cache()
            # Force garbage collection again after GPU cleanup
            gc.collect()

            if verbose:
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                logger.info(f"GPU memory after cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    except ImportError:
        pass  # PyTorch not installed


def estimate_df_size_mb(df) -> float:
    """Estimated in-memory size of a Polars/pandas DataFrame in MB."""
    if isinstance(df, pl.DataFrame):
        return float(df.estimated_size("mb"))
    if isinstance(df, pd.DataFrame):
        return float(df.memory_usage(deep=True).sum() / 1024**2)
    return 0.0


def get_process_rss_mb() -> float:
    """Current process RSS in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024**2
    except ImportError:
        return 0.0


def should_clean_ram(baseline_rss_mb: float, df_size_mb: float, min_growth_mb: float = 500.0) -> bool:
    """True iff a clean_ram call (~0.6s) is likely justified.

    Triggers when either:
      - RSS grew beyond baseline by max(min_growth_mb, 30% of DF size) — accumulated
        temp state worth collecting; OR
      - free system RAM < 2x DF size — OOM risk, gc may release Arrow buffers.
    """
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / 1024**2
        free_mb = psutil.virtual_memory().available / 1024**2
    except Exception as e:
        logger.debug("should_clean_ram: RAM measurement failed, falling back to clean", exc_info=e)
        return True  # can't measure → fall back to cleaning
    growth = rss_mb - baseline_rss_mb
    return (growth > max(min_growth_mb, 0.3 * df_size_mb)) or (free_mb < 2 * df_size_mb)


def maybe_clean_ram_and_gpu(
    baseline_rss_mb: float,
    df_size_mb: float,
    verbose: bool = False,
    reason: str = "",
) -> bool:
    """Call clean_ram_and_gpu only when RAM metrics indicate it's worthwhile.

    On small DFs this avoids 0.6s of pure overhead per call; on large production
    DFs (or when the process is growing) it still fires at every site.
    """
    if should_clean_ram(baseline_rss_mb, df_size_mb):
        clean_ram_and_gpu(verbose=verbose)
        if verbose:
            logger.info(f"  clean_ram fired ({reason})" if reason else "  clean_ram fired")
        return True
    return False


def filter_existing(df, cols) -> list:
    """Return only the column names from `cols` that exist in `df.columns`.

    Preserves input order. Accepts any iterable of column names. When `df`
    has no `columns` attribute (e.g. numpy ndarray), returns an empty list
    so callers can safely chain without pre-checks.
    """
    columns = getattr(df, "columns", None)
    if columns is None:
        return []
    existing = set(columns)
    return [c for c in cols if c in existing]


def log_phase(msg: str, n: int = 160) -> None:
    """Log a phase separator with message."""
    logger.info("-" * n)
    logger.info(msg)
    logger.info("-" * n)


def drop_columns_from_dataframe(
    df: Union[pd.DataFrame, pl.DataFrame],
    additional_columns_to_drop: Optional[list] = None,
    config_drop_columns: Optional[list] = None,
    verbose: int = 1,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Drop columns from dataframe.

    Args:
        df: DataFrame to drop columns from
        additional_columns_to_drop: Additional columns to drop (e.g., from preprocessor)
        config_drop_columns: Columns from config to drop
        verbose: Verbosity level

    Returns:
        DataFrame with columns dropped
    """
    if not additional_columns_to_drop and not config_drop_columns:
        return df

    all_cols_to_drop = []
    if additional_columns_to_drop:
        all_cols_to_drop.extend(additional_columns_to_drop)
    if config_drop_columns:
        all_cols_to_drop.extend(config_drop_columns)

    # Remove duplicates
    all_cols_to_drop = set(all_cols_to_drop)

    if not all_cols_to_drop:
        return df

    if verbose:
        logger.info(f"Dropping {len(all_cols_to_drop)} column(s): {shorten(','.join(all_cols_to_drop),250)}...")

    if isinstance(df, pl.DataFrame):
        df = df.drop(all_cols_to_drop, strict=False)
    else:
        existing_cols = all_cols_to_drop.intersection(set(df.columns))
        if existing_cols:
            df = df.drop(columns=existing_cols)

    clean_ram()
    if verbose:
        log_ram_usage()

    return df


# NOTE: save_mlframe_model and load_mlframe_model are in io.py
# Use: from .io import save_mlframe_model, load_mlframe_model


def get_pandas_view_of_polars_df(df: pl.DataFrame) -> pd.DataFrame:
    """
    Return a zero-copy (Arrow-backed) pandas DataFrame view of a Polars DataFrame.

    Args:
        df: Polars DataFrame

    Returns:
        Zero-copy pandas DataFrame view

    Notes:
        - Numeric, boolean, string columns: zero-copy Arrow view
        - Categorical (dictionary) columns: converted to string
    """
    if not isinstance(df, (pl.DataFrame, pl.Series)):
        raise TypeError(f"Input must be a Polars DataFrame or Series, got {type(df).__name__}")

    tbl = df.to_arrow()

    # Note: short-circuit on "no dictionary columns" was benchmarked 2026-04-14 and
    # delivered only 1.16x on pure-numeric workloads (below 1.2x threshold), so the
    # per-column scan is retained for code uniformity.
    fixed_cols = []
    for col in tbl.columns:
        if pa.types.is_dictionary(col.type):
            # Convert dictionary array to its string representation
            col = pa.compute.cast(col, pa.string())
        fixed_cols.append(col)

    tbl_fixed = pa.table(fixed_cols, names=tbl.column_names)

    # Use numpy-backed pandas (types_mapper=None) for broad model compatibility.
    # LightGBM (and some other sklearn-family models) reject pandas columns with
    # ArrowDtype (e.g. 'float[pyarrow]'). Numpy-backed columns are still near-zero-copy
    # for numeric types because Arrow's to_pandas picks up zero-copy numpy views when possible.
    pandas_df = tbl_fixed.to_pandas()

    return pandas_df


def save_series_or_df(
    obj: Union[pd.Series, pd.DataFrame, pl.Series, pl.DataFrame],
    file: str,
    compression: str = "zstd",
    name: Optional[str] = None,
) -> None:
    """
    Save a pandas/polars Series or DataFrame to parquet.

    Args:
        obj: Series or DataFrame to save
        file: Output file path
        compression: Compression algorithm (default: zstd)
        name: Name for Series (if converting to DataFrame)

    Raises:
        FileNotFoundError: If the parent directory does not exist.
    """
    parent_dir = os.path.dirname(file)
    if parent_dir and not os.path.exists(parent_dir):
        raise FileNotFoundError(f"Parent directory does not exist: {parent_dir}")

    if isinstance(obj, (pd.Series, pl.Series)):
        if name:
            obj = obj.to_frame(name=name)
        else:
            obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        obj.to_parquet(file, compression=compression)
    elif isinstance(obj, pl.DataFrame):
        obj.write_parquet(file, compression=compression)


def _process_special_values(
    df: Union[pl.DataFrame, pd.DataFrame],
    expr_func: Optional[Callable[[], pl.Expr]] = None,
    fill_func_name: Optional[str] = None,
    kind: str = "",
    fill_value: Optional[float] = None,
    drop_columns: bool = False,
    verbose: int = 1,
) -> Union[pl.DataFrame, pd.DataFrame]:
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
                # Numeric constants: min == max. Per-column loop measured faster than
                # a single df.agg(['min','max']) pass on both narrow-tall and wide-short
                # shapes (benchmark 2026-04-14: 10x slower on 1000x200, 0.74x on 100k x 50).
                # NaN-skipping min/max preserves Polars semantics ([1.0, NaN, 1.0] constant).
                constant_cols = [col for col in df.select_dtypes(include="number").columns if df[col].min() == df[col].max()]
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
            logger.info(f"Found {len(errors_df)} columns with {kind}s out of {ncols} columns:")
            logger.info(f"\n{errors_df}")

        if drop_columns:
            cols_to_drop = errors_df["column"].to_list() if is_polars else errors_df["column"].tolist()
            if cols_to_drop:
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
                    # For infinities: only ±inf are replaced. NaN is preserved here
                    # (cs.numeric().replace does not touch NaN). Callers that also want
                    # NaN filled should run `process_nans` first.
                    df = df.with_columns(cs.numeric().replace([float("inf"), float("-inf")], fill_value))
                else:
                    df = df.with_columns(getattr(cs.numeric(), fill_func_name)(fill_value))
            else:
                if "NaN" in kind or "null" in kind:
                    df = df.fillna(fill_value)
                elif "infinite" in kind:
                    df = df.replace([float("inf"), float("-inf")], fill_value)
            if verbose:
                logger.info(f"{kind.capitalize()}s filled with {fill_value} value.")
                log_ram_usage()

    return df


def process_nans(df: Union[pl.DataFrame, pd.DataFrame], fill_value: float = 0.0, verbose: int = 1) -> Union[pl.DataFrame, pd.DataFrame]:
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


def process_nulls(df: Union[pl.DataFrame, pd.DataFrame], fill_value: float = 0.0, verbose: int = 1) -> Union[pl.DataFrame, pd.DataFrame]:
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


def process_infinities(df: Union[pl.DataFrame, pd.DataFrame], fill_value: float = 0.0, verbose: int = 1) -> Union[pl.DataFrame, pd.DataFrame]:
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


def get_numeric_columns(df: Union[pl.DataFrame, pd.DataFrame]) -> list:
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


def get_categorical_columns(df: Union[pl.DataFrame, pd.DataFrame], include_string: bool = True) -> list:
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
        # import here would form a strategies→utils→strategies cycle at module load.
        from .strategies import get_polars_cat_columns
        if include_string:
            return get_polars_cat_columns(df)
        else:
            return [name for name, dtype in df.schema.items() if dtype == pl.Categorical]
    else:
        # Function-local import (see note above) — breaks strategies↔utils cycle.
        from .strategies import PANDAS_CATEGORICAL_DTYPES
        if include_string:
            return df.select_dtypes(include=list(PANDAS_CATEGORICAL_DTYPES)).columns.tolist()
        else:
            return df.select_dtypes(include=["category"]).columns.tolist()


def remove_constant_columns(df: Union[pl.DataFrame, pd.DataFrame], verbose: int = 1) -> Union[pl.DataFrame, pd.DataFrame]:
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
    """
    is_polars = isinstance(df, pl.DataFrame)

    if is_polars:
        # Process numeric columns (min == max)
        df = _process_special_values(
            df=df,
            expr_func=lambda: (cs.numeric().min() == cs.numeric().max()),
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
        # Per-column loop measured faster than df.agg(['min','max']) — see audit 2026-04-14.
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
                # Unhashable values (e.g. list/np.ndarray embeddings) — can't be constant-checked.
                continue

        constant_cols = list(constant_num_cols) + list(constant_cat_cols)

        if constant_cols and verbose:
            logger.info(f"Removing {len(constant_cols)} constant columns: {constant_cols}")

        if constant_cols:
            df = df.drop(columns=constant_cols)

    return df


__all__ = [
    "log_ram_usage",
    "log_phase",
    "clean_ram_and_gpu",
    "estimate_df_size_mb",
    "get_process_rss_mb",
    "should_clean_ram",
    "maybe_clean_ram_and_gpu",
    "drop_columns_from_dataframe",
    "get_pandas_view_of_polars_df",
    "save_series_or_df",
    "process_nans",
    "process_nulls",
    "process_infinities",
    "get_numeric_columns",
    "get_categorical_columns",
    "remove_constant_columns",
    "filter_existing",
]
