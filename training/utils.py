"""
Utility functions for mlframe training pipeline.

Functions for RAM management, file I/O, dataframe conversions, and data cleaning.
"""

import logging
import os
from typing import Union, Optional
import dill
import numpy as np
import zstandard as zstd
import pandas as pd
import polars as pl
import pyarrow as pa
import polars.selectors as cs
from pyutilz.system import clean_ram
from mlframe.helpers import get_own_ram_usage

logger = logging.getLogger(__name__)


def log_ram_usage() -> None:
    """Log current RAM usage."""
    logger.info(f"Done. RAM usage: {get_own_ram_usage():.1f}GB.")


def log_phase(msg: str, n: int = 80) -> None:
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
        logger.info(f"Dropping {len(all_cols_to_drop)} column(s)...")

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


def save_mlframe_model(model: object, file: str, zstd_kwargs: dict = None, verbose: int = 1) -> bool:
    """
    Save a model to disk with zstd compression.

    Args:
        model: Model object to save
        file: Output file path
        zstd_kwargs: Compression settings (default: level=4, threads=-1)
        verbose: Verbosity level

    Returns:
        True if successful, False otherwise
    """
    if zstd_kwargs is None:
        zstd_kwargs = dict(level=4, write_checksum=True, write_content_size=True, threads=-1)
    try:
        with open(file, "wb") as f:
            compressor = zstd.ZstdCompressor(**zstd_kwargs)
            with compressor.stream_writer(f) as zf:
                dill.dump(model, zf)
        if verbose > 0:
            size_mb = os.path.getsize(file) / (1024 * 1024)
            logger.info(f"Model saved successfully to {file}. Size: {size_mb:.2f} Mb")
        return True
    except Exception as e:
        logger.error(f"Could not save model to file {file}: {e}")
        return False


def load_mlframe_model(file: str) -> object:
    """
    Load a model from disk with zstd decompression.

    Args:
        file: Input file path

    Returns:
        Model object or None if loading fails
    """
    try:
        with open(file, "rb") as f:
            decompressor = zstd.ZstdDecompressor()
            with decompressor.stream_reader(f) as zf:
                model = dill.load(zf)
        return model
    except Exception as e:
        logger.error(f"Could not load model from file {file}: {e}")
        return None


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
    assert isinstance(df, (pl.DataFrame, pl.Series)), "Input must be a Polars DataFrame or Series"

    tbl = df.to_arrow()

    fixed_cols = []
    for col in tbl.columns:
        if pa.types.is_dictionary(col.type):
            # Convert dictionary array to its string representation
            col = pa.compute.cast(col, pa.string())
        fixed_cols.append(col)

    tbl_fixed = pa.table(fixed_cols, names=tbl.column_names)

    pandas_df = tbl_fixed.to_pandas(
        types_mapper=pd.ArrowDtype,  # keep Arrow-backed columns for zero-copy
    )

    return pandas_df


def save_series_or_df(
    obj: Union[pd.Series, pd.DataFrame, pl.Series, pl.DataFrame],
    file: str,
    compression: str = "zstd",
    name: Optional[str] = None,
):
    """
    Save a pandas/polars Series or DataFrame to parquet.

    Args:
        obj: Series or DataFrame to save
        file: Output file path
        compression: Compression algorithm (default: zstd)
        name: Name for Series (if converting to DataFrame)
    """
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
    expr_func=None,
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
    is_pandas = isinstance(df, pd.DataFrame)

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
            errors_df = pl.DataFrame({"column": constant_cols, "nerrors": [1] * len(constant_cols)})
        else:
            if len(qos_df) > 0:
                errors_df = pl.DataFrame({"column": qos_df.columns, "nerrors": qos_df.row(0)}).filter(pl.col("nerrors") > 0).sort("nerrors", descending=True)
            else:
                errors_df = pl.DataFrame({"column": [], "nerrors": []})
        nrows, ncols = df.shape
    else:
        # Pandas: different handling
        if drop_columns:
            # For constant column detection
            if "numeric" in kind:
                # Numeric constants: min == max
                constant_cols = [col for col in df.select_dtypes(include='number').columns if df[col].min() == df[col].max()]
            else:
                # Categorical constants: n_unique == 1
                constant_cols = [col for col in df.select_dtypes(exclude='number').columns if df[col].nunique() == 1]
            errors_df = pd.DataFrame({"column": constant_cols, "nerrors": [1] * len(constant_cols)})
        else:
            # For NaN/null/inf detection
            numeric_cols = df.select_dtypes(include='number').columns
            if "NaN" in kind:
                nerrors = {col: df[col].isna().sum() for col in numeric_cols}
            elif "null" in kind:
                nerrors = {col: df[col].isnull().sum() for col in numeric_cols}
            elif "infinite" in kind:
                nerrors = {col: (~df[col].replace([float('inf'), float('-inf')], None).isnull()).sum() - (~df[col].isnull()).sum() for col in numeric_cols}
            else:
                nerrors = {}
            errors_df = pd.DataFrame({"column": list(nerrors.keys()), "nerrors": list(nerrors.values())})
            errors_df = errors_df[errors_df["nerrors"] > 0].sort_values("nerrors", ascending=False)
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
                    existing_cols = [col for col in cols_to_drop if col in df.columns]
                    if existing_cols:
                        df = df.drop(existing_cols)
                else:
                    existing_cols = [col for col in cols_to_drop if col in df.columns]
                    if existing_cols:
                        df = df.drop(columns=existing_cols)
            if verbose:
                logger.info(f"Dropped {len(cols_to_drop)} {kind}.")
        elif fill_value is not None:
            if is_polars:
                df = df.with_columns(getattr(cs.numeric(), fill_func_name)(fill_value))
            else:
                if "NaN" in kind or "null" in kind:
                    df = df.fillna(fill_value)
                elif "infinite" in kind:
                    df = df.replace([float('inf'), float('-inf')], fill_value)
            if verbose:
                logger.info(f"{kind.capitalize()}s filled with {fill_value} value.")
                log_ram_usage()

    return df


def process_nans(df: Union[pl.DataFrame, pd.DataFrame], fill_value: float = 0.0, verbose: int = 1):
    """Process NaN values in numeric columns."""
    return _process_special_values(
        df=df,
        expr_func=lambda: (cs.numeric().is_nan().sum() if isinstance(df, pl.DataFrame) else None),
        fill_func_name="fill_nan",
        kind="NaN",
        fill_value=fill_value,
        verbose=verbose,
    )


def process_nulls(df: Union[pl.DataFrame, pd.DataFrame], fill_value: float = 0.0, verbose: int = 1):
    """Process NULL values in numeric columns."""
    return _process_special_values(
        df=df,
        expr_func=lambda: (cs.numeric().is_null().sum() if isinstance(df, pl.DataFrame) else None),
        fill_func_name="fill_null",
        kind="null",
        fill_value=fill_value,
        verbose=verbose,
    )


def process_infinities(df: Union[pl.DataFrame, pd.DataFrame], fill_value: float = 0.0, verbose: int = 1):
    """Process infinite values in numeric columns."""
    return _process_special_values(
        df=df,
        expr_func=lambda: (cs.numeric().is_infinite().sum() if isinstance(df, pl.DataFrame) else None),
        fill_func_name="replace",
        kind="infinite",
        fill_value=fill_value,
        verbose=verbose,
    )


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
        # Pandas: faster constant column detection
        first = df.iloc[0]
        mask_equal = df.eq(first).all()
        mask_all_nan = df.isna().all()
        all_constant_cols = df.columns[mask_equal | mask_all_nan].tolist()

        # Separate numeric and non-numeric constant columns
        constant_num_cols = [col for col in all_constant_cols
                            if np.issubdtype(df[col].dtype, np.number)]
        constant_cat_cols = [col for col in all_constant_cols
                            if not np.issubdtype(df[col].dtype, np.number)]
        constant_cols = constant_num_cols + constant_cat_cols

        if constant_cols and verbose:
            logger.info(f"Removing {len(constant_cols)} constant columns: {constant_cols}")

        if constant_cols:
            df = df.drop(columns=constant_cols)

    return df


__all__ = [
    'log_ram_usage',
    'log_phase',
    'drop_columns_from_dataframe',
    'save_mlframe_model',
    'load_mlframe_model',
    'get_pandas_view_of_polars_df',
    'save_series_or_df',
    'process_nans',
    'process_nulls',
    'process_infinities',
    'remove_constant_columns',
]
