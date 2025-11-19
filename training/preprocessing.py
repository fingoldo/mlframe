"""
Data preprocessing functions for mlframe training pipeline.

Handles data loading, cleaning, train/val/test splitting, and artifact saving.
"""

import logging
import numpy as np
import pandas as pd
import polars as pl
from typing import Union, Optional, Tuple, Any
from os.path import join, exists
from sklearn.model_selection import train_test_split
from pyutilz.pandaslib import ensure_dataframe_float32_convertability
from pyutilz.system import ensure_dir_exists
from pyutilz.strings import slugify

from .utils import (
    process_nans,
    process_nulls,
    process_infinities,
    remove_constant_columns,
    save_series_or_df,
    log_ram_usage,
)
from .configs import PreprocessingConfig, TrainingSplitConfig

logger = logging.getLogger(__name__)


def load_and_prepare_dataframe(
    df: Union[pl.DataFrame, str],
    config: PreprocessingConfig,
    verbose: int = 1,
) -> pl.DataFrame:
    """
    Load and prepare dataframe for training (Polars only).

    Args:
        df: Polars DataFrame or path to parquet file
        config: Preprocessing configuration
        verbose: Verbosity level

    Returns:
        Polars DataFrame

    Notes:
        - Only supports Polars (for efficiency)
        - Column dropping happens AFTER features_and_targets_extractor.transform() in core.py
          (columns might be needed by features_and_targets_extractor or created by it)
    """
    # Load from file if path provided
    if isinstance(df, str):
        if verbose:
            logger.info(f"Loading dataframe from {df} with Polars...")

        if not df.lower().endswith(".parquet"):
            raise ValueError(f"Only parquet format supported, got: {df}")

        # Build efficient loading parameters
        load_params = {"parallel": "columns"}

        # Use n_rows at load time for efficiency
        if config.n_rows:
            load_params["n_rows"] = config.n_rows
            if verbose:
                logger.info(f"Loading first {config.n_rows} rows...")

        # Use columns at load time for efficiency
        if config.columns:
            load_params["columns"] = config.columns
            if verbose:
                logger.info(f"Loading {len(config.columns)} columns...")

        # Use read_parquet if columns specified (scan_parquet doesn't support columns parameter)
        if config.columns or config.n_rows:
            df = pl.read_parquet(df, **load_params)
        else:
            df = pl.scan_parquet(df, **load_params)

    # Apply tail if specified (after loading)
    if config.tail:
        if verbose:
            logger.info(f"Taking last {config.tail} rows...")
        df = df.tail(config.tail)

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    if verbose:
        log_ram_usage()

    return df


def preprocess_dataframe(
    df: Union[pl.DataFrame, pd.DataFrame],
    config: PreprocessingConfig,
    verbose: int = 1,
) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Preprocess dataframe: handle nulls, NaNs, infinities, constants, dtypes.

    Args:
        df: Input DataFrame
        config: Preprocessing configuration
        verbose: Verbosity level

    Returns:
        Preprocessed DataFrame
    """
    original_shape = df.shape

    # Remove constant columns
    df = remove_constant_columns(df, verbose=verbose)

    # Ensure float32 dtypes if requested
    if config.ensure_float32_dtypes:
        if isinstance(df, pd.DataFrame):
            df = ensure_dataframe_float32_convertability(df)
        # Polars already uses efficient dtypes

    # Process nulls
    if config.fillna_value is not None:
        df = process_nulls(df, fill_value=config.fillna_value, verbose=verbose)
        df = process_nans(df, fill_value=config.fillna_value, verbose=verbose)

    # Process infinities
    if config.fix_infinities and config.fillna_value is not None:
        df = process_infinities(df, fill_value=config.fillna_value, verbose=verbose)

    if verbose:
        logger.info(f"Preprocessing: {original_shape} -> {df.shape}")
        log_ram_usage()

    return df



def save_split_artifacts(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    timestamps: Optional[Union[pd.Series, pl.Series]],
    group_ids_raw: Optional[Union[np.ndarray, pd.Series, pl.Series]],
    artifacts: Optional[Any],
    data_dir: Optional[str],
    models_dir: str,
    target_name: str,
    model_name: str,
    compression: str = "zstd",
):
    """
    Save split artifacts (timestamps, group_ids, artifacts) for each split.

    Args:
        train_idx: Training indices
        val_idx: Validation indices
        test_idx: Test indices
        timestamps: Timestamp series
        group_ids_raw: Group ID series/array
        artifacts: Additional artifacts from features_and_targets_extractor
        data_dir: Base data directory
        models_dir: Models subdirectory
        target_name: Target name (for directory structure)
        model_name: Model name (for directory structure)
        compression: Compression algorithm
        verbose: Verbosity level
    """
    if data_dir is not None and models_dir:
        ensure_dir_exists(join(data_dir, models_dir, slugify(target_name), slugify(model_name)))
        for idx, idx_name in zip([train_idx, val_idx, test_idx], "train val test".split()):
            if idx is None:
                continue
            if timestamps is not None and len(timestamps) > 0:
                ts_file = join(data_dir, models_dir, slugify(target_name), slugify(model_name), f"{idx_name}_timestamps.parquet")
                if not exists(ts_file):
                    save_series_or_df(timestamps[idx], ts_file, compression, name="ts")
            if group_ids_raw is not None and len(group_ids_raw) > 0:
                gid_file = join(data_dir, models_dir, slugify(target_name), slugify(model_name), f"{idx_name}_group_ids_raw.parquet")
                if not exists(gid_file):
                    save_series_or_df(group_ids_raw[idx], gid_file, compression)
            if artifacts is not None and len(artifacts) > 0:
                art_file = join(data_dir, models_dir, slugify(target_name), slugify(model_name), f"{idx_name}_artifacts.parquet")
                if not exists(art_file):
                    save_series_or_df(artifacts[idx], art_file, compression)


def create_split_dataframes(
    df: Union[pd.DataFrame, pl.DataFrame],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[Union[pd.DataFrame, pl.DataFrame], Union[pd.DataFrame, pl.DataFrame], Union[pd.DataFrame, pl.DataFrame]]:
    """
    Create train, val, test dataframes from indices.

    Args:
        df: Original DataFrame
        train_idx: Training indices
        val_idx: Validation indices
        test_idx: Test indices

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    is_polars = isinstance(df, pl.DataFrame)

    if is_polars:
        train_df = df[train_idx]
        val_df = df[val_idx] if len(val_idx) > 0 else pl.DataFrame()
        test_df = df[test_idx] if len(test_idx) > 0 else pl.DataFrame()
    else:
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx] if len(val_idx) > 0 else pd.DataFrame()
        test_df = df.iloc[test_idx] if len(test_idx) > 0 else pd.DataFrame()

    return train_df, val_df, test_df


__all__ = [
    "load_and_prepare_dataframe",
    "preprocess_dataframe",
    "save_split_artifacts",
    "create_split_dataframes",
]
