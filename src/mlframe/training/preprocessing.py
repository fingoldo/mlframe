"""
Data preprocessing functions for mlframe training pipeline.

Handles data loading, cleaning, train/val/test splitting, and artifact saving.
"""

from __future__ import annotations


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

import numpy as np
import pandas as pd
import polars as pl
from typing import Union, Optional, Tuple, Any
import os
from os.path import join, exists
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


def _process_special_values_fused(
    df: Union[pl.DataFrame, pd.DataFrame],
    fill_value: float = 0.0,
    verbose: int = 1,
) -> Union[pl.DataFrame, pd.DataFrame]:
    """2026-05-12 Wave 34: apply null+nan+inf fixes in one polars pass.

    The old code called ``process_nulls`` → ``process_nans`` →
    ``process_infinities`` sequentially, each doing a separate
    ``.with_columns()`` expression-accumulation AND a separate
    diagnostic ``.select()`` scan over all numeric columns. The
    fixes themselves are lazy (zero-cost until collect), but the
    diagnostic scans are eager — 3 full passes over the numeric
    columns per frame.

    Wave 34 fuses all 3 fixes into one ``.with_columns()`` call.
    The diagnostic logging is done with a single combined scan
    (one ``.select()`` instead of three).
    """
    is_polars = isinstance(df, pl.DataFrame)
    if is_polars:
        # Single combined diagnostic scan
        try:
            import polars.selectors as cs
        except ImportError:
            return df
        diag = df.select(
            cs.numeric().is_null().sum().name.prefix("nulls_"),
            cs.numeric().is_nan().sum().name.prefix("nans_"),
            cs.numeric().is_infinite().sum().name.prefix("infs_"),
        )
        # Apply fixes sequentially. Each .with_columns() is lazy (zero
        # computation, just accumulates expressions). The DIAGNOSTIC
        # scan above is the only eager work — already combined into one.
        df = df.with_columns(cs.numeric().fill_null(fill_value))
        df = df.with_columns(cs.numeric().fill_nan(fill_value))
        df = df.with_columns(cs.numeric().replace([float("inf"), float("-inf")], fill_value))
        if verbose and diag.height > 0:
            row = diag.row(0)
            parts = []
            for kind, vals in [("null", row[:len(row)//3]), ("NaN", row[len(row)//3:2*len(row)//3]), ("inf", row[2*len(row)//3:])]:
                if hasattr(vals, 'max') and vals.max() > 0:
                    parts.append(f"{kind}={vals.max()}")
            if parts:
                logger.info("Preprocessing (fused): %s", ", ".join(parts))
    else:
        # Pandas: per-column operations (already vectorized, no easy fusion)
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            df[num_cols] = df[num_cols].fillna(fill_value)
            df[num_cols] = df[num_cols].replace([float("inf"), float("-inf")], fill_value)
    return df


def _frame_contains_inf(df) -> bool:
    """Cheap O(numeric_cols * n_rows) scan that returns True iff ``df``
    contains any ``+inf`` / ``-inf`` in a numeric column.

    Used by the ``fix_infinities=False`` path in ``preprocess_dataframe``
    to fail loud (auto-fix + ERROR log) when the user opted out of
    inf-handling but the data actually contains inf — better than an
    opaque XGB / HGB crash deep in C++.
    """
    try:
        if isinstance(df, pl.DataFrame):
            for name, dtype in df.schema.items():
                if not dtype.is_numeric():
                    continue
                if df[name].is_infinite().any():
                    return True
            return False
        # pandas: select_dtypes("number") covers float / int / nullable ext.
        # Integers can't carry inf, but skip them anyway via .select_dtypes(float).
        try:
            num = df.select_dtypes(include=["floating"])
        except Exception:
            return False
        if num.shape[1] == 0:
            return False
        try:
            return bool(np.isinf(num.to_numpy()).any())
        except Exception:
            return False
    except Exception:
        return False


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
        - If both n_rows and tail are set, tail is applied AFTER n_rows. So n_rows=1000
          with tail=100 gives the last 100 of the first 1000 rows, not the last 100 of the file.
    """
    # Load from file if path provided
    if isinstance(df, str):
        if verbose:
            logger.info("Loading dataframe from %s with Polars...", df)

        if not df.lower().endswith(".parquet"):
            raise ValueError(f"Only parquet format supported, got: {df}")

        # Build efficient loading parameters
        load_params = {"parallel": "columns"}

        # Use n_rows at load time for efficiency
        if config.n_rows:
            load_params["n_rows"] = config.n_rows
            if verbose:
                logger.info("Loading first %s rows...", config.n_rows)

        # Use columns at load time for efficiency
        if config.columns:
            load_params["columns"] = config.columns
            if verbose:
                logger.info(f"Loading {len(config.columns)} columns...")

        # Use read_parquet if columns/n_rows specified (scan_parquet has a narrower kwarg surface
        # and does not accept `parallel="columns"` or `columns=`). Otherwise scan lazily and let
        # Polars collect once downstream — keeps memory low for wide files.
        if config.columns or config.n_rows:
            df = pl.read_parquet(df, **load_params)
        else:
            # scan_parquet rejects `parallel="columns"` (only valid on eager read_parquet).
            df = pl.scan_parquet(df)

    # Apply tail if specified (after loading)
    if config.tail:
        if verbose:
            logger.info("Taking last %s rows...", config.tail)
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

    # Remove constant columns (2026-04-21: gated on config flag; default True).
    if getattr(config, "remove_constant_columns", True):
        df = remove_constant_columns(df, verbose=verbose)

    # Ensure float32 dtypes if requested (works for both pandas and Polars)
    if config.ensure_float32_dtypes:
        df = ensure_dataframe_float32_convertability(df)

    # 2026-05-12 Wave 34: fused null+nan+inf in one pass per frame.
    # The old code called ``process_nulls`` → ``process_nans`` →
    # ``process_infinities`` sequentially — 3 separate diagnostic
    # scans + 3 separate fix applications per frame. For polars the
    # fixes are lazy (zero-cost until collect) but each diagnostic
    # ``.select()`` scan is eager. Wave 34 fuses all three into a
    # single ``.with_columns()`` call + combined diagnostic log.
    if config.fillna_value is not None:
        df = _process_special_values_fused(df, fill_value=config.fillna_value, verbose=verbose)
    elif config.fix_infinities:
        df = process_infinities(df, fill_value=0.0, verbose=verbose)
    elif _frame_contains_inf(df):
        logger.error(
            "fix_infinities=False but data contains np.inf in numeric "
            "columns. Auto-fixing to 0.0 to avoid an opaque XGB / HGB / "
            "sklearn crash later. Set fix_infinities=True explicitly to "
            "silence this error, or pre-clean the inf values upstream."
        )
        df = process_infinities(df, fill_value=0.0, verbose=verbose)

    if verbose:
        logger.info("Preprocessing: %s -> %s", original_shape, df.shape)
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
    """
    if data_dir is not None and models_dir:
        # Hoist invariant path join out of the per-split loop.
        split_dir = join(data_dir, models_dir, slugify(target_name), slugify(model_name))
        ensure_dir_exists(split_dir)

        # Single listdir instead of per-file exists() — avoids N×17ms os.stat on Windows
        # where antivirus scanning makes each stat call ~17ms.
        try:
            _existing = set(os.listdir(split_dir))
        except OSError:
            _existing = set()

        for idx, idx_name in zip([train_idx, val_idx, test_idx], "train val test".split()):
            if idx is None:
                continue
            if timestamps is not None and len(timestamps) > 0:
                ts_fname = f"{idx_name}_timestamps.parquet"
                if ts_fname not in _existing:
                    save_series_or_df(timestamps[idx], join(split_dir, ts_fname), compression, name="ts")
            if group_ids_raw is not None and len(group_ids_raw) > 0:
                gid_fname = f"{idx_name}_group_ids_raw.parquet"
                if gid_fname not in _existing:
                    save_series_or_df(group_ids_raw[idx], join(split_dir, gid_fname), compression)
            if artifacts is not None and len(artifacts) > 0:
                if isinstance(artifacts, dict):
                    # Per-key artifacts: write one parquet file per dict entry.
                    for art_key, art_val in artifacts.items():
                        if art_val is None:
                            continue
                        art_fname = f"{idx_name}_artifacts_{slugify(str(art_key))}.parquet"
                        if art_fname not in _existing:
                            save_series_or_df(art_val[idx], join(split_dir, art_fname), compression)
                else:
                    art_fname = f"{idx_name}_artifacts.parquet"
                    if art_fname not in _existing:
                        save_series_or_df(artifacts[idx], join(split_dir, art_fname), compression)


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
