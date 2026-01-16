"""
Pipeline functions for mlframe training.

Handles Polars-ds and sklearn pipeline creation, fitting, and transformation.
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

import pandas as pd
import polars as pl
import polars.selectors as cs
from typing import Union, Optional, List, Tuple
from collections import Counter
from pyutilz.system import clean_ram
from pyutilz.pandaslib import ensure_dataframe_float32_convertability

from .utils import log_ram_usage
from .configs import PolarsPipelineConfig


def prepare_df_for_catboost(df: pd.DataFrame, cat_features: List[str]) -> None:
    """
    Prepare categorical features for CatBoost.

    Args:
        df: DataFrame (modified in-place)
        cat_features: List of categorical feature names
    """
    for col in cat_features:
        if col in df.columns:
            if df[col].dtype.name not in ["category"]:
                df[col] = df[col].astype("category")


def create_polarsds_pipeline(
    train_df: pl.DataFrame,
    config: PolarsPipelineConfig,
    pipeline_name: str = "feature_pipeline",
    verbose: int = 1,
):
    """
    Create a Polars-ds pipeline for scaling and encoding.

    Args:
        train_df: Training DataFrame (Polars)
        config: Pipeline configuration
        pipeline_name: Name for the pipeline
        verbose: Verbosity level

    Returns:
        Materialized PdsPipeline or None if polars-ds not available
    """
    try:
        from polars_ds.pipeline import Pipeline as PdsPipeline, Blueprint as PdsBlueprint
    except Exception as e:
        logger.warning(f"Could not import polars-ds: {e}")
        return None

    if verbose:
        logger.info(f"Creating Polars-ds pipeline...")

    # Build blueprint
    bp = PdsBlueprint(train_df, name=pipeline_name)

    # Add scaling
    if config.scaler_name:
        if config.scaler_name == "robust":
            bp = bp.robust_scale(cs.numeric(), q_low=config.robust_q_low, q_high=config.robust_q_high)
        else:
            bp = bp.scale(cs.numeric(), method=config.scaler_name)

    # Add categorical encoding
    if config.categorical_encoding == "ordinal":
        bp = bp.ordinal_encode(cols=None, null_value=-1, unknown_value=-2)
    elif config.categorical_encoding == "onehot":
        bp = bp.one_hot_encode(cols=None, drop_first=False, drop_cols=True)
    # Add more encoding methods as needed

    # Convert int to float32 for better compatibility
    bp = bp.int_to_float(f32=True)

    # Materialize the pipeline
    pipeline = bp.materialize()
    clean_ram()

    if verbose:
        log_ram_usage()

    return pipeline


def fit_and_transform_pipeline(
    train_df: Union[pd.DataFrame, pl.DataFrame],
    val_df: Optional[Union[pd.DataFrame, pl.DataFrame]],
    test_df: Optional[Union[pd.DataFrame, pl.DataFrame]],
    config: PolarsPipelineConfig,
    ensure_float32: bool = True,
    verbose: int = 1,
) -> Tuple[Union[pd.DataFrame, pl.DataFrame], Optional[Union[pd.DataFrame, pl.DataFrame]], Optional[Union[pd.DataFrame, pl.DataFrame]], object, List[str]]:
    """
    Fit and apply a data pipeline to train/val/test splits.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame (optional)
        test_df: Test DataFrame (optional)
        config: Pipeline configuration
        ensure_float32: Whether to ensure float32 dtypes
        verbose: Verbosity level

    Returns:
        Tuple of (train_df, val_df, test_df, pipeline, cat_features)
    """
    pipeline = None
    cat_features = []

    # Handle Polars DataFrames with polars-ds
    if isinstance(train_df, pl.DataFrame) and config.use_polarsds_pipeline:
        pipeline = create_polarsds_pipeline(train_df, config, verbose=verbose)

        if pipeline is not None:
            if verbose:
                logger.info(f"Applying Polars-ds pipeline...")

            # Transform all splits and ensure float32 dtypes
            train_df = pipeline.transform(train_df)
            if ensure_float32:
                train_df = ensure_dataframe_float32_convertability(train_df)

            if val_df is not None and len(val_df) > 0:
                val_df = pipeline.transform(val_df)
                if ensure_float32:
                    val_df = ensure_dataframe_float32_convertability(val_df)

            if test_df is not None and len(test_df) > 0:
                test_df = pipeline.transform(test_df)
                if ensure_float32:
                    test_df = ensure_dataframe_float32_convertability(test_df)

            if verbose:
                logger.info(f"train_df dtypes after pipeline: {Counter(train_df.dtypes)}")

        # Detect categorical features from schema (works whether pipeline succeeded or not)
        # This ensures cat_features is populated even if polars-ds is not available
        cat_features = [name for name, dtype in train_df.schema.items() if dtype in (pl.Categorical, pl.Utf8, pl.String)]

    # Handle Polars DataFrames without polars-ds pipeline - just detect cat_features
    elif isinstance(train_df, pl.DataFrame) and not config.use_polarsds_pipeline:
        # Detect categorical features from schema (no transformation, just detection)
        cat_features = [name for name, dtype in train_df.schema.items() if dtype in (pl.Categorical, pl.Utf8, pl.String)]
        if verbose and cat_features:
            logger.info(f"Detected {len(cat_features)} categorical features from Polars schema: {cat_features}")

    # Handle pandas DataFrames with sklearn-style pipeline
    elif isinstance(train_df, pd.DataFrame):
        # Identify categorical features
        cat_features = [
            col for col in train_df.columns if train_df[col].dtype.name in ("object", "category", "string", "string[pyarrow]", "large_string[pyarrow]")
        ]

        # Apply categorical encoding if specified (for models that don't support categorical natively)
        if cat_features and config.categorical_encoding in ["ordinal", "onehot"]:
            if verbose:
                logger.info(f"Applying {config.categorical_encoding} encoding to {len(cat_features)} categorical features...")

            from category_encoders import OrdinalEncoder, OneHotEncoder

            # Create appropriate encoder
            if config.categorical_encoding == "ordinal":
                encoder = OrdinalEncoder(cols=cat_features, handle_unknown="value", handle_missing="value")
            else:  # onehot
                encoder = OneHotEncoder(cols=cat_features, use_cat_names=True, drop_invariant=False)

            # Fit on train and transform all splits
            train_df = encoder.fit_transform(train_df)
            if val_df is not None and len(val_df) > 0:
                val_df = encoder.transform(val_df)
            if test_df is not None and len(test_df) > 0:
                test_df = encoder.transform(test_df)

            pipeline = encoder  # Store encoder as pipeline

            # After encoding, cat_features are no longer categorical (they're numeric)
            cat_features = []

        # Prepare categorical features for CatBoost (if not already encoded)
        elif cat_features:
            if verbose:
                logger.info(f"Preparing {len(cat_features)} categorical features for CatBoost...")

            for df in [train_df, val_df, test_df]:
                if df is not None and len(df) > 0:
                    prepare_df_for_catboost(df, cat_features)

    # Clean up empty validation/test sets
    if val_df is not None and len(val_df) == 0:
        val_df = None

    if test_df is not None and len(test_df) == 0:
        test_df = None

    clean_ram()
    if verbose:
        log_ram_usage()

    return train_df, val_df, test_df, pipeline, cat_features


__all__ = [
    "prepare_df_for_catboost",
    "create_polarsds_pipeline",
    "fit_and_transform_pipeline",
]
