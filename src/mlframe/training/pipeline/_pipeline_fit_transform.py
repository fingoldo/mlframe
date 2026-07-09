"""Carved out of ``mlframe.training.pipeline``.

Re-imported at the parent's module bottom so historical
``from mlframe.training.pipeline import apply_preprocessing_extensions``
resolves transparently.
"""
from __future__ import annotations

# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging
from timeit import default_timer as timer

import pandas as pd
import polars as pl
from typing import Union, Optional, List, Tuple
from collections import Counter
from ..utils import maybe_clean_ram_adaptive, log_ram_usage
from pyutilz.pandaslib import ensure_dataframe_float32_convertability

from ..configs import PreprocessingBackendConfig
from ..strategies import PANDAS_CATEGORICAL_DTYPES, get_polars_cat_columns

logger = logging.getLogger("mlframe.training.pipeline")

# Thread-count env vars must be set BEFORE Julia/PySR boots; we defer the set until the first
# ``_apply_pysr_fe`` call so importers who never touch PySR don't get their env mutated.


def fit_and_transform_pipeline(
    train_df: Union[pd.DataFrame, pl.DataFrame],
    val_df: Optional[Union[pd.DataFrame, pl.DataFrame]],
    test_df: Optional[Union[pd.DataFrame, pl.DataFrame]],
    config: PreprocessingBackendConfig,
    ensure_float32: bool = True,
    verbose: int = 1,
    text_features: Optional[List[str]] = None,
    embedding_features: Optional[List[str]] = None,
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
        text_features: Columns to exclude from encoding/scaling (free-text for CatBoost)
        embedding_features: Columns to exclude from encoding/scaling (list-of-float vectors)

    Returns:
        Tuple of (train_df, val_df, test_df, pipeline, cat_features)
    """
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from . import _warn_on_schema_drift, create_polarsds_pipeline, prepare_dfs_for_catboost_joint
    # Columns that must be excluded from encoding (they're not categoricals)
    _exclude_from_encoding = set(text_features or []) | set(embedding_features or [])
    pipeline = None
    cat_features = []

    # Datetime column decomposition is performed in
    # ``train_mlframe_models_suite`` (core.py) BEFORE the pre-pipeline
    # polars-clone point so the clone inherits the numeric decomposition.
    # Calling it here too would be a no-op -- the caller has already
    # decomposed any datetime columns by the time we run.

    # Handle Polars DataFrames with polars-ds
    _polarsds_fell_back_to_sklearn = False
    if isinstance(train_df, pl.DataFrame) and config.prefer_polarsds:
        # Detect cat_features from the ORIGINAL schema before the pipeline possibly
        # ordinal/one-hot-encodes them to numeric (which would erase their categorical dtype).
        _orig_cat_features = [c for c in get_polars_cat_columns(train_df) if c not in _exclude_from_encoding]
        pipeline = create_polarsds_pipeline(
            train_df, config, verbose=verbose,
            exclude_from_encoding=_exclude_from_encoding,
        )

        # ``fallback_to_sklearn``: polars-ds returned no pipeline (unavailable / build failed).
        # When enabled, convert the splits to pandas and route through the sklearn pandas branch
        # below so scaling/encoding still happen instead of silently passing the frame through raw.
        if pipeline is None and config.fallback_to_sklearn:
            logger.warning("polars-ds pipeline unavailable; falling back to sklearn pandas backend (fallback_to_sklearn=True)")
            train_df = train_df.to_pandas()
            if val_df is not None and isinstance(val_df, pl.DataFrame):
                val_df = val_df.to_pandas()
            if test_df is not None and isinstance(test_df, pl.DataFrame):
                test_df = test_df.to_pandas()
            _polarsds_fell_back_to_sklearn = True

        if _polarsds_fell_back_to_sklearn:
            pass
        elif pipeline is not None:
            if verbose:
                logger.info("Applying Polars-ds pipeline...")

            # Capture train schema BEFORE the fit-time transform so we
            # can compare val/test schemas against it below. Without
            # this snapshot, pipeline.transform(val_df) was called
            # without any schema validation; missing/extra cols or dtype
            # mismatches silently propagated either to a downstream
            # sklearn shape-error or garbage output.
            _train_schema_snapshot = dict(train_df.schema)

            t0_transform = timer()
            # Transform all splits and ensure float32 dtypes
            train_df = pipeline.transform(train_df)
            if ensure_float32:
                train_df = ensure_dataframe_float32_convertability(train_df)

            if val_df is not None and len(val_df) > 0:
                _warn_on_schema_drift(_train_schema_snapshot, val_df, "val")
                val_df = pipeline.transform(val_df)
                if ensure_float32:
                    val_df = ensure_dataframe_float32_convertability(val_df)

            if test_df is not None and len(test_df) > 0:
                _warn_on_schema_drift(_train_schema_snapshot, test_df, "test")
                test_df = pipeline.transform(test_df)
                if ensure_float32:
                    test_df = ensure_dataframe_float32_convertability(test_df)

            if verbose:
                transform_elapsed = timer() - t0_transform
                logger.info("  Polars-ds transform done -- train: %s x%s, %.1fs", f"{train_df.shape[0]:_}", train_df.shape[1], transform_elapsed)
                logger.info("  train_df dtypes after pipeline: %s", Counter(train_df.dtypes))

        # Detect categorical features from schema (works whether pipeline succeeded or not)
        # This ensures cat_features is populated even if polars-ds is not available.
        # Prefer the ORIGINAL cat columns (captured before transform) -- after ordinal/onehot
        # encoding they're no longer Categorical/Utf8 in the transformed frame.
        # Skipped when we fell back to sklearn: ``train_df`` is now pandas and handled by the pandas branch below.
        if not _polarsds_fell_back_to_sklearn:
            post_cat = [c for c in get_polars_cat_columns(train_df) if c not in _exclude_from_encoding]
            cat_features = _orig_cat_features if _orig_cat_features else post_cat

    # Handle Polars DataFrames without polars-ds pipeline - just detect cat_features
    elif isinstance(train_df, pl.DataFrame) and not config.prefer_polarsds:
        # Detect categorical features from schema (no transformation, just detection)
        cat_features = [c for c in get_polars_cat_columns(train_df) if c not in _exclude_from_encoding]
        if verbose and cat_features:
            logger.info("Detected %s categorical features from Polars schema: %s", len(cat_features), cat_features)

    # Handle pandas DataFrames with sklearn-style pipeline (also the fallback target when polars-ds was unavailable).
    if isinstance(train_df, pd.DataFrame) and (not isinstance(train_df, pl.DataFrame)):
        # Identify categorical features (exclude text/embedding columns).
        # Embedding columns can sneak past the dtype filter when stored as
        # pandas object-of-ndarray (an embedding vector per row). They look
        # categorical via dtype.name=='object' but their cells are ndarrays
        # that hash() raises on, crashing category_encoders.OrdinalEncoder's
        # internal .unique() call. Detect and exclude via first-cell shape.
        def _looks_embedding(_series):
            """True iff an object-dtype column's cells look like embedding vectors (ndarray-like, non-str/bytes) rather than scalar categoricals, sampled from the first up-to-8 non-null values."""
            if _series.dtype != object:
                return False
            try:
                _first = next((v for v in _series.head(8) if v is not None), None)
            except Exception:
                return False
            if _first is None:
                return False
            return hasattr(_first, "shape") or (hasattr(_first, "__len__") and not isinstance(_first, (str, bytes)))

        # High-cardinality object/string columns are free-text, not categoricals.
        # The downstream auto-detect (_phase_auto_detect_feature_types) will
        # promote them to text_features, but it runs AFTER this pipeline-fit
        # phase. Without this guard, the elif branch below converts text_col
        # to pandas Categorical in-place; CB Pool then rejects it ("dtype
        # 'category' but not in cat_features list") because by the time the
        # CB Pool is built, text_col is correctly listed in text_features.
        # Threshold mirrors FeatureTypesConfig.cat_text_cardinality_threshold
        # default (300) and uses a SAMPLE-based unique count for cheap detection
        # on million-row frames. Surfaced by fuzz iter#49 (object+text_col+cb).
        _CAT_CARDINALITY_LIMIT = 300
        _SAMPLE_SIZE = 5000

        def _looks_text(_series):
            """True iff a string-like column looks like free text rather than a categorical: cardinality above ``_CAT_CARDINALITY_LIMIT`` measured on a cheap ``_SAMPLE_SIZE``-row sample (so million-row frames stay fast)."""
            dtype_name = _series.dtype.name
            # Include "str" for pandas-3.0 future.infer_string dtype.
            if dtype_name not in ("object", "string", "string[pyarrow]", "large_string[pyarrow]", "str"):
                return False
            n_rows = len(_series)
            if n_rows == 0:
                return False
            sample = _series.iloc[: min(_SAMPLE_SIZE, n_rows)]
            try:
                n_unique_sample = sample.nunique(dropna=True)
            except TypeError:
                return False
            return n_unique_sample > _CAT_CARDINALITY_LIMIT

        cat_features = [
            col for col in train_df.columns
            if train_df[col].dtype.name in PANDAS_CATEGORICAL_DTYPES
            and col not in _exclude_from_encoding
            and not _looks_embedding(train_df[col])
            and not _looks_text(train_df[col])
        ]

        # Apply categorical encoding if specified (for models that don't support categorical natively).
        # EXCEPTION: when we fell back here from an UNAVAILABLE polars-ds pipeline, mirror the polars-ds path's
        # contract and leave categoricals RAW (preserving ``cat_features`` for the downstream model strategy, e.g.
        # CatBoost's native handling). Ordinal-encoding them here would clear cat_features to [] and silently strip
        # CatBoost's native categorical support purely because polars-ds was missing -- a behaviour divergence the
        # polars-ds path never has (it returns the original cat columns untouched).
        if (
            cat_features
            and config.categorical_encoding in ["ordinal", "onehot"]
            and not config.skip_categorical_encoding
            and not _polarsds_fell_back_to_sklearn
        ):
            if verbose:
                logger.info("Applying %s encoding to %s categorical features: %s", config.categorical_encoding, len(cat_features), cat_features)

            t0_encode = timer()
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

            if verbose:
                encode_elapsed = timer() - t0_encode
                logger.info("  Encoding done -- train: %sx%s, %.1fs", f"{train_df.shape[0]:_}", train_df.shape[1], encode_elapsed)

            # After encoding, cat_features are no longer categorical (they're numeric)
            cat_features = []

        # Prepare categorical features for CatBoost (if not already encoded)
        elif cat_features:
            if verbose:
                logger.info("Preparing %s categorical features for CatBoost...", len(cat_features))

            # Joint train+val union for stable codes across splits.
            _safe_val = val_df if (val_df is not None and len(val_df) > 0) else None
            _safe_test = test_df if (test_df is not None and len(test_df) > 0) else None
            if train_df is not None and len(train_df) > 0:
                prepare_dfs_for_catboost_joint(
                    train_df=train_df, val_df=_safe_val, test_df=_safe_test,
                    cat_features=cat_features,
                )

    # Clean up empty validation/test sets
    if val_df is not None and len(val_df) == 0:
        val_df = None

    if test_df is not None and len(test_df) == 0:
        test_df = None

    # gc.collect on a 20-30GB heap with Arrow buffers can take a full minute
    # after a just-freed raw DataFrame. Previously this was the mystery "PHASE 3
    # black box" -- a minute passed between "Detected N categorical features"
    # and "Done. RAM usage:" with no log. Now wrapped so we can see it and
    # reason about disabling for polars-fastpath runs.
    t0_gc = timer()
    maybe_clean_ram_adaptive()
    gc_elapsed = timer() - t0_gc
    if verbose:
        if gc_elapsed > 1.0:
            logger.info("  maybe_clean_ram_adaptive took %.1fs (gc + arena trim)", gc_elapsed)
        log_ram_usage()

    return train_df, val_df, test_df, pipeline, cat_features
