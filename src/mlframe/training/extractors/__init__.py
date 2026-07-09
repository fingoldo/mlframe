"""
Feature and target extraction utilities for mlframe.

Provides classes for preparing DataFrames before ML training, including:
- Target extraction and transformation
- Sample weight computation
- Feature engineering hooks
"""

from __future__ import annotations


import logging
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np
import pandas as pd
import polars as pl

from pyutilz.pythonlib import get_parent_func_args, store_params_in_object

from ._extractors_dtype_helpers import (  # -- re-export
    _safe_int_cast_numpy,
    _smallest_safe_int_dtype,
    get_dataframe_info,
    get_sample_weights_by_recency,
    intize_targets,
)
from ._extractors_showcase import showcase_features_and_targets  # -- re-export
from ..utils import get_pandas_view_of_polars_df, log_ram_usage

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Extractor Classes
# -----------------------------------------------------------------------------


@runtime_checkable
class FeaturesAndTargetsExtractorProtocol(Protocol):
    """Typed contract for any FTE the training suite consumes.

    The training suite reads ``ftextractor_emitted_columns`` (a mapping
    ``source_col -> [derived_col, ...]``) so the auto-detect / pipeline
    phases can avoid re-emitting columns the FTE already added (eg datetime
    decomposition). This Protocol makes the contract explicit so external
    extractors are checked at runtime via ``isinstance(obj, ...)`` instead
    of attribute sniffing.

    Concrete subclasses (``FeaturesAndTargetsExtractor`` below,
    ``SimpleFeaturesAndTargetsExtractor`` further down) already satisfy
    this Protocol; new third-party extractors only need to expose the
    same attribute name to be accepted.
    """

    ftextractor_emitted_columns: Dict[str, List[str]]

    def add_features(self, df: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]: ...

    def build_targets(self, df: Union[pd.DataFrame, pl.DataFrame]) -> dict: ...


class FeaturesAndTargetsExtractor:
    """Base class that prepares a DataFrame before ML training.

    This class provides hooks for:
    - Adding features via add_features()
    - Building targets via build_targets()
    - Computing sample weights via get_sample_weights()
    - Preparing artifacts via prepare_artifacts()

    Subclass and override these methods for custom behavior.
    """

    # __init__ params are set on self dynamically via store_params_in_object(); declared here so
    # mypy can type-check reads/writes of this one (checked/defaulted right after in __init__, and
    # mutated by subclasses' build_targets()).
    columns_to_drop: Optional[set]
    ts_field: Optional[str]
    datetime_features: Optional[dict]
    group_field: Optional[str]
    allowed_targets: Optional[Iterable]
    verbose: int
    sequence_columns: Optional[Tuple[str, ...]]
    sequence_group_column: Optional[str]

    def __init__(
        self,
        ts_field: Optional[str] = None,
        datetime_features: Optional[dict] = None,
        group_field: Optional[str] = None,
        columns_to_drop: Optional[set] = None,
        allowed_targets: Optional[Iterable] = None,
        verbose: int = 0,
        # Sequence extraction parameters (for recurrent models)
        sequence_columns: Optional[Tuple[str, ...]] = None,
        sequence_group_column: Optional[str] = None,
    ):
        params = get_parent_func_args()
        store_params_in_object(obj=self, params=params)

        if self.columns_to_drop is None:
            self.columns_to_drop = set()

        # Record source-column -> [derived column names] for any datetime decomposition the FTE does in ``add_features``. The suite reads this in ``_phase_fit_pipeline`` to skip re-decomposing columns FTE already handled (otherwise the suite's second ``create_date_features`` call would emit duplicate ``{ts}_year`` / ``{ts}_month`` cols on top of FTE's). Default-empty so base FTEs (no add_features override) don't trigger the skip.
        self.ftextractor_emitted_columns: Dict[str, List[str]] = {}

    def add_features(self, df: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]:
        """Add engineered features to the DataFrame.

        Override in subclasses to add custom features.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with added features.
        """
        return df

    def build_targets(self, df: Union[pd.DataFrame, pl.DataFrame]) -> dict:
        """Build target variables from the DataFrame.

        Override in subclasses to extract targets.

        Args:
            df: Input DataFrame.

        Returns:
            Dictionary mapping TargetTypes to dicts of target arrays.
        """
        return {}

    def show_raw_data(self, df: Union[pd.DataFrame, pl.DataFrame]) -> None:
        """Display information about raw data.

        Routed through the module logger rather than bare ``print`` so the
        output interleaves correctly with the rest of the training log
        (previously appeared out-of-order in Jupyter because stdout flushed
        separately from the logger stream).
        """
        info = get_dataframe_info(df)
        logger.info("Raw data:\n%s", info)

    def show_processed_data(self, df: Union[pd.DataFrame, pl.DataFrame], target_by_type: dict) -> None:
        """Display information about processed data and targets."""
        logger.info("Processed data:")
        showcase_features_and_targets(df, target_by_type)
        log_ram_usage()

    def prepare_artifacts(self, df: Union[pd.DataFrame, pl.DataFrame]) -> dict:
        """Prepare any artifacts needed for training.

        Override in subclasses for custom artifacts.

        Args:
            df: Input DataFrame.

        Returns:
            Dictionary of artifacts.
        """
        return {}

    def get_sequences(self, df: Union[pd.DataFrame, pl.DataFrame]) -> Optional[List[np.ndarray]]:
        """Extract sequences from DataFrame for recurrent models.

        Uses sequence_columns and sequence_group_column parameters configured
        during initialization.

        Args:
            df: Input DataFrame containing sequence data.

        Returns:
            List of (seq_len, n_columns) arrays, one per group/entity,
            or None if sequence_columns is not configured.

        Example:
            >>> extractor = FeaturesAndTargetsExtractor(
            ...     sequence_columns=("flux", "flux_err"),
            ...     sequence_group_column="object_id",
            ... )
            >>> sequences = extractor.get_sequences(df)
            >>> # sequences[0].shape = (seq_len_0, 2)
        """
        if self.sequence_columns is None:
            return None

        from mlframe.training.neural import extract_sequences

        # extract_sequences expects one row per entity (list-column format), so grouping is
        # already implicit in the row structure; sequence_group_column has no effect here and
        # extract_sequences has no such parameter -- passing it crashed on every call.
        return extract_sequences(
            df,
            columns=self.sequence_columns,
        )

    def get_sample_weights(self, df: Union[pd.DataFrame, pl.DataFrame], timestamps: Optional[pd.Series] = None) -> Dict[str, np.ndarray]:
        """Compute sample weights.

        Override in subclasses for custom weight schemes.

        Args:
            df: The DataFrame.
            timestamps: Timestamp series if available (from ts_field).

        Returns:
            Dict mapping weight schema names to weight arrays.
            Empty dict means uniform weights only.
        """
        return {}

    def transform(self, df: Union[pd.DataFrame, pl.DataFrame]) -> Tuple[
        Union[pd.DataFrame, pl.DataFrame],  # df
        Dict[str, Dict[str, Any]],  # target_by_type
        Optional[Union[pd.Series, pl.Series]],  # group_ids_raw
        Optional[np.ndarray],  # group_ids
        Optional[pd.Series],  # timestamps
        Dict[str, Any],  # artifacts
        set,  # columns_to_drop
        Dict[str, np.ndarray],  # sample_weights
    ]:
        """Transform the DataFrame into features and targets.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of:
            - df: Processed DataFrame
            - target_by_type: Dict of targets by type
            - group_ids_raw: Raw group IDs
            - group_ids: Encoded group IDs
            - timestamps: Timestamp series
            - artifacts: Dict of artifacts
            - columns_to_drop: Set of columns to drop
            - sample_weights: Dict of sample weight arrays
        """
        # Reset per-call state so the same FTE instance can be re-used across multiple
        # train_mlframe_models_suite calls (CV loop, A/B sweeps, fold-by-fold retraining)
        # without leaking N's target columns into N+1's drop list and N's emitted
        # date-decomposition columns into N+1's metadata. Pre-2026-05-20 build_targets
        # called ``self.columns_to_drop.add(col)`` per target column and
        # add_features did ``self.ftextractor_emitted_columns[self.ts_field] = ...``,
        # both growing monotonically per FTE-instance lifetime and contaminating the
        # downstream suite metadata on every subsequent call.
        #
        # Snapshot of the user-supplied initial columns_to_drop is captured on first
        # transform() entry (pre-fix this set's identity was the user's set; we
        # preserved that bug-compatibly but now also reset to its initial contents
        # at the top of every transform).
        if not hasattr(self, "_initial_columns_to_drop_snapshot"):
            self._initial_columns_to_drop_snapshot = set(self.columns_to_drop or ())
        self.columns_to_drop = set(self._initial_columns_to_drop_snapshot)
        self.ftextractor_emitted_columns = {}
        self.show_raw_data(df)

        df = self.add_features(df)
        if self.verbose:
            logger.info("After add_features")
            log_ram_usage()

        if self.verbose:
            logger.info("build_targets...")
        target_by_type = self.build_targets(df)
        if self.verbose:
            log_ram_usage()

        if self.ts_field:
            timestamps = df[self.ts_field]
            if isinstance(timestamps, pl.Series):
                # Bridge the pl.Series via to_frame -> to_arrow -> to_pandas to avoid the bare-to_pandas
                # consolidation pass that allocates a fresh datetime64 buffer instead of viewing the Arrow
                # buffer in-place. On 10M-row date columns this saves ~3s + dups RAM.
                _ts_pdf = get_pandas_view_of_polars_df(timestamps.to_frame(self.ts_field))
                timestamps = _ts_pdf[self.ts_field]
        else:
            timestamps = None

        group_ids_raw = None
        group_ids = None
        if self.group_field is not None and self.group_field in df.columns:
            group_ids_raw = df[self.group_field]
            group_ids_raw_np = group_ids_raw.to_numpy() if isinstance(group_ids_raw, pl.Series) else group_ids_raw.values
            # pd.factorize handles None/NaN and mixed types; -1 codes map to a "__null__" sentinel
            codes, _ = pd.factorize(group_ids_raw_np)
            group_ids = codes
            # The group column carries the CV/grouping id (now captured in group_ids), NOT a predictive feature: a
            # high-cardinality identifier (well_id, user_id, ...) cannot generalise to UNSEEN groups, and if it is a
            # string it crashes the numeric FE (e.g. the DCD PCA "could not convert string to float"). Exclude it from
            # features -- same pattern as targets above -- so it neither leaks into selection nor bloats the codes matrix.
            self.columns_to_drop.add(self.group_field)

        artifacts = self.prepare_artifacts(df)

        # Only show visualizations for verbose >= 2 (df.info() dumps are slow on wide frames)
        if self.verbose >= 2:
            self.show_processed_data(df, target_by_type)
            if self.columns_to_drop or self.datetime_features:
                pass  # clean_ram()

        # Build sample_weights dict - subclasses can override get_sample_weights()
        sample_weights = self.get_sample_weights(df, timestamps)

        return (
            df,
            target_by_type,
            group_ids_raw,
            group_ids,
            timestamps,
            artifacts,
            self.columns_to_drop,
            sample_weights,
        )


from ._extractors_simple import SimpleFeaturesAndTargetsExtractor

__all__ = [
    # Helper functions
    "get_dataframe_info",
    "intize_targets",
    "get_sample_weights_by_recency",
    "showcase_features_and_targets",
    # Classes
    "FeaturesAndTargetsExtractor",
    "SimpleFeaturesAndTargetsExtractor",
    "FeaturesAndTargetsExtractorProtocol",
]
