"""
Feature and target extraction utilities for mlframe.

Provides classes for preparing DataFrames before ML training, including:
- Target extraction and transformation
- Sample weight computation
- Feature engineering hooks
"""

from __future__ import annotations


import io
import logging
from typing import Union, Iterable, Optional, Dict, Any, Tuple, List, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from pyutilz.pythonlib import (
    store_params_in_object,
    get_parent_func_args,
    is_jupyter_notebook,
)
from pyutilz.polarslib import polars_df_info

from mlframe.feature_engineering.basic import create_date_features

from .configs import TargetTypes
from .utils import get_pandas_view_of_polars_df, log_ram_usage

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def get_dataframe_info(df: Union[pd.DataFrame, pl.DataFrame]) -> str:
    """Get a summary info string for a DataFrame.

    Args:
        df: Pandas or Polars DataFrame.

    Returns:
        Info string similar to pandas df.info() output.

    Raises:
        TypeError: If df is not a pandas or Polars DataFrame.
    """
    if isinstance(df, pd.DataFrame):
        buf = io.StringIO()
        df.info(buf=buf, verbose=False)
        return buf.getvalue()
    elif isinstance(df, pl.DataFrame):
        return polars_df_info(df)
    raise TypeError(f"Unsupported DataFrame type: {type(df).__name__}. Expected pandas or Polars DataFrame.")


def _smallest_safe_int_dtype(min_val: int, max_val: int) -> np.dtype:
    """Pick the smallest signed-int numpy dtype that can hold [min_val, max_val].

    Promotion ladder: int8 -> int16 -> int32 -> int64. Multiclass classification
    targets routinely have ``n_classes`` up to thousands (label-encoded
    categorical features, ID-based class assignments); the historical default of
    forcing every classification target to ``int8`` silently wrapped any label
    >127 on the pandas path (``np.array([200]).astype(np.int8) -> -56``) and
    raised hard on the polars path (asymmetric, the trap that surfaced this).
    Polars Series.cast(pl.Int8) raises ``InvalidOperationError``; pandas just
    wraps without warning -- pandas users got scrambled labels downstream.
    """
    if -128 <= min_val and max_val <= 127:
        return np.dtype(np.int8)
    if -32_768 <= min_val and max_val <= 32_767:
        return np.dtype(np.int16)
    if -2_147_483_648 <= min_val and max_val <= 2_147_483_647:
        return np.dtype(np.int32)
    return np.dtype(np.int64)


def _safe_int_cast_numpy(arr: np.ndarray, target_name: str) -> np.ndarray:
    """Cast a numpy array to the smallest signed-int dtype that preserves every value."""
    if arr.size == 0:
        return arr.astype(np.int8, copy=False)
    if np.issubdtype(arr.dtype, np.integer):
        _min, _max = int(arr.min()), int(arr.max())
    elif np.issubdtype(arr.dtype, np.floating):
        # Refuse fractional floats (e.g. accidentally-numeric target); only allow integer-valued floats.
        if not np.isfinite(arr).all() or not np.all(np.equal(np.mod(arr, 1), 0)):
            raise ValueError(
                f"target {target_name!r}: numeric target contains non-integer or non-finite values; "
                f"cannot safely cast to int. Drop NaN/inf rows or use a regression target type."
            )
        _min, _max = int(arr.min()), int(arr.max())
    elif np.issubdtype(arr.dtype, np.bool_):
        return arr.astype(np.int8, copy=False)
    else:
        # Object / string -- pandas/numpy will surface an error from astype itself; let it propagate
        # rather than introducing a separate error path here.
        return arr.astype(np.int8, copy=False)
    _dtype = _smallest_safe_int_dtype(_min, _max)
    if arr.dtype == _dtype:
        return arr
    return arr.astype(_dtype, copy=False)


def intize_targets(targets: Dict[str, Union[pd.Series, pl.Series, np.ndarray]]) -> None:
    """Convert target values to the smallest signed-int numpy dtype that preserves every value.

    Multiclass labels with cardinality >127 used to silently wrap on the pandas
    path (``astype(np.int8)`` is modulo arithmetic for over-range ints) while
    failing loudly on the polars path. Now promotes int8 -> int16 -> int32 -> int64
    based on the actual value range. Multiclass datasets with thousands of
    classes (label-encoded categorical IDs, hash-encoded targets) round-trip
    correctly under this fix.

    Args:
        targets: Dictionary mapping target names to target arrays/series.

    Raises:
        TypeError: If target is not a supported type (pd.Series, pl.Series, np.ndarray).
        ValueError: If a numeric target contains fractional / non-finite values
            (silent truncation hazard).
    """
    for target_name, target in targets.copy().items():
        if isinstance(target, np.ndarray):
            targets[target_name] = _safe_int_cast_numpy(target, target_name)
        elif isinstance(target, pl.Series):
            # to_numpy first (zero-copy on contiguous numeric), then route through the
            # numpy-side range-aware cast so polars + pandas paths share one promotion table.
            targets[target_name] = _safe_int_cast_numpy(target.to_numpy(), target_name)
        elif isinstance(target, pd.Series):
            targets[target_name] = _safe_int_cast_numpy(target.to_numpy(), target_name)
        else:
            raise TypeError(f"Unsupported target type for '{target_name}': {type(target).__name__}")


def get_sample_weights_by_recency(
    date_series: pd.Series,
    min_weight: float = 1.0,
    weight_drop_per_year: float = 0.1,
) -> np.ndarray:
    """Compute sample weights based on recency.

    More recent samples get higher weights. The formula is log-linear
    in days-from-most-recent so that very old samples don't vanish
    entirely while the newest samples get the highest weight.

    Bug fix 2026-04-19: the previous implementation applied
    ``np.log((max - date).days)`` directly. For the single most-recent
    sample (where ``max - date == 0 days``), ``np.log(0) = -inf``, so
    the weight evaluated to ``+inf``. Training-time weighted loss was
    then dominated by that single row (CatBoost/sklearn treat +inf
    weights by clamping or NaN-ing the sample, producing silent fit
    bias that was invisible in the training curve). Also: if all dates
    are identical (``span == 0``), ``np.log(0) → -inf`` produces an
    all-NaN weight array.

    Now: days-from-max is clipped to ``>= 1`` before the log so the
    newest sample gets a *large finite* weight (floor at
    ``min_weight + max_drop``), and a zero-span series returns uniform
    ``min_weight`` for every row (log-span itself is clipped too).

    Args:
        date_series: Series of datetime values.
        min_weight: Minimum weight for oldest samples.
        weight_drop_per_year: How much weight drops per year of age.

    Returns:
        Array of sample weights (all finite, no NaN / +inf).
    """
    # Use total_seconds() / 86400 instead of .days: ``.days`` floors to integer
    # days and returns 0 for intraday-only datasets (e.g. a single trading day
    # of tick data), which then triggers the uniform-weight branch even though
    # the data has meaningful sub-day age structure.
    #
    # Polymorphic input: callers historically pass a pandas datetime Series
    # (where max - min returns a Timedelta with .total_seconds()) but the
    # FTE also accepts a numeric ts column (int64 / float64 epoch-seconds
    # or any monotone numeric proxy). Numeric (max - min) returns a scalar
    # that has no .total_seconds() method and raises
    # ``AttributeError: 'int' object has no attribute 'total_seconds'``.
    # Detect the numeric path via dtype kind ('i', 'u', 'f') and interpret
    # the raw difference as already-seconds; preserve the datetime path
    # via the original .total_seconds() call.
    _dtype_kind = getattr(getattr(date_series, "dtype", None), "kind", None)
    _is_numeric_ts = _dtype_kind in ("i", "u", "f")
    if _is_numeric_ts:
        # Numeric ts: treat values as seconds-since-some-epoch. max-min is
        # already in seconds; just divide by 86400 to get days.
        span_seconds = float(date_series.max() - date_series.min())
        span_days = span_seconds / 86400.0
    else:
        span_days = (date_series.max() - date_series.min()).total_seconds() / 86400.0
    # Zero-span guard: all dates equal -> uniform weighting. No log needed.
    if span_days <= 0:
        return np.full(len(date_series), float(min_weight))

    # Sub-day resolution preserved via total_seconds(). Floor at one
    # second (~ 1/86400 day) so log never hits zero -- the previous
    # one-day floor erased intraday gradient by clamping every row to
    # log(1)=0.
    if _is_numeric_ts:
        # Numeric path: max-row is a scalar; subtraction is element-wise
        # numeric and the result is already seconds-since-row.
        _delta_secs = np.asarray(date_series.max() - date_series, dtype=np.float64)
    else:
        _delta_secs = (date_series.max() - date_series).dt.total_seconds().to_numpy()
    _min_age_days = 1.0 / 86400.0  # one-second floor
    days_from_max = np.maximum(_delta_secs / 86400.0, _min_age_days)
    # log(span_days) for span<1 day is negative -> max_drop negative.
    # Use log(span_in_seconds) baseline so the gradient stays positive
    # for sub-day spans too.
    max_drop = (np.log(span_days) - np.log(_min_age_days)) * weight_drop_per_year

    sample_weight = (
        min_weight
        + max_drop
        - (np.log(days_from_max) - np.log(_min_age_days)) * weight_drop_per_year
    )

    return sample_weight


def showcase_features_and_targets(
    df: Union[pd.DataFrame, pl.DataFrame],
    target_by_type: Dict[str, Dict[str, Any]],
    max_hist_samples: int = 100_000,
    random_seed: int = 42,
) -> None:
    """Show distribution of features and targets.

    Args:
        df: DataFrame with features.
        target_by_type: Dictionary of targets by type (e.g., {TargetTypes.REGRESSION: {"target1": array}}).
        max_hist_samples: Maximum samples to use for histogram (performance threshold).
    """
    print(get_dataframe_info(df))

    head = df.head(5)
    if isinstance(df, pl.DataFrame):
        # Audit D P2-4 (2026-05-18): bare ``.to_pandas()`` is the slow consolidation copy path,
        # but ``head(5)`` is 5 rows -- the copy cost is dominated by per-cell overhead, not by
        # bulk-buffer consolidation. Display-only, cold-path. NEEDED for downstream pandas-only
        # ``.style.set_caption`` / Jupyter display rich rendering.
        head = head.to_pandas()

    non_floats = head.select_dtypes(exclude=np.float32)

    caption = "Non-float32 dtypes"

    logger.info(f"{caption}: {non_floats.dtypes.to_dict()}")

    in_jupyter = is_jupyter_notebook()

    if in_jupyter:
        from IPython.display import display

        display(non_floats.style.set_caption("Non-float32 dtypes"))

    for target_type, targets in target_by_type.items():
        for target_name, target in targets.items():
            line = f"{target_type} {target_name}"
            if in_jupyter:
                from IPython.display import display

                display(line)
            else:
                print(line)
            if target_type == TargetTypes.REGRESSION:
                # Subsample if target is large to speed up histogram. Use a
                # local seeded Generator instead of the global numpy RNG so
                # histograms shown to the user are reproducible across runs
                # and don't depend on whatever else mutated np.random state.
                _hist_rng = np.random.default_rng(random_seed)
                if len(target) > max_hist_samples:
                    if isinstance(target, (pl.Series, pd.Series)):
                        sample_idx = _hist_rng.choice(
                            len(target), max_hist_samples, replace=False
                        )
                        # Audit D L-3 (2026-05-18): polars Series ``target[sample_idx]``
                        # with a numpy integer array works (polars treats it as
                        # ``.gather(sample_idx)``); pandas Series uses ``.iloc`` for positional
                        # indexing to avoid the FutureWarning on label-vs-position dispatch.
                        sample = (
                            target.iloc[sample_idx].values
                            if isinstance(target, pd.Series)
                            else target[sample_idx].to_numpy()
                        )
                    else:
                        sample_idx = _hist_rng.choice(
                            len(target), max_hist_samples, replace=False
                        )
                        sample = target[sample_idx]
                    # Add min and max to preserve full range (if not already in sample)
                    if isinstance(target, pl.Series):
                        min_val, max_val = target.min(), target.max()
                    elif isinstance(target, pd.Series):
                        min_val, max_val = target.min(), target.max()
                    else:  # np.ndarray
                        min_val, max_val = np.min(target), np.max(target)
                    extras = []
                    if min_val not in sample:
                        extras.append(min_val)
                    if max_val not in sample:
                        extras.append(max_val)
                    plot_data = np.concatenate([sample, extras]) if extras else sample
                else:
                    # Convert to numpy array if needed
                    if isinstance(target, pl.Series):
                        plot_data = target.to_numpy()
                    elif isinstance(target, pd.Series):
                        plot_data = target.values
                    else:
                        plot_data = target
                plt.hist(plot_data, bins=30, color="skyblue", edgecolor="black")

                # Add titles and labels
                plt.title(f"{target_name} Histogram")
                plt.xlabel("Value")
                plt.ylabel("Frequency")

                # Show the plot
                plt.show()

                if isinstance(target, (pl.Series, pd.Series)):
                    desc_data = target.describe()
                elif isinstance(target, np.ndarray):
                    desc_data = pl.Series(target).describe()

                if in_jupyter:
                    from IPython.display import display

                    display(desc_data)
                else:
                    print(desc_data)

            elif target_type == TargetTypes.BINARY_CLASSIFICATION:
                if isinstance(target,  pd.Series):
                    desc_data = target.value_counts(normalize=True)
                elif isinstance(target, pl.Series):
                    desc_data = target.value_counts(normalize=True, sort=True)
                elif isinstance(target, np.ndarray):
                    desc_data = pl.Series(target).value_counts(normalize=True, sort=True)

                if in_jupyter:
                    from IPython.display import display

                    display(desc_data)
                else:
                    print(desc_data)

    if in_jupyter:
        from IPython.display import display

        display(head)

        tail = df.tail(5)
        if isinstance(df, pl.DataFrame):
            # Audit D P2-4 (2026-05-18): see head() comment above. Display-only, cold-path.
            tail = tail.to_pandas()

        display(tail)


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

    def add_features(
        self, df: Union[pd.DataFrame, pl.DataFrame]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
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

    def show_processed_data(
        self, df: Union[pd.DataFrame, pl.DataFrame], target_by_type: dict
    ) -> None:
        """Display information about processed data and targets."""
        print("Processed data:")
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

    def get_sequences(
        self, df: Union[pd.DataFrame, pl.DataFrame]
    ) -> Optional[List[np.ndarray]]:
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

        return extract_sequences(
            df,
            columns=self.sequence_columns,
            group_column=self.sequence_group_column,
        )

    def get_sample_weights(
        self, df: Union[pd.DataFrame, pl.DataFrame], timestamps: Optional[pd.Series] = None
    ) -> Dict[str, np.ndarray]:
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

    def transform(
        self, df: Union[pd.DataFrame, pl.DataFrame]
    ) -> Tuple[
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
            group_ids_raw_np = (
                group_ids_raw.to_numpy()
                if isinstance(group_ids_raw, pl.Series)
                else group_ids_raw.values
            )
            # pd.factorize handles None/NaN and mixed types; -1 codes map to a "__null__" sentinel
            codes, _ = pd.factorize(group_ids_raw_np)
            group_ids = codes

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


class SimpleFeaturesAndTargetsExtractor(FeaturesAndTargetsExtractor):
    """Simple extractor for common regression and classification targets.

    Supports:
    - Regression targets (columns used directly)
    - Classification targets with optional thresholds or exact values
    - Recency-based sample weights

    Parameters
    ----------
    ts_field : str, optional
        Name of timestamp field for recency-based sample weights.
    datetime_features : dict, optional
        Datetime feature extraction configuration.
    group_field : str, optional
        Name of group/entity identifier field for grouped cross-validation.
    columns_to_drop : set, optional
        Columns to exclude from features.
    allowed_targets : Iterable, optional
        If specified, only allow these target names.
    verbose : int, default=0
        Verbosity level (0=silent, 1=info, 2=debug with plots).
    regression_targets : Iterable, optional
        Column names to use as regression targets.
    classification_targets : Iterable, optional
        Column names to use as classification targets.
    classification_exact_values : dict, optional
        Dict mapping column names to exact values for binary classification.
        Example: {"status": 1} creates target "status_eq_1".
    classification_lower_thresholds : dict, optional
        Dict mapping column names to lower threshold values for binary classification.
        Example: {"score": 0.5} creates target "score_above_0.5".
    classification_upper_thresholds : dict, optional
        Dict mapping column names to upper threshold values for binary classification.
        Example: {"score": 0.8} creates target "score_below_0.8".
    use_uniform_weighting : bool, default=True
        If True, include uniform weighting (None) in sample weights dict. Default is
        True so every run produces a uniform baseline — without one, a single
        non-uniform schema (e.g. recency) cannot be attributed: is a metric win
        due to the weighting or would the same training plan win uniformly?
        (Observed 2026-04-23 prod log: a recency-only suite reported AUC=0.999
        VAL / 0.71 TEST without any uniform comparator, making the attribution
        impossible.)
    use_recency_weighting : bool, default=True
        If True and timestamps are available, include recency-based sample weights.
        Combined with ``use_uniform_weighting=True``, time-indexed data gets
        ``{uniform, recency}`` and non-temporal data gets ``{uniform}`` only
        (recency silently skips when ``timestamps is None``).

    Example
    -------
    >>> extractor = SimpleFeaturesAndTargetsExtractor(
    ...     regression_targets=["price"],
    ...     classification_targets=["quality"],
    ...     classification_lower_thresholds={"quality": 3},
    ...     classification_upper_thresholds={"quality": 8},
    ...     ts_field="date",
    ... )
    >>> df, targets, *rest = extractor.transform(df)
    """

    def __init__(
        self,
        ts_field: Optional[str] = None,
        datetime_features: Optional[dict] = None,
        group_field: Optional[str] = None,
        columns_to_drop: Optional[set] = None,
        allowed_targets: Optional[Iterable] = None,
        verbose: int = 0,
        #
        regression_targets: Optional[Iterable] = None,
        classification_targets: Optional[Iterable] = None,
        classification_exact_values: Optional[dict] = None,
        classification_lower_thresholds: Optional[dict] = None,
        classification_upper_thresholds: Optional[dict] = None,
        classification_thresholds: Optional[dict] = None,  # alias for classification_lower_thresholds
        # Weighting options
        use_uniform_weighting: bool = True,
        use_recency_weighting: bool = True,
        # Sequence extraction (for recurrent models)
        sequence_columns: Optional[Tuple[str, ...]] = None,
        sequence_group_column: Optional[str] = None,
    ):
        super().__init__(
            ts_field=ts_field,
            datetime_features=datetime_features,
            group_field=group_field,
            columns_to_drop=columns_to_drop,
            # ``allowed_targets`` was documented as a filter at lines 648-649 but pre-fix the
            # subclass __init__ silently swallowed the kwarg without forwarding to the base
            # class (which DOES accept + store it via store_params_in_object). A caller
            # passing allowed_targets=["a"] alongside classification_targets=["a","b","c"]
            # expecting filtering got all three trained instead.
            allowed_targets=allowed_targets,
            verbose=verbose,
            sequence_columns=sequence_columns,
            sequence_group_column=sequence_group_column,
        )

        self.regression_targets = regression_targets
        self.classification_targets = classification_targets
        # classification_thresholds is an alias for classification_lower_thresholds
        if classification_thresholds is not None and classification_lower_thresholds is None:
            classification_lower_thresholds = classification_thresholds
        self.classification_lower_thresholds = classification_lower_thresholds
        self.classification_upper_thresholds = classification_upper_thresholds
        self.classification_exact_values = classification_exact_values
        self.use_uniform_weighting = use_uniform_weighting
        self.use_recency_weighting = use_recency_weighting

    def add_features(self, df: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]:
        if self.ts_field and self.datetime_features:
            if self.verbose:
                logger.info("create_date_features %s over column %s...", self.datetime_features, self.ts_field)
            _pre_cols = set(df.columns)
            df = create_date_features(df, cols=[self.ts_field], delete_original_cols=False, methods=self.datetime_features)
            _derived = [c for c in df.columns if c not in _pre_cols]
            # Record so the suite (``_phase_fit_pipeline``) can SKIP re-decomposing ``ts_field`` -- the second pass would emit duplicate / overwriting cols.
            self.ftextractor_emitted_columns[self.ts_field] = _derived
        return df

    def build_targets(self, df: Union[pd.DataFrame, pl.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Build regression and classification targets from DataFrame columns.

        Args:
            df: Input DataFrame.

        Returns:
            Dictionary mapping TargetTypes to dicts of target arrays.

        Raises:
            KeyError: If a required target column is not found in the DataFrame.
        """
        target_by_type = {}
        is_pandas = isinstance(df, pd.DataFrame)
        is_polars = isinstance(df, pl.DataFrame)
        df_columns = set(df.columns)

        if self.classification_targets:
            targets = {}
            for col in self.classification_targets:
                if col not in df_columns:
                    raise KeyError(f"Classification target column '{col}' not found in DataFrame. Available: {list(df.columns)[:10]}...")
                # Impute NaNs with 0 for classification targets
                col_data = df[col].fillna(0) if is_pandas else df[col].fill_null(0)

                # Process lower thresholds
                if self.classification_lower_thresholds and col in self.classification_lower_thresholds:
                    thresh_val = self.classification_lower_thresholds[col]
                    target_name = f"{col}_above_{thresh_val}"
                    if is_pandas:
                        targets[target_name] = (col_data >= thresh_val).astype(np.int8)
                    elif is_polars:
                        targets[target_name] = (col_data >= thresh_val).cast(pl.Int8)

                # Process upper thresholds
                if self.classification_upper_thresholds and col in self.classification_upper_thresholds:
                    thresh_val = self.classification_upper_thresholds[col]
                    target_name = f"{col}_below_{thresh_val}"
                    if is_pandas:
                        targets[target_name] = (col_data <= thresh_val).astype(np.int8)
                    elif is_polars:
                        targets[target_name] = (col_data <= thresh_val).cast(pl.Int8)

                # Process exact values
                if (
                    self.classification_exact_values
                    and col in self.classification_exact_values
                ):
                    exact_val = self.classification_exact_values[col]
                    # Wave 29 P1 fix (2026-05-20): pre-fix accepted only
                    # ``list``; ``classification_exact_values={"col": (1,2,3)}``
                    # (natural Python idiom for fixed sets) got wrapped as
                    # ``[(1,2,3)]`` then ``col_data == (1,2,3)`` raised on
                    # both pandas and polars. Accept any iterable container.
                    if isinstance(exact_val, (list, tuple, set, frozenset)):
                        exact_vals = list(exact_val)
                    else:
                        exact_vals = [exact_val]

                    for val in exact_vals:
                        target_name = f"{col}_eq_{val}"
                        if is_pandas:
                            targets[target_name] = (col_data == val).astype(np.int8)
                        elif is_polars:
                            targets[target_name] = (col_data == val).cast(pl.Int8)

                # Default: use column as-is. Don't pre-cast to int8 here -- intize_targets()
                # below promotes int8/16/32/64 based on actual value range, so multiclass labels
                # with cardinality >127 don't wrap silently (pandas) or raise (polars).
                if (
                    col not in (self.classification_lower_thresholds or {})
                    and col not in (self.classification_upper_thresholds or {})
                    and col not in (self.classification_exact_values or {})
                ):
                    target_name = col
                    targets[target_name] = col_data

                self.columns_to_drop.add(col)

            intize_targets(targets)
            target_by_type[TargetTypes.BINARY_CLASSIFICATION] = targets

        if self.regression_targets:
            targets = {}
            for col in self.regression_targets:
                if col not in df_columns:
                    raise KeyError(f"Regression target column '{col}' not found in DataFrame. Available: {list(df.columns)[:10]}...")
                targets[col] = df[col]
                self.columns_to_drop.add(col)
            target_by_type[TargetTypes.REGRESSION] = targets

        # Apply allowed_targets filter (docstring promised feature, never implemented pre-fix).
        # When set, only keep target NAMES present in the set across every TargetTypes bucket.
        # Names that don't match any built target are reported at WARNING so the caller catches
        # typos in their allowlist rather than silently training an empty target_by_type.
        _allowed = getattr(self, "allowed_targets", None)
        if _allowed is not None:
            _allowed_set = set(_allowed) if not isinstance(_allowed, set) else _allowed
            _filtered: dict = {}
            _kept: set[str] = set()
            for _tt, _named in target_by_type.items():
                if isinstance(_named, dict):
                    _kept_for_tt = {k: v for k, v in _named.items() if k in _allowed_set}
                    if _kept_for_tt:
                        _filtered[_tt] = _kept_for_tt
                        _kept.update(_kept_for_tt.keys())
            _missing = _allowed_set - _kept
            if _missing and self.verbose:
                logger.warning(
                    "allowed_targets filter: %d name(s) not found in built targets: %s. "
                    "Built names: %s.",
                    len(_missing), sorted(_missing), sorted(_kept),
                )
            target_by_type = _filtered

        return target_by_type

    def get_sample_weights(
        self, df: Union[pd.DataFrame, pl.DataFrame], timestamps: Optional[pd.Series] = None
    ) -> Dict[str, np.ndarray]:
        """Return sample weights based on configured weighting options.

        Args:
            df: The DataFrame.
            timestamps: Timestamp series if available.

        Returns:
            Dict with enabled weight schemes. Empty dict if no weighting enabled.
        """
        weights = {}

        if self.use_uniform_weighting:
            weights["uniform"] = None

        if self.use_recency_weighting and timestamps is not None:
            weights["recency"] = get_sample_weights_by_recency(timestamps)

        return weights


__all__ = [
    # Helper functions
    "get_dataframe_info",
    "intize_targets",
    "get_sample_weights_by_recency",
    "showcase_features_and_targets",
    # Classes
    "FeaturesAndTargetsExtractor",
    "SimpleFeaturesAndTargetsExtractor",
]
