"""
Feature and target extraction utilities for mlframe.

Provides classes for preparing DataFrames before ML training, including:
- Target extraction and transformation
- Sample weight computation
- Feature engineering hooks
"""

import io
import logging
from typing import Union, Iterable, Optional, Dict, Any, Tuple

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

from .configs import TargetTypes
from .utils import log_ram_usage

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


def intize_targets(targets: Dict[str, Union[pd.Series, pl.Series, np.ndarray]]) -> None:
    """Convert target values to int8 numpy arrays in-place.

    Args:
        targets: Dictionary mapping target names to target arrays/series.

    Raises:
        TypeError: If target is not a supported type (pd.Series, pl.Series, np.ndarray).
    """
    for target_name, target in targets.copy().items():
        if isinstance(target, pl.Series):
            targets[target_name] = target.cast(pl.Int8).to_numpy()
        elif isinstance(target, pd.Series):
            targets[target_name] = target.astype(np.int8).values
        elif isinstance(target, np.ndarray):
            targets[target_name] = target.astype(np.int8)
        else:
            raise TypeError(f"Unsupported target type for '{target_name}': {type(target).__name__}")


def get_sample_weights_by_recency(
    date_series: pd.Series,
    min_weight: float = 1.0,
    weight_drop_per_year: float = 0.1,
) -> np.ndarray:
    """Compute sample weights based on recency.

    More recent samples get higher weights.

    Args:
        date_series: Series of datetime values.
        min_weight: Minimum weight for oldest samples.
        weight_drop_per_year: How much weight drops per year of age.

    Returns:
        Array of sample weights.
    """
    span = (date_series.max() - date_series.min()).days
    max_drop = np.log(span) * weight_drop_per_year

    sample_weight = (
        min_weight
        + max_drop
        - np.log((date_series.max() - date_series).dt.days) * weight_drop_per_year
    )

    return sample_weight


def showcase_features_and_targets(
    df: Union[pd.DataFrame, pl.DataFrame],
    target_by_type: Dict[str, Dict[str, Any]],
    max_hist_samples: int = 100_000,
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
                # Subsample if target is large to speed up histogram
                if len(target) > max_hist_samples:
                    if isinstance(target, (pl.Series, pd.Series)):
                        sample_idx = np.random.choice(
                            len(target), max_hist_samples, replace=False
                        )
                        sample = (
                            target.iloc[sample_idx].values
                            if isinstance(target, pd.Series)
                            else target[sample_idx].to_numpy()
                        )
                    else:
                        sample_idx = np.random.choice(
                            len(target), max_hist_samples, replace=False
                        )
                        sample = target[sample_idx]
                    # Add min and max to preserve full range (if not already in sample)
                    min_val, max_val = np.min(target), np.max(target)
                    extras = []
                    if min_val not in sample:
                        extras.append(min_val)
                    if max_val not in sample:
                        extras.append(max_val)
                    plot_data = np.concatenate([sample, extras]) if extras else sample
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
                if isinstance(target, (pl.Series, pd.Series)):
                    desc_data = target.value_counts(normalize=True)
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
            tail = tail.to_pandas()

        display(tail)


# -----------------------------------------------------------------------------
# Extractor Classes
# -----------------------------------------------------------------------------


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
    ):
        params = get_parent_func_args()
        store_params_in_object(obj=self, params=params)

        if self.columns_to_drop is None:
            self.columns_to_drop = set()

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
        """Display information about raw data."""
        info = get_dataframe_info(df)
        print("Raw data:")
        print(info)

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
                timestamps = timestamps.to_pandas()
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
            _, group_ids = np.unique(group_ids_raw_np, return_inverse=True)

        artifacts = self.prepare_artifacts(df)

        # Only show visualizations for verbose >= 2 (histograms are slow)
        if self.verbose:
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
    classification_thresholds : dict, optional
        Dict mapping column names to threshold values for binary classification.
        Example: {"score": 0.5} creates target "score_above_0.5".

    Example
    -------
    >>> extractor = SimpleFeaturesAndTargetsExtractor(
    ...     regression_targets=["price"],
    ...     classification_targets=["quality"],
    ...     classification_thresholds={"quality": 3},
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
        classification_thresholds: Optional[dict] = None,
    ):
        super().__init__(
            ts_field=ts_field,
            datetime_features=datetime_features,
            group_field=group_field,
            columns_to_drop=columns_to_drop,
            verbose=verbose,
        )

        self.regression_targets = regression_targets
        self.classification_targets = classification_targets
        self.classification_thresholds = classification_thresholds
        self.classification_exact_values = classification_exact_values

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
                if self.classification_thresholds and col in self.classification_thresholds:
                    thresh_val = self.classification_thresholds[col]
                    target_name = f"{col}_above_{thresh_val}"
                    if is_pandas:
                        targets[target_name] = (df[col] >= thresh_val).astype(np.int8)
                    elif is_polars:
                        targets[target_name] = (df[col] >= thresh_val).cast(pl.Int8)
                elif (
                    self.classification_exact_values
                    and col in self.classification_exact_values
                ):
                    exact_val = self.classification_exact_values[col]
                    target_name = f"{col}_eq_{exact_val}"
                    if is_pandas:
                        targets[target_name] = (df[col] == exact_val).astype(np.int8)
                    elif is_polars:
                        targets[target_name] = (df[col] == exact_val).cast(pl.Int8)
                else:
                    target_name = col
                    if is_pandas:
                        targets[target_name] = df[col].astype(np.int8)
                    elif is_polars:
                        targets[target_name] = df[col].cast(pl.Int8)
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

        return target_by_type

    def get_sample_weights(
        self, df: Union[pd.DataFrame, pl.DataFrame], timestamps: Optional[pd.Series] = None
    ) -> Dict[str, np.ndarray]:
        """Return recency-based sample weights if timestamps are available.

        Args:
            df: The DataFrame.
            timestamps: Timestamp series if available.

        Returns:
            Dict with 'recency' weights if timestamps provided, else empty dict.
        """
        if timestamps is None:
            return {}

        weights = {}
        # Add recency-based weights if timestamps available
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
