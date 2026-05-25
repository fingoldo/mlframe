"""SimpleFeaturesAndTargetsExtractor carved out of ``mlframe.training.extractors``.

Re-imported at the parent module's bottom so historical
``from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor`` import
sites keep working.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

from mlframe.feature_engineering.basic import create_date_features

from .configs import TargetTypes
from .extractors import FeaturesAndTargetsExtractor
from ._extractors_dtype_helpers import (
    get_sample_weights_by_recency,
    intize_targets,
)

logger = logging.getLogger("mlframe.training.extractors")


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
        True so every run produces a uniform baseline -- without one, a single
        non-uniform schema (e.g. recency) cannot be attributed: is a metric win
        due to the weighting or would the same training plan win uniformly?
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
                # Wave 50 (2026-05-20): the prior fillna(0) silently labelled NaN rows as
                # the positive class when thresh_val<=0 (common for residual / mean-centered
                # targets) and identical-to-real-0 rows otherwise. Raise honestly instead;
                # callers must drop NaN targets explicitly upstream.
                if is_pandas:
                    if df[col].isna().any():
                        raise ValueError(
                            f"Classification target '{col}' contains NaN; drop or impute upstream "
                            "(silent fillna(0) was changed in wave-50 to surface this honestly)."
                        )
                    col_data = df[col]
                else:
                    if df[col].null_count() > 0:
                        raise ValueError(
                            f"Classification target '{col}' contains nulls; drop or impute upstream "
                            "(silent fill_null(0) was changed in wave-50 to surface this honestly)."
                        )
                    col_data = df[col]

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
                    # got wrapped as ``[(1,2,3)]`` then ``col_data == (1,2,3)`` raised on
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
