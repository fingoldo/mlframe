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

from mlframe.feature_engineering.basic import CYCLICAL_ENCODING_VERSION, LEGACY_CYCLICAL_ENCODING_VERSION, create_date_features

from ..configs import TargetTypes
from . import FeaturesAndTargetsExtractor
from ._extractors_dtype_helpers import (
    get_sample_weights_by_recency,
    intize_targets,
)

logger = logging.getLogger("mlframe.training.extractors")


def _missing_label_mask(col_data: "Union[pd.Series, pl.Series]") -> Optional[np.ndarray]:
    """Rows of a classification source column without a label, or ``None`` when every row has one.

    Nulls plus float NaN: polars ``null_count()`` misses NaN, and ``NaN >= t`` is True in polars, so a NaN row used to
    become the positive class under a lower threshold.
    """
    if isinstance(col_data, pd.Series):
        missing = col_data.isna().to_numpy()
    else:
        missing = col_data.is_null().to_numpy()
        if col_data.dtype.is_float():
            missing |= col_data.is_nan().fill_null(False).to_numpy()
    return missing if missing.any() else None


def _binary_target(comparison: "Union[pd.Series, pl.Series]", missing: Optional[np.ndarray]) -> Any:
    """A 0/1 target from a boolean comparison: int8 when every row is labelled, else float32 with NaN on missing rows."""
    if missing is None:
        return comparison.astype(np.int8) if isinstance(comparison, pd.Series) else comparison.cast(pl.Int8)
    values = comparison.to_numpy() if isinstance(comparison, pd.Series) else comparison.fill_null(False).to_numpy()
    out = values.astype(np.float32)
    out[missing] = np.nan
    return out


# (attribute name, column-name suffix, comparison). The suffix names the operator itself (gt/lt/gte/lte) rather than a
# reader-inferred direction ("above"/"below" read as strict but ``classification_lower_thresholds`` was >=, not >).
_THRESHOLD_OPS: Tuple[Tuple[str, str, Any], ...] = (
    ("classification_gt_thresholds", "gt", lambda s, t: s > t),
    ("classification_lt_thresholds", "lt", lambda s, t: s < t),
    ("classification_gte_thresholds", "gte", lambda s, t: s >= t),
    ("classification_lte_thresholds", "lte", lambda s, t: s <= t),
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
    learning_to_rank_targets : Iterable, optional
        Column names of graded-relevance labels to use as LEARNING_TO_RANK targets. Pair with ``group_field`` (the
        query-id column) so the suite forms per-query ranking blocks; without it ranking has no query grouping.
    classification_exact_values : dict, optional
        Dict mapping column names to exact values for binary classification.
        Example: {"status": 1} creates target "status_eq_1".
    classification_gt_thresholds : dict, optional
        Dict mapping column names to strict lower bounds (col > value).
        Example: {"score": 0.5} creates target "score_gt_0.5".
    classification_lt_thresholds : dict, optional
        Dict mapping column names to strict upper bounds (col < value).
        Example: {"score": 0.8} creates target "score_lt_0.8".
    classification_gte_thresholds : dict, optional
        Dict mapping column names to inclusive lower bounds (col >= value).
        Example: {"score": 0.5} creates target "score_gte_0.5".
    classification_lte_thresholds : dict, optional
        Dict mapping column names to inclusive upper bounds (col <= value).
        Example: {"score": 0.8} creates target "score_lte_0.8".
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
    >>> import numpy as np, pandas as pd
    >>> df = pd.DataFrame({
    ...     "date": pd.date_range("2024-01-01", periods=20, freq="D"),
    ...     "x": np.arange(20.0),
    ...     "price": np.linspace(1, 20, 20),
    ...     "quality": np.tile([1, 5, 9, 4], 5),
    ... })
    >>> extractor = SimpleFeaturesAndTargetsExtractor(
    ...     regression_targets=["price"],
    ...     classification_targets=["quality"],
    ...     classification_gte_thresholds={"quality": 3},
    ...     classification_lte_thresholds={"quality": 8},
    ...     ts_field="date",
    ... )
    >>> df, targets, *rest = extractor.transform(df)
    >>> sorted(name for per_type in targets.values() for name in per_type)
    ['price', 'quality_gte_3', 'quality_lte_8']
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
        learning_to_rank_targets: Optional[Iterable] = None,
        classification_exact_values: Optional[dict] = None,
        classification_gt_thresholds: Optional[dict] = None,
        classification_lt_thresholds: Optional[dict] = None,
        classification_gte_thresholds: Optional[dict] = None,
        classification_lte_thresholds: Optional[dict] = None,
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
        self.learning_to_rank_targets = learning_to_rank_targets
        self.classification_gt_thresholds = classification_gt_thresholds
        self.classification_lt_thresholds = classification_lt_thresholds
        self.classification_gte_thresholds = classification_gte_thresholds
        self.classification_lte_thresholds = classification_lte_thresholds
        self.classification_exact_values = classification_exact_values
        self.use_uniform_weighting = use_uniform_weighting
        self.use_recency_weighting = use_recency_weighting
        self.cyclical_version = CYCLICAL_ENCODING_VERSION  # pinned at construction; predict with this object replays it

    def add_features(self, df: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]:
        """Derive datetime features from ``ts_field`` (if configured) and record the emitted column names so the suite skips re-decomposing the same timestamp column downstream."""
        if self.ts_field and self.datetime_features:
            if self.verbose:
                logger.info("create_date_features %s over column %s...", self.datetime_features, self.ts_field)
            _pre_cols = set(df.columns)
            # An extractor pickled before the encoding was versioned has no attribute and replays version 1.
            _version = getattr(self, "cyclical_version", LEGACY_CYCLICAL_ENCODING_VERSION)
            df = create_date_features(df, cols=[self.ts_field], delete_original_cols=False, methods=self.datetime_features, cyclical_version=_version)
            _derived = [c for c in df.columns if c not in _pre_cols]
            # Record so the suite (``_phase_fit_pipeline``) can SKIP re-decomposing ``ts_field`` -- the second pass would emit duplicate / overwriting cols.
            self.ftextractor_emitted_columns[self.ts_field] = _derived
        return df

    def build_targets(self, df: Union[pd.DataFrame, pl.DataFrame]) -> Dict[TargetTypes, Dict[str, Any]]:
        """Build regression and classification targets from DataFrame columns.

        Args:
            df: Input DataFrame.

        Returns:
            Dictionary mapping TargetTypes to dicts of target arrays.

        Raises:
            KeyError: If a required target column is not found in the DataFrame.
        """
        target_by_type: Dict[TargetTypes, Dict[str, Any]] = {}
        if self.columns_to_drop is None:
            self.columns_to_drop = set()
        df_columns = set(df.columns)

        if self.classification_targets:
            targets = {}
            for col in self.classification_targets:
                if col not in df_columns:
                    raise KeyError(f"Classification target column '{col}' not found in DataFrame. Available: {list(df.columns)[:10]}...")
                # A missing source label stays missing in every target derived from it: a fillna(0) used to label those
                # rows as the positive class when thresh_val<=0 and as a real 0 otherwise. The suite decides what to do
                # with rows without a label.
                col_data = df[col]
                missing = _missing_label_mask(col_data)

                # Threshold targets: one column per configured (attr, op) whose mapping names this col.
                _has_threshold = False
                for _attr, _suffix, _cmp in _THRESHOLD_OPS:
                    _mapping = getattr(self, _attr, None)
                    if _mapping and col in _mapping:
                        thresh_val = _mapping[col]
                        target_name = f"{col}_{_suffix}_{thresh_val}"
                        targets[target_name] = _binary_target(_cmp(col_data, thresh_val), missing)
                        _has_threshold = True

                # Process exact values
                if self.classification_exact_values and col in self.classification_exact_values:
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
                        targets[target_name] = _binary_target(col_data == val, missing)

                # Default: use column as-is. Don't pre-cast to int8 here -- intize_targets()
                # below promotes int8/16/32/64 based on actual value range, so multiclass labels
                # with cardinality >127 don't wrap silently (pandas) or raise (polars).
                if not _has_threshold and col not in (self.classification_exact_values or {}):
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

        if self.learning_to_rank_targets:
            # Learning-to-rank: graded relevance labels (numeric). Query groups come from ``group_field`` (the
            # base transform emits group_ids); without it the suite cannot form per-query ranking blocks, so warn.
            if self.group_field is None:
                logger.warning(
                    "learning_to_rank_targets set but group_field is None: ranking has no query grouping, so the "
                    "suite will treat every row as its own/one query. Pass group_field=<query-id column> for real LtR."
                )
            targets = {}
            for col in self.learning_to_rank_targets:
                if col not in df_columns:
                    raise KeyError(f"Learning-to-rank target column '{col}' not found in DataFrame. Available: {list(df.columns)[:10]}...")
                targets[col] = df[col]
                self.columns_to_drop.add(col)
            target_by_type[TargetTypes.LEARNING_TO_RANK] = targets

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

    def get_sample_weights(self, df: Union[pd.DataFrame, pl.DataFrame], timestamps: Optional[pd.Series] = None) -> Dict[str, np.ndarray]:
        """Return sample weights based on configured weighting options.

        Args:
            df: The DataFrame.
            timestamps: Timestamp series if available.

        Returns:
            Dict with enabled weight schemes. Empty dict if no weighting enabled.
        """
        weights: Dict[str, Any] = {}

        if self.use_uniform_weighting:
            weights["uniform"] = None

        if self.use_recency_weighting and timestamps is not None:
            weights["recency"] = get_sample_weights_by_recency(timestamps)

        return weights
