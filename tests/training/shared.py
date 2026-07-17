"""
Shared utilities for training module tests.

This module contains shared classes and helper functions used across multiple test files.
"""

import pytest
import numpy as np
import pandas as pd

from mlframe.training.configs import TargetTypes


class SimpleFeaturesAndTargetsExtractor:
    """Mock FeaturesAndTargetsExtractor for testing.

    target_type override
    --------------------
    By default, ``regression=True`` → REGRESSION; ``regression=False`` →
    BINARY_CLASSIFICATION. For multiclass / multilabel tests, pass
    ``target_type`` directly:
        - ``target_type=TargetTypes.MULTICLASS_CLASSIFICATION`` — target column
          is a 1-D int label (0..K-1)
        - ``target_type=TargetTypes.MULTILABEL_CLASSIFICATION`` — target column
          is either:
            (a) a polars ``pl.List(pl.Int8)`` / ``pl.Array(pl.Int8, K)`` column
                (auto-unpacked to (N, K) ndarray), OR
            (b) a pandas object column where each cell is a list/tuple of K ints
                (auto-stacked to (N, K) ndarray), OR
            (c) a 2-D ndarray already.
    """

    def __init__(
        self,
        target_column="target",
        regression=True,
        target_type=None,
        ts_field=None,
        group_field=None,
        weight_schemas=None,
        target_carrier="numpy",
        extra_targets=None,
    ):
        self.target_column = target_column
        self.regression = regression
        self._explicit_target_type = target_type
        # 2026-04-27 Session 7 batch 6: optional timestamp-column hint.
        # When set, train_mlframe_models_suite picks this up via
        # ``getattr(features_and_targets_extractor, 'ts_field', None)``
        # for the temporal_audit auto-detect (drift report graph + change
        # points). Mirrors the same attribute on the production
        # SimpleFeaturesAndTargetsExtractor.
        self.ts_field = ts_field
        # 2026-04-27 Session 7 batch 7: surface ``group_field`` and
        # ``weight_schemas`` so fuzz combos can exercise the
        # group-aware (per-group AUC, GroupKFold) and recency-weighting
        # code paths. Previously these were dead axes — combo carried
        # the values but the fixture's transform always returned
        # ``group_ids=None, sample_weights={}``, so the suite silently
        # fell back to the no-group / uniform-weighting branches.
        self.group_field = group_field
        # ``target_carrier='native'`` keeps single-output targets as
        # pandas/polars Series instead of normalising to ndarray. Fuzz uses
        # this to exercise real suite target-carrier paths such as MRMR's
        # supervised fit_transform target handling.
        self.target_carrier = target_carrier
        # ``weight_schemas`` is a tuple/list/None of strings, e.g.
        # ``("uniform",)`` (default behaviour) or ``("uniform", "recency")``.
        # The fixture generates the actual weight arrays at transform()
        # time so the suite's per-weight loop runs once per name.
        self.weight_schemas = tuple(weight_schemas) if weight_schemas else None
        # 2026-05-21 iter150 -- multi-target axis. When set, transform()
        # populates target_by_type with EXTRA (target_type, target_name)
        # entries so the suite's per-target outer loop runs more than
        # once per fit. Values:
        #   None              -- legacy single-target.
        #   "same_type_2"     -- add 1 more target of the SAME primary type.
        #                        Exercises ``targets.items()`` inner loop.
        #   "mixed_reg_bin"   -- add a BINARY classification secondary
        #                        target alongside the regression primary.
        #                        Exercises ``target_by_type.items()`` outer
        #                        loop with 2 different keys.
        # Extra-target values are synthesised from a per-extractor RNG
        # (seeded off the primary target_column for reproducibility) so
        # the fuzz frame builder doesn't need to emit extra columns.
        self.extra_targets = extra_targets

    def _resolve_target_type(self):
        if self._explicit_target_type is not None:
            return self._explicit_target_type
        return TargetTypes.REGRESSION if self.regression else TargetTypes.BINARY_CLASSIFICATION

    def _extract_target_values(self, df):
        """Returns 1-D or 2-D ndarray. Auto-detects polars list/array
        target columns and unpacks them to (N, K)."""
        target_type = self._resolve_target_type()
        col = df[self.target_column]
        if isinstance(df, pd.DataFrame):
            raw = col.values
        else:  # Polars
            raw = col
        # Multilabel: unpack list-typed column or stack object cells.
        if target_type == TargetTypes.MULTILABEL_CLASSIFICATION:
            if not isinstance(df, pd.DataFrame):
                # Polars: try List/Array dtype first.
                import polars as pl

                if isinstance(col.dtype, pl.List) or (hasattr(pl, "Array") and isinstance(col.dtype, pl.Array)):
                    # Stack list-of-lists into (N, K) ndarray.
                    py_list = col.to_list()
                    return np.asarray(py_list, dtype=np.int8)
                # Fallback: bare 2-D-aware to_numpy
                arr = col.to_numpy()
                if arr.ndim == 2:
                    return arr.astype(np.int8)
                # 1-D object column with list cells (rare) — stack
                return np.stack([np.asarray(c, dtype=np.int8) for c in arr])
            else:
                arr = raw
                if arr.ndim == 2:
                    return arr.astype(np.int8)
                if arr.dtype == object:
                    return np.stack([np.asarray(c, dtype=np.int8) for c in arr])
                return arr
        # Multi-target regression: unpack the (N, K) continuous list column
        # exactly like multilabel, but keep float dtype (multilabel forces
        # int8). The fuzz frame builder emits this column only for combos
        # whose every model natively handles a 2-D continuous target (see
        # build_frame_for_combo's _NATIVE_MTR_MODELS gate); non-native combos
        # are downgraded to a 1-D ``target_reg`` column upstream and reach
        # this method as REGRESSION, never here.
        if target_type == TargetTypes.MULTI_TARGET_REGRESSION:
            if not isinstance(df, pd.DataFrame):
                import polars as pl

                if isinstance(col.dtype, pl.List) or (hasattr(pl, "Array") and isinstance(col.dtype, pl.Array)):
                    return np.asarray(col.to_list(), dtype=np.float32)
                arr = col.to_numpy()
                if arr.ndim == 2:
                    return arr.astype(np.float32)
                return np.stack([np.asarray(c, dtype=np.float32) for c in arr])
            arr = raw
            if arr.ndim == 2:
                return arr.astype(np.float32)
            if arr.dtype == object:
                return np.stack([np.asarray(c, dtype=np.float32) for c in arr])
            return arr
        # Default: 1-D values (regression / binary / multiclass label).
        if self.target_carrier == "native":
            return col
        if isinstance(df, pd.DataFrame):
            return raw
        return col.to_numpy()

    def transform(self, df):
        """
        Transform method that returns the expected tuple.

        Returns: (df, target_by_type, group_ids_raw, group_ids, timestamps, artifacts, columns_to_drop, sample_weights)
        """
        target_values = self._extract_target_values(df)
        target_type = self._resolve_target_type()
        target_by_type = {target_type: {self.target_column: target_values}}

        # 2026-05-21 iter150 -- inject extra synthetic targets per
        # ``self.extra_targets``. Synthetic values come from a deterministic
        # RNG seeded off the primary target_column hash + N so re-running
        # the same fuzz combo produces identical extra targets.
        if self.extra_targets:
            _n_rows = df.shape[0] if hasattr(df, "shape") else len(df)
            _seed = abs(hash((self.target_column, _n_rows))) % (2**32)
            _rng = np.random.default_rng(_seed)
            if self.extra_targets == "same_type_2":
                # Add a second target of the SAME primary type. Synthesise
                # in the shape the suite expects per target_type so
                # downstream model dispatch works without special-casing.
                _name2 = self.target_column + "_extra"
                if target_type == TargetTypes.REGRESSION:
                    _extra_vals = _rng.standard_normal(_n_rows).astype(np.float32)
                elif target_type == TargetTypes.BINARY_CLASSIFICATION:
                    _extra_vals = (_rng.random(_n_rows) > 0.5).astype(np.int8)
                elif target_type == TargetTypes.MULTICLASS_CLASSIFICATION:
                    # 3-class default; the suite's class-discovery handles
                    # any K so this is fine even if primary K differs.
                    _extra_vals = _rng.integers(0, 3, size=_n_rows).astype(np.int8)
                else:
                    _extra_vals = None
                if _extra_vals is not None:
                    target_by_type[target_type][_name2] = _extra_vals
            elif self.extra_targets == "mixed_reg_bin":
                # Add a binary secondary target alongside the primary
                # target_type. The outer target_by_type dict ends up
                # with at least 2 different target_type keys -- exercises
                # the ``for target_type, targets in target_by_type.items()``
                # outer loop at core/main.py:952. The fuzz canonicaliser
                # collapses (mixed_reg_bin, primary!=regression) combos
                # for dedup, but the surviving combo's actual field value
                # still reaches here -- and adding a binary extra alongside
                # any non-binary primary remains a valid multi-type test.
                _bin_vals = (_rng.random(_n_rows) > 0.5).astype(np.int8)
                target_by_type.setdefault(TargetTypes.BINARY_CLASSIFICATION, {})[self.target_column + "_bin"] = _bin_vals
            elif self.extra_targets == "mixed_reg_bin_2each":
                # Cartesian product: 2 primary-type targets AND 2 binary
                # secondary targets. The suite trains all 4 in one
                # invocation, so BOTH the target_by_type.items() outer
                # loop AND the targets.items() inner loop iterate twice
                # each. Stress-tests per-target isolation (no FS / pre-
                # pipeline cache contamination between targets), ensemble
                # flavour assembly across heterogeneous types, and the
                # nested-dict metadata layout. Canonicaliser keeps this
                # value only when primary == regression.
                _add_name = self.target_column + "_extra"
                _bin_name1 = self.target_column + "_bin"
                _bin_name2 = self.target_column + "_bin2"
                if target_type == TargetTypes.REGRESSION:
                    _extra_vals = _rng.standard_normal(_n_rows).astype(np.float32)
                elif target_type == TargetTypes.BINARY_CLASSIFICATION:
                    _extra_vals = (_rng.random(_n_rows) > 0.5).astype(np.int8)
                elif target_type == TargetTypes.MULTICLASS_CLASSIFICATION:
                    _extra_vals = _rng.integers(0, 3, size=_n_rows).astype(np.int8)
                else:
                    _extra_vals = None
                if _extra_vals is not None:
                    target_by_type[target_type][_add_name] = _extra_vals
                # 2 binary targets (each from a fresh RNG draw for
                # statistical independence).
                _bin1_vals = (_rng.random(_n_rows) > 0.5).astype(np.int8)
                _bin2_vals = (_rng.random(_n_rows) > 0.4).astype(np.int8)
                target_by_type.setdefault(TargetTypes.BINARY_CLASSIFICATION, {})[_bin_name1] = _bin1_vals
                target_by_type[TargetTypes.BINARY_CLASSIFICATION][_bin_name2] = _bin2_vals

        # group_ids extraction (Session 7 batch 7): when ``group_field``
        # is set and present in df, return the column as group_ids_raw
        # AND group_ids (suite consumers may use either; prod FTE
        # mirrors the same value into both slots when it has a single
        # group source).
        group_ids_raw = None
        group_ids = None
        if self.group_field is not None and self.group_field in df.columns:
            col = df[self.group_field]
            if isinstance(df, pd.DataFrame):
                group_ids_raw = col.values
            else:
                group_ids_raw = col.to_numpy()
            group_ids = group_ids_raw

        # timestamps extraction: prod FTE returns the ts_field column
        # alongside the targets. Several downstream consumers (recency
        # weighting, time-aware splits, temporal_audit) consume this.
        timestamps = None
        if self.ts_field is not None and self.ts_field in df.columns:
            col = df[self.ts_field]
            if isinstance(df, pd.DataFrame):
                timestamps = col.values
            else:
                timestamps = col.to_numpy()

        # sample_weights: previously always {}. When the caller asks
        # for weight_schemas, generate one weight array per name. Keeps
        # the dead ``weight_schemas`` axis active in fuzz combos.
        sample_weights: dict = {}
        if self.weight_schemas:
            n = df.shape[0] if hasattr(df, "shape") else len(df)
            for name in self.weight_schemas:
                if name == "uniform":
                    sample_weights[name] = np.ones(n, dtype=np.float32)
                elif name == "recency":
                    # Linear ramp: oldest row -> 0.5, newest -> 1.5.
                    # Production FTE uses an exponential decay over
                    # timestamps; the test fixture just needs SOMETHING
                    # non-uniform so the recency code branch runs.
                    if timestamps is not None:
                        ts_arr = np.asarray(timestamps)
                        # Convert datetime to float for ranking
                        try:
                            ts_f = ts_arr.astype("datetime64[ns]").astype(np.int64).astype(np.float64)
                        except (TypeError, ValueError):
                            ts_f = np.arange(n, dtype=np.float64)
                        rank = np.argsort(np.argsort(ts_f)).astype(np.float64) / max(n - 1, 1)
                    else:
                        rank = np.arange(n, dtype=np.float64) / max(n - 1, 1)
                    sample_weights[name] = (0.5 + rank).astype(np.float32)
                else:
                    # Unknown schema: fall back to uniform so the suite
                    # doesn't crash on a missing key.
                    sample_weights[name] = np.ones(n, dtype=np.float32)

        # columns_to_drop: include ts_field / group_field along with
        # target_column so downstream models don't see the timestamp
        # or grouping column as a feature (prod FTE has the same
        # contract — these are bookkeeping columns, not features).
        cols_to_drop = [self.target_column]
        if self.ts_field is not None and self.ts_field not in cols_to_drop:
            cols_to_drop.append(self.ts_field)
        if self.group_field is not None and self.group_field not in cols_to_drop:
            cols_to_drop.append(self.group_field)

        return (
            df,
            target_by_type,
            group_ids_raw,
            group_ids,
            timestamps,
            None,  # artifacts
            cols_to_drop,
            sample_weights,
        )


class TimestampedFeaturesExtractor:
    """Mock FeaturesAndTargetsExtractor with timestamp support for testing sample weights."""

    def __init__(self, target_column="target", regression=True, ts_field=None, group_field=None, sample_weights=None):
        self.target_column = target_column
        self.regression = regression
        self.ts_field = ts_field
        self.group_field = group_field
        self._sample_weights = sample_weights or {}

    def transform(self, df):
        """
        Transform method that returns the expected tuple with sample_weights.
        """
        # Extract target
        if isinstance(df, pd.DataFrame):
            target_values = df[self.target_column].values
        else:  # Polars
            target_values = df[self.target_column].to_numpy()

        # Create target_by_type dict
        target_type = TargetTypes.REGRESSION if self.regression else TargetTypes.BINARY_CLASSIFICATION
        target_by_type = {target_type: {self.target_column: target_values}}

        # Extract timestamps if available
        timestamps = None
        if self.ts_field and self.ts_field in df.columns:
            if isinstance(df, pd.DataFrame):
                timestamps = df[self.ts_field]
            else:
                timestamps = df[self.ts_field].to_pandas()

        # Extract group_ids if available
        group_ids_raw = None
        group_ids = None
        if self.group_field and self.group_field in df.columns:
            if isinstance(df, pd.DataFrame):
                group_ids_raw = df[self.group_field]
            else:
                group_ids_raw = df[self.group_field].to_pandas()
            unique_vals, group_ids = np.unique(group_ids_raw.values, return_inverse=True)

        # Return all expected values
        return (
            df,  # df
            target_by_type,  # target_by_type
            group_ids_raw,  # group_ids_raw
            group_ids,  # group_ids
            timestamps,  # timestamps
            None,  # artifacts
            [self.target_column],  # columns_to_drop
            self._sample_weights,  # sample_weights dict
        )


def get_cpu_config(model_name, iterations=10):
    """Get CPU-forced config override for a model.

    Args:
        model_name: Name of the model (cb, xgb, lgb, mlp, etc.)
        iterations: Number of iterations for tree models

    Returns:
        Dict with config overrides to force CPU execution
    """
    config = {"iterations": iterations}
    overrides = {
        "cb": {"cb_kwargs": {"task_type": "CPU"}},
        "xgb": {"xgb_kwargs": {"device": "cpu"}},
        "lgb": {"lgb_kwargs": {"device_type": "cpu"}},
        "mlp": {"mlp_kwargs": {"trainer_params": {"devices": 1}}},
    }
    config.update(overrides.get(model_name, {}))
    return config


def skip_if_dependency_missing(model_name):
    """Check if model dependency is available, skip test if not.

    Args:
        model_name: Name of the model to check

    Returns:
        None if dependency available, calls pytest.skip() otherwise
    """
    deps = {
        "ngb": ("ngboost", "NGBoost"),
        "mlp": ("pytorch_lightning", "PyTorch Lightning"),
    }
    if model_name not in deps:
        return

    module_name, display_name = deps[model_name]
    try:
        __import__(module_name)
    except ImportError:
        pytest.skip(f"{display_name} not available")
