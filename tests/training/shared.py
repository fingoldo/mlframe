"""
Shared utilities for training module tests.

This module contains shared classes and helper functions used across multiple test files.
"""

import pytest
import numpy as np
import pandas as pd

from mlframe.training.configs import TargetTypes


class SimpleFeaturesAndTargetsExtractor:
    """Mock FeaturesAndTargetsExtractor for testing."""

    def __init__(self, target_column='target', regression=True):
        self.target_column = target_column
        self.regression = regression

    def transform(self, df):
        """
        Transform method that returns the expected tuple.

        Returns: (df, target_by_type, group_ids_raw, group_ids, timestamps, artifacts, columns_to_drop, sample_weights)
        """
        # Extract target
        if isinstance(df, pd.DataFrame):
            target_values = df[self.target_column].values
        else:  # Polars
            target_values = df[self.target_column].to_numpy()

        # Create target_by_type dict
        target_type = TargetTypes.REGRESSION if self.regression else TargetTypes.BINARY_CLASSIFICATION
        target_by_type = {
            target_type: {
                self.target_column: target_values
            }
        }

        # Return all expected values
        return (
            df,  # df
            target_by_type,  # target_by_type
            None,  # group_ids_raw
            None,  # group_ids
            None,  # timestamps
            None,  # artifacts
            [self.target_column],  # columns_to_drop
            {},  # sample_weights (empty dict = uniform weights only)
        )


class TimestampedFeaturesExtractor:
    """Mock FeaturesAndTargetsExtractor with timestamp support for testing sample weights."""

    def __init__(self, target_column='target', regression=True, ts_field=None, group_field=None, sample_weights=None):
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
        target_by_type = {
            target_type: {
                self.target_column: target_values
            }
        }

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
