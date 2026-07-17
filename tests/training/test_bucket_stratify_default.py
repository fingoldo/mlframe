"""Regression sensor for D1 P2 #7.

``TrainingSplitConfig.bucket_stratify`` defaults to ``True`` and applies
to ALL target types:
- Classification: existing stratify-by-class path (already covered).
- Regression: bin y into deciles (quartiles if n<5000), stratify by bin so
  heavy-tail rows don't concentrate in val or test.

Combine with groups: when ``iterative-stratification`` is unavailable the
group constraint wins (no per-row group leakage is the stronger invariant);
INFO surfaces the precedence decision so the operator sees it.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import pytest


class _DummyTargetTypeRegr:
    name = "regression"


class _DummyTargetTypeClass:
    name = "classification"


class _DummySplitConfig:
    """Minimal config mock matching what _phase_train_val_test_split reads."""

    def __init__(self, **overrides):
        self.test_size = overrides.get("test_size", 0.2)
        self.val_size = overrides.get("val_size", 0.2)
        self.composite_cardinality_cap = overrides.get("composite_cardinality_cap", 200)
        self.bucket_stratify = overrides.get("bucket_stratify", True)
        self.use_groups = overrides.get("use_groups", True)
        self.calib_size = None
        self.shuffle_val = False
        self.shuffle_test = False
        self.val_sequential_fraction = 0.5
        self.test_sequential_fraction = None
        self.trainset_aging_limit = None
        self.wholeday_splitting = True
        self.random_seed = 42
        self.val_placement = "forward"

    def model_dump(self, exclude=None):
        d = {
            "test_size": self.test_size,
            "val_size": self.val_size,
            "shuffle_val": self.shuffle_val,
            "shuffle_test": self.shuffle_test,
            "val_sequential_fraction": self.val_sequential_fraction,
            "test_sequential_fraction": self.test_sequential_fraction,
            "trainset_aging_limit": self.trainset_aging_limit,
            "wholeday_splitting": self.wholeday_splitting,
            "random_seed": self.random_seed,
            "val_placement": self.val_placement,
        }
        if exclude:
            for k in exclude:
                d.pop(k, None)
        return d


def test_d1_p2_7_regression_bucket_stratify_runs_on_heavy_tail_target(caplog):
    """Heavy-tail regression target: bucket-stratify INFO fires + buckets compute."""
    from mlframe.training.core._phase_helpers_fit_split import _phase_train_val_test_split

    rng = np.random.default_rng(42)
    n = 1000
    # Heavy-tail regression: most values small, a small tail of large values.
    y = np.concatenate([rng.exponential(scale=1.0, size=int(n * 0.95)), rng.exponential(scale=20.0, size=int(n * 0.05))])
    rng.shuffle(y)
    df = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n)})
    target_by_type = {_DummyTargetTypeRegr(): {"y": y}}
    metadata = {}

    with caplog.at_level(logging.INFO, logger="mlframe.training.core._phase_helpers_fit_split"):
        _phase_train_val_test_split(
            df=df,
            target_by_type=target_by_type,
            timestamps=None,
            group_ids=None,
            group_ids_raw=None,
            artifacts=None,
            sequences=None,
            split_config=_DummySplitConfig(),
            behavior_config=type("B", (), {"fairness_features": None})(),
            metadata=metadata,
            data_dir=None,
            models_dir=None,
            target_name="y",
            model_name="m",
            df_size_mb=1.0,
            verbose=False,
        )
    info_msgs = [r.message for r in caplog.records if r.levelno >= logging.INFO]
    bucket_msgs = [m for m in info_msgs if "Bucket-stratify" in m and "quantile bucket" in m]
    assert bucket_msgs, f"Expected bucket-stratify INFO log; got: {info_msgs}"


def test_d1_p2_7_regression_bucket_stratify_disabled_when_flag_false(caplog):
    """When bucket_stratify=False, no INFO line + no bucket strat."""
    from mlframe.training.core._phase_helpers_fit_split import _phase_train_val_test_split

    rng = np.random.default_rng(42)
    n = 1000
    y = rng.exponential(scale=1.0, size=n)
    df = pd.DataFrame({"x1": rng.standard_normal(n)})
    target_by_type = {_DummyTargetTypeRegr(): {"y": y}}

    with caplog.at_level(logging.INFO, logger="mlframe.training.core._phase_helpers_fit_split"):
        _phase_train_val_test_split(
            df=df,
            target_by_type=target_by_type,
            timestamps=None,
            group_ids=None,
            group_ids_raw=None,
            artifacts=None,
            sequences=None,
            split_config=_DummySplitConfig(bucket_stratify=False),
            behavior_config=type("B", (), {"fairness_features": None})(),
            metadata={},
            data_dir=None,
            models_dir=None,
            target_name="y",
            model_name="m",
            df_size_mb=1.0,
            verbose=False,
        )
    info_msgs = [r.message for r in caplog.records if r.levelno >= logging.INFO]
    assert not any("Bucket-stratify" in m for m in info_msgs), f"Did not expect bucket-stratify when flag=False; got: {info_msgs}"


def test_d1_p2_7_bucket_distribution_balanced_in_val(caplog):
    """KS test: val bucket distribution should NOT differ wildly from train."""
    from mlframe.training.core._phase_helpers_fit_split import _phase_train_val_test_split
    from scipy.stats import ks_2samp

    rng = np.random.default_rng(123)
    n = 2000
    # Heavy-tail with strong skew
    y = np.concatenate([rng.exponential(scale=1.0, size=int(n * 0.9)), rng.exponential(scale=30.0, size=int(n * 0.1))])
    rng.shuffle(y)
    df = pd.DataFrame({"x1": rng.standard_normal(n)})
    target_by_type = {_DummyTargetTypeRegr(): {"y": y}}

    metadata = {}
    res = _phase_train_val_test_split(
        df=df,
        target_by_type=target_by_type,
        timestamps=None,
        group_ids=None,
        group_ids_raw=None,
        artifacts=None,
        sequences=None,
        split_config=_DummySplitConfig(),
        behavior_config=type("B", (), {"fairness_features": None})(),
        metadata=metadata,
        data_dir=None,
        models_dir=None,
        target_name="y",
        model_name="m",
        df_size_mb=1.0,
        verbose=False,
    )
    train_y = y[res.train_idx]
    val_y = y[res.val_idx]
    # KS test: stratified should keep val distribution near train (>0.05 p-value typically)
    stat, p = ks_2samp(train_y, val_y)
    assert p > 0.001, f"Bucket-stratified val distribution differs from train (KS p={p:.4f})"
