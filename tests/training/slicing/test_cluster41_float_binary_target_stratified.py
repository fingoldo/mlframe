"""Cluster 41 regression: float-coded binary targets get stratified shards (audit 2026-11-20, finding 41-F22)."""

import numpy as np

from mlframe.training.slicing._slice_helpers import _is_classification_target, _random_shards


def test_float_zero_one_target_is_classification():
    """A 0.0/1.0 float target is recognised as classification, like its integer twin (41-F22)."""
    y = np.zeros(100, dtype=float)
    y[:5] = 1.0
    assert _is_classification_target(y)


def test_float_target_shards_keep_positives_in_every_shard():
    """With 5% positives in 100 rows and 5 shards, each shard holds a positive under stratification (41-F22)."""
    y = np.zeros(100, dtype=float)
    y[:5] = 1.0
    shards = _random_shards(100, 5, y, random_state=0)
    assert all(y[idx].sum() >= 1 for idx in shards)


def test_continuous_float_target_is_not_classification():
    """Control: a continuous float target stays regression-like (41-F22)."""
    y = np.linspace(0.0, 1.0, 100)
    assert not _is_classification_target(y)
