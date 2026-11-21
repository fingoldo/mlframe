"""Audit 2026-11-20 finding 40-F17: replay uses the effective_k the model was fitted with, not one recomputed from the serving batch."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.training.pipeline._cross_sectional_composite_fe import _ReplayConfig, apply_cross_sectional_composite_fe


def _frame(n_snapshots, seed=0):
    """A frame with several snapshots of several rows each."""
    rng = np.random.default_rng(seed)
    rows = n_snapshots * 4
    return pd.DataFrame({"snap": np.repeat(np.arange(n_snapshots), 4), "f": rng.normal(size=rows)})


def _cfg(k):
    """A config for the step with the given neighbour count."""
    return SimpleNamespace(cross_sectional_neighbors_snapshot_col="snap", cross_sectional_neighbors_feature_cols=["f"], cross_sectional_neighbors_k=k, cross_sectional_neighbors_agg_stats=["mean"])


def test_replay_config_carries_the_fitted_effective_k():
    """The fitted value persisted in metadata is what the replay config holds."""
    assert _ReplayConfig({"cross_sectional_neighbors_effective_k": 2})._fitted_effective_k == 2


def test_a_fitted_effective_k_overrides_the_serving_batch_derivation():
    """With a fitted effective_k of 2, a config with k=10 produces the same features as k=2 on a larger serving batch."""
    serving = _frame(n_snapshots=8)
    cfg_replay = _cfg(k=10)
    cfg_replay._fitted_effective_k = 2
    replayed, _, _ = apply_cross_sectional_composite_fe(serving, None, None, cfg_replay, metadata=None)
    direct, _, _ = apply_cross_sectional_composite_fe(serving, None, None, _cfg(k=2), metadata=None)
    pd.testing.assert_frame_equal(replayed, direct)
