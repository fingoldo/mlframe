"""A split that cannot produce the cross-sectional neighbor columns must fail at fit time, not pass through with a raw schema.

Both silent paths let a train/test column mismatch reach the encoder later, where it surfaces as an unrelated dimension error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.pipeline import _cross_sectional_composite_fe as _m


class _Cfg:
    """Cross-sectional config carrying the snapshot key and two feature columns."""

    cross_sectional_neighbors_snapshot_col = "snap"
    cross_sectional_neighbors_feature_cols = ["f0", "f1"]
    cross_sectional_neighbors_k = 2
    cross_sectional_neighbors_agg_stats = ("mean",)


def _frame(n_snap: int = 6, per: int = 10) -> pd.DataFrame:
    """Frame with a snapshot key, two float features and ``per`` rows per snapshot."""
    rng = np.random.default_rng(1)
    snap = np.repeat(np.arange(n_snap), per)
    return pd.DataFrame({"snap": snap, "f0": rng.normal(size=n_snap * per), "f1": rng.normal(size=n_snap * per)})


def test_split_missing_snapshot_column_raises():
    """A val split without the snapshot key raises, naming the split, instead of being returned with its raw schema."""
    train = _frame()
    val = _frame().drop(columns=["snap"])

    with pytest.raises(ValueError, match="split 'val' lacks snapshot column"):
        _m.apply_cross_sectional_composite_fe(train, val, None, _Cfg(), {}, verbose=0)


def test_neighbor_computation_failure_propagates(monkeypatch):
    """A failure inside the neighbor computation is raised to the caller rather than swallowed for that split."""
    train = _frame()
    test = _frame()

    def _boom(*args, **kwargs):
        """Stand-in neighbour computation that always fails."""
        raise RuntimeError("neighbor search failed")

    monkeypatch.setattr(_m, "compute_cross_sectional_neighbor_features", _boom)

    with pytest.raises(RuntimeError, match="neighbor search failed"):
        _m.apply_cross_sectional_composite_fe(train, None, test, _Cfg(), {}, verbose=0)
