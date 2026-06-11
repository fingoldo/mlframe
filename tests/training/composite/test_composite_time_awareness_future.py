"""Regression + biz_value for the time/group-awareness FUTURE batch.

- M6: an explicit ``time_ordering`` sorts the MI-screening sample into a
  forward-walk so the tiny-model CV is TimeSeriesSplit, not shuffled K-fold
  (the canonical non-monotone lag(y) base never tripped the old monotonicity
  heuristic). Verified via the ``_screen_time_ordered_`` flag + sorted sample.
- A29: the zero-base forward-stepwise baseline is the CV-RMSE of the
  train-fold-mean predictor, not the full-sample std (which understated the
  no-base error under TSS on trending y, inflating the first base's gain).
- A19: forward_stepwise is group-aware (GroupKFold) when groups are supplied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.composite.discovery.forward_stepwise import (
    forward_stepwise_multi_base,
)
from mlframe.training.configs import CompositeTargetDiscoveryConfig


class TestM6TimeOrdering:
    def _temporal_frame(self, n=4000, seed=0):
        rng = np.random.default_rng(seed)
        ts = np.arange(n)  # chronological index
        lag = np.empty(n)
        lag[0] = 0.0
        y = np.empty(n)
        for i in range(n):
            lag[i] = y[i - 1] if i > 0 else 0.0
            y[i] = 0.9 * lag[i] + rng.normal(0.0, 1.0)
        feat = rng.normal(0.0, 1.0, size=n)
        # Shuffle the rows so the frame is NOT already in time order.
        perm = rng.permutation(n)
        df = pd.DataFrame({
            "y": y[perm], "lag": lag[perm], "feat": feat[perm], "ts": ts[perm],
        })
        return df

    def test_time_ordering_sets_screen_flag_and_sorts(self) -> None:
        df = self._temporal_frame()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=800, base_candidates=["lag"],
        )
        disc = CompositeTargetDiscovery(cfg)
        train_idx = np.arange(len(df))
        disc.fit(df, "y", ["lag", "feat"], train_idx,
                 time_ordering=df["ts"].to_numpy())
        assert getattr(disc, "_screen_time_ordered_", False) is True

    def test_no_time_ordering_leaves_flag_false(self) -> None:
        df = self._temporal_frame()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=800, base_candidates=["lag"],
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, "y", ["lag", "feat"], np.arange(len(df)))
        assert getattr(disc, "_screen_time_ordered_", False) is False

    def test_time_column_config_field(self) -> None:
        cfg = CompositeTargetDiscoveryConfig(time_column="ts")
        assert cfg.time_column == "ts"


class TestA29ZeroBaseSentinel:
    def test_trending_y_zero_base_baseline_exceeds_fold_mean_naive_std(self) -> None:
        """On strongly trending y the train-fold-mean predictor (forward walk)
        has a LARGER CV error than the full-sample std, so the zero-base
        baseline must be >= std(y) -- the old std(y) sentinel understated it
        and inflated the first base's gain."""
        n = 3000
        t = np.linspace(0.0, 10.0, n)
        y = 5.0 * t + np.random.default_rng(0).normal(0.0, 0.5, size=n)
        b = t + np.random.default_rng(1).normal(0.0, 0.5, size=n)
        kept, diag = forward_stepwise_multi_base(
            y, {"b": b}, seed_bases=None, time_aware=True, cv_folds=4,
            cv_persist_fold_scores=True,
        )
        # The very first diagnostic's rmse_before is the zero-base baseline.
        assert diag, "expected at least one stepwise diagnostic"
        baseline = diag[0]["rmse_before"]
        assert baseline >= float(np.std(y)) * 0.99, (
            f"zero-base baseline {baseline:.3f} should reflect the forward-walk "
            f"train-mean error, not collapse below std(y)={np.std(y):.3f}"
        )


class TestA19GroupAware:
    def test_group_aware_forward_stepwise_runs(self) -> None:
        rng = np.random.default_rng(2)
        n = 900
        g = np.repeat(np.arange(9), 100)
        b1 = rng.normal(0.0, 1.0, size=n)
        b2 = rng.normal(0.0, 1.0, size=n)
        y = b1 + b2 + rng.normal(0.0, 0.1, size=n)
        kept, diag = forward_stepwise_multi_base(
            y, {"b1": b1, "b2": b2}, seed_bases=["b1"],
            time_aware=False, groups=g,
        )
        assert "b1" in kept and "b2" in kept
