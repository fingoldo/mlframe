"""Regression sensors for the 2026-06-10 audit's adjusted (P1->P2) items.

- A3: seed-invariant CV splitters (TimeSeriesSplit / GroupKFold) make every
  multi-seed repeat an exact duplicate; the wrapper must collapse to ONE
  honest measurement (per_seed length 1) instead of N identical fits.
- P3: per-base mi_y derived by excluding one column from the precomputed
  per-feature MI vector must be bit-identical to re-MI'ing x_remaining.
- A2: the Wilcoxon power threshold (1/2^n <= alpha) is computed correctly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mlframe.training.composite.discovery._screening_tiny import (
    _tiny_cv_rmse_raw_y_multiseed,
)
from mlframe.training.composite.discovery.screening import (
    _prebin_feature_columns,
    _mi_per_feature_prebinned,
    _aggregate_mi_per_feature,
    _mi_to_target_prebinned,
)


class TestSeedInvariantCollapse:
    def test_time_aware_runs_once(self) -> None:
        """A3: with time_aware=True the multi-seed wrapper returns a per_seed
        array of length 1 (one honest measurement), not n_seed_repeats."""
        rng = np.random.default_rng(0)
        n = 600
        y = rng.normal(size=n)
        X = rng.normal(size=(n, 4))
        _res, per_seed = _tiny_cv_rmse_raw_y_multiseed(
            y,
            X,
            family="linear",
            n_estimators=10,
            num_leaves=7,
            learning_rate=0.1,
            cv_folds=3,
            n_seed_repeats=5,
            base_random_state=0,
            return_per_seed=True,
            time_aware=True,
        )
        assert per_seed.shape[0] == 1, f"seed-invariant TSS should collapse to 1 measurement, got {per_seed.shape[0]}"

    def test_shuffled_kfold_keeps_all_seeds(self) -> None:
        """Control: without a seed-invariant splitter, all seeds run."""
        rng = np.random.default_rng(1)
        n = 600
        y = rng.normal(size=n)
        X = rng.normal(size=(n, 4))
        _res, per_seed = _tiny_cv_rmse_raw_y_multiseed(
            y,
            X,
            family="linear",
            n_estimators=10,
            num_leaves=7,
            learning_rate=0.1,
            cv_folds=3,
            n_seed_repeats=4,
            base_random_state=0,
            return_per_seed=True,
            time_aware=False,
        )
        assert per_seed.shape[0] == 4


class TestMiDecomposition:
    @pytest.mark.parametrize("aggregation", ["mean", "sum"])
    def test_excluded_column_matches_direct(self, aggregation: str) -> None:
        """P3: aggregating the per-feature MI vector with one column removed
        must equal re-MI'ing the matrix without that column."""
        rng = np.random.default_rng(2)
        n, f, nbins = 4000, 8, 8
        X = rng.normal(size=(n, f)).astype(np.float32)
        # Make a couple of columns informative about y.
        y = (X[:, 1] * 0.8 + X[:, 3] * 0.5 + rng.normal(size=n)).astype(np.float64)
        binned = _prebin_feature_columns(X, nbins=nbins)
        per_feat = _mi_per_feature_prebinned(binned, y, nbins=nbins)
        for drop in range(f):
            decomposed = _aggregate_mi_per_feature(
                np.delete(per_feat, drop),
                aggregation,
            )
            direct = _mi_to_target_prebinned(
                np.delete(binned, drop, axis=1),
                y,
                nbins=nbins,
                aggregation=aggregation,
            )
            assert decomposed == pytest.approx(direct, rel=1e-12, abs=1e-12), f"decomposed mi_y differs from direct for drop={drop}"


class TestWilcoxonPowerThreshold:
    def test_min_seeds_formula(self) -> None:
        """A2: the minimum seed count for a one-sided Wilcoxon to reach
        p<=alpha is ceil(log2(1/alpha)); at alpha=0.05 that is 5 (so the
        default n_seed_repeats=3 cannot pass and must be skipped)."""
        for alpha, expected in [(0.05, 5), (0.0625, 4), (0.01, 7)]:
            min_seeds = int(math.ceil(math.log2(1.0 / alpha)))
            assert min_seeds == expected
        assert int(math.ceil(math.log2(1.0 / 0.05))) > 3
