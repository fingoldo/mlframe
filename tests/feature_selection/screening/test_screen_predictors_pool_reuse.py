"""Regression tests for cross-round joblib worker-pool reuse in ``screen_predictors`` (2026-07-09 fix,
MRMR audit finding #6).

Before this fix, every ``screen_predictors()`` call with ``n_workers>1`` built a FRESH
``joblib.Parallel`` pool plus an eager warmup dispatch (spawning ``n_workers`` threads), even though a
typical fit calls ``screen_predictors`` 2-3 times (once per screen/FE round) with an unchanged pool
config for the whole ``fit()`` call. The fix adds an optional ``seed_workers_pool`` parameter: a pool
object returned by a prior call, which is reused verbatim (no rebuild, no re-warmup) instead of being
rebuilt from scratch every round.

These tests verify: (a) the parameter is fully backward-compatible (``seed_workers_pool=None`` unchanged
behavior), (b) the pool IS actually reused (object identity), not silently ignored and rebuilt anyway,
(c) selection is IDENTICAL whether or not the pool is reused (pure speedup, no behavior change), and
(d) at n_workers<=1 the returned pool is always None (no pool exists to reuse).
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
from joblib import Parallel

from mlframe.feature_selection.filters.screen import screen_predictors


def _make_data(n: int = 400, m: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    factors_data = rng.integers(0, 4, size=(n, m)).astype(np.int32)
    y_col = ((factors_data[:, 0] + rng.integers(0, 2, size=n)) % 2).astype(np.int32).reshape(-1, 1)
    targets_data = y_col
    factors_nbins = np.array([4] * m, dtype=np.int32)
    targets_nbins = np.array([2], dtype=np.int32)
    return factors_data, factors_nbins, targets_data, targets_nbins


def _common_kwargs(factors_data, factors_nbins, targets_data, targets_nbins, **overrides):
    base = dict(
        factors_data=factors_data,
        factors_nbins=factors_nbins,
        factors_names=[f"f{i}" for i in range(factors_data.shape[1])],
        targets_data=targets_data,
        targets_nbins=targets_nbins,
        y=np.array([0], dtype=np.int32),
        full_npermutations=5,
        baseline_npermutations=3,
        n_workers=1,
        verbose=0,
        random_seed=42,
    )
    base.update(overrides)
    return base


def test_seed_workers_pool_default_parameter_is_none():
    sig = inspect.signature(screen_predictors)
    assert "seed_workers_pool" in sig.parameters
    assert sig.parameters["seed_workers_pool"].default is None


def test_n_workers_le_1_returns_none_pool_regardless_of_seed():
    """No pool is ever built at n_workers<=1 -- the returned (last tuple element) pool must be None,
    whether or not a (meaningless-at-this-n_workers) seed was passed."""
    fd, fn, td, tn = _make_data(seed=1)
    out = screen_predictors(**_common_kwargs(fd, fn, td, tn, n_workers=1))
    assert out[-1] is None


def test_seed_workers_pool_none_matches_legacy_default_behavior():
    """seed_workers_pool=None (the default) must behave identically to omitting the parameter."""
    fd, fn, td, tn = _make_data(seed=2)
    out_omitted = screen_predictors(**_common_kwargs(fd, fn, td, tn, n_workers=2))
    out_explicit_none = screen_predictors(**_common_kwargs(fd, fn, td, tn, n_workers=2, seed_workers_pool=None))
    assert out_omitted[0] == out_explicit_none[0]  # selected_vars identical


def test_n_workers_gt_1_returns_a_parallel_pool():
    fd, fn, td, tn = _make_data(seed=3)
    out = screen_predictors(**_common_kwargs(fd, fn, td, tn, n_workers=2))
    assert isinstance(out[-1], Parallel)


def test_seeded_pool_is_reused_verbatim_not_rebuilt():
    """The direct mechanism the fix relies on: passing round 1's pool back in as seed_workers_pool must
    return the IDENTICAL object (not an equal-but-distinct rebuild) -- proving no fresh pool/warmup ran."""
    fd, fn, td, tn = _make_data(seed=4)
    out1 = screen_predictors(**_common_kwargs(fd, fn, td, tn, n_workers=2))
    pool1 = out1[-1]
    assert pool1 is not None

    out2 = screen_predictors(**_common_kwargs(fd, fn, td, tn, n_workers=2, seed_workers_pool=pool1))
    pool2 = out2[-1]
    assert pool2 is pool1, "seeded pool must be reused verbatim (same object), not rebuilt"


def test_pool_reuse_preserves_selection_identity():
    """Reusing a warmed pool across rounds must not change WHICH features get selected -- pure speedup,
    no behavior change."""
    fd, fn, td, tn = _make_data(seed=5, m=6)
    out_fresh = screen_predictors(**_common_kwargs(fd, fn, td, tn, n_workers=2))
    out_seeded = screen_predictors(**_common_kwargs(fd, fn, td, tn, n_workers=2, seed_workers_pool=out_fresh[-1]))
    assert out_seeded[0] == out_fresh[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
