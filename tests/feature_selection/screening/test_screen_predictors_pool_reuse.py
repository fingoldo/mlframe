"""Regression tests for the ``seed_workers_pool``/``n_workers`` contract of ``screen_predictors``.

Originally written 2026-07-09 (MRMR audit finding #6) for a cross-round joblib worker-pool reuse
feature: at ``n_workers>1``, a ``joblib.Parallel`` pool was built once and returned so a caller could
pass it back in via ``seed_workers_pool`` on the next round instead of rebuilding/re-warming it.

That pool construction was REMOVED 2026-07-19 (see ``_screen_predictors.py``'s own comment at the
``workers_pool = None`` assignment): an isolated/warmed/best-of-3+ A/B at realistic
``evaluate_candidates`` scales (m=10 -> 0.03x, m=320 -> 0.72-0.73x, m=820/n_workers=8 -> 0.81x) found
the pool NEVER wins over the serial path -- it's GIL-bound at the joblib dispatch boundary even though
the underlying njit kernels themselves release the GIL. ``evaluate_candidates`` now always runs serial;
``n_workers``/``seed_workers_pool`` are kept as accepted-but-inert parameters (other call sites, e.g.
the Fleuret conditional-confirmation gate, still branch on ``n_workers`` for unrelated reasons).

These tests verify the CURRENT, validated contract: (a) the parameter is fully backward-compatible
(``seed_workers_pool=None`` unchanged behavior), (b) the returned pool is ALWAYS ``None`` regardless of
``n_workers``/``seed_workers_pool`` (no pool is ever built or reused -- the 2026-07-19 removal is a
permanent, validated design decision, not a bug), and (c) selection is IDENTICAL regardless of what's
passed for ``seed_workers_pool`` (the parameter is truly inert, not silently changing results).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from mlframe.feature_selection.filters.screen import screen_predictors


def _make_data(n: int = 400, m: int = 6, seed: int = 0):
    """Make data."""
    rng = np.random.default_rng(seed)
    factors_data = rng.integers(0, 4, size=(n, m)).astype(np.int32)
    y_col = ((factors_data[:, 0] + rng.integers(0, 2, size=n)) % 2).astype(np.int32).reshape(-1, 1)
    targets_data = y_col
    factors_nbins = np.array([4] * m, dtype=np.int32)
    targets_nbins = np.array([2], dtype=np.int32)
    return factors_data, factors_nbins, targets_data, targets_nbins


def _common_kwargs(factors_data, factors_nbins, targets_data, targets_nbins, **overrides):
    """Common kwargs."""
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
    """Seed workers pool default parameter is none."""
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


def test_n_workers_gt_1_still_returns_none_pool():
    """Re-framed 2026-08-01: pool construction at n_workers>1 was deliberately removed 2026-07-19
    (measured 0.03x-0.81x -- the joblib.Parallel pool never beats serial at any tested scale, see
    ``_screen_predictors.py``'s own comment at ``workers_pool = None``). The returned pool is now
    ALWAYS None, at any n_workers -- this is the validated, current contract, not a regression."""
    fd, fn, td, tn = _make_data(seed=3)
    out = screen_predictors(**_common_kwargs(fd, fn, td, tn, n_workers=2))
    assert out[-1] is None


def test_seed_workers_pool_param_is_inert_at_n_workers_gt_1():
    """Re-framed 2026-08-01: ``seed_workers_pool`` is accepted for backward compatibility but no
    longer used (no pool is ever built to reuse) -- passing ANY value (including a stale/foreign
    object) must not raise and must not change the returned pool (always None) or the selection."""
    fd, fn, td, tn = _make_data(seed=4)
    out1 = screen_predictors(**_common_kwargs(fd, fn, td, tn, n_workers=2))
    assert out1[-1] is None

    sentinel = object()  # deliberately not a real Parallel instance -- proves the param is unread
    out2 = screen_predictors(**_common_kwargs(fd, fn, td, tn, n_workers=2, seed_workers_pool=sentinel))
    assert out2[-1] is None
    assert out2[0] == out1[0], "seed_workers_pool must not influence selection -- it's inert"


def test_pool_reuse_preserves_selection_identity():
    """Reusing a warmed pool across rounds must not change WHICH features get selected -- pure speedup,
    no behavior change."""
    fd, fn, td, tn = _make_data(seed=5, m=6)
    out_fresh = screen_predictors(**_common_kwargs(fd, fn, td, tn, n_workers=2))
    out_seeded = screen_predictors(**_common_kwargs(fd, fn, td, tn, n_workers=2, seed_workers_pool=out_fresh[-1]))
    assert out_seeded[0] == out_fresh[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
