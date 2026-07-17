"""Regression: the unary (``requires_base=False``) MI result is computed ONCE and reused bit-identically.

A unary transform ignores the base column entirely, so ``MI(T_unary, X_full)`` and the resulting ``mi_gain`` are
base-independent. ``eval_one_transform`` now memoises a unary's finished candidate on its sentinel context keyed by
``transform_name`` (see ``_eval._eval_one_transform_impl`` + ``build_unary_base_context``), so a second call for the
same unary -- the per-base fallback when the sentinel context is unavailable, or any re-dispatch -- returns the
cached result without recomputing the (per-feature, full-X) MI.

These tests pin BOTH halves of the contract:

* The memoised (cached) result is BIT-IDENTICAL to a fresh per-base recompute via ``_eval_one_transform_impl`` --
  proving the optimization changes nothing numerically.
* The per-feature, full-X MI kernel (``_mi_to_target_prebinned``) is invoked FEWER times with the memo on than a
  naive per-base re-evaluation would invoke it -- proving the redundant work is actually removed.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from mlframe.training.composite.discovery import _eval as _eval_mod
from mlframe.training.composite.discovery._eval import (
    build_unary_base_context,
    eval_one_transform,
)
from mlframe.training.composite.transforms import get_transform
from mlframe.training.configs import CompositeTargetDiscoveryConfig


class _Disc:
    """Minimal stand-in exposing the ``self.config`` + ``self._reject`` surface ``eval_one_transform`` reads."""

    def __init__(self, config):
        self.config = config

    def _reject(self, base, transform_name, mi_y, valid_frac, *, reason):
        return {"spec": None, "kept": False, "reason": reason, "base": base, "transform": transform_name}


def _make_unary_ctx(n=4000, f=8, nbins=10, seed=7):
    """Build a full-X prebinned unary sentinel context + the shared train/screen targets."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, f)).astype(np.float32)
    # Strictly-positive, right-skewed y so cbrt_y / log_y are in-domain and informative.
    y = np.abs(0.7 * x[:, 0] + 0.4 * x[:, 1]).astype(np.float64) + 0.3
    y += rng.standard_normal(n) * 0.1
    y = np.abs(y) + 0.5

    from mlframe.training.composite.discovery.screening import (
        _mi_per_feature_prebinned,
        _prebin_feature_columns,
    )

    full_prebinned = _prebin_feature_columns(x, nbins=nbins)
    per_feat_y = _mi_per_feature_prebinned(full_prebinned, y, nbins=nbins)
    sample_idx = np.arange(n)
    ctx = build_unary_base_context(
        full_x_matrix=x,
        full_x_prebinned=full_prebinned,
        per_feat_y_full=per_feat_y,
        y_screen=y,
        n_train=n,
        sample_idx=sample_idx,
        mi_aggregation="mean",
        mi_nbins=nbins,
        mi_n_neighbors=3,
        random_state=seed,
        mi_estimator="bin",
    )
    return ctx, y


@pytest.mark.parametrize("transform_name", ["cbrt_y", "log_y", "yeo_johnson_y"])
def test_unary_cached_result_bit_identical_to_per_base_recompute(transform_name):
    """The memoised cached candidate equals a fresh ``_eval_one_transform_impl`` recompute, bit-for-bit."""
    cfg = CompositeTargetDiscoveryConfig(
        mi_nbins=10,
        mi_estimator="bin",
        mi_gain_bootstrap_n=0,
        min_valid_domain_frac=0.0,
        random_state=7,
    )
    disc = _Disc(cfg)
    transform = get_transform(transform_name)
    assert not transform.requires_base

    ctx, y = _make_unary_ctx()
    base = ""
    base_contexts = {base: ctx}

    # First call populates the memo and returns the freshly-computed result.
    first = eval_one_transform(
        disc,
        base,
        transform_name,
        transform,
        base_contexts=base_contexts,
        y_train=y,
        y_screen=y,
        target_col="y",
    )
    # Second call must hit the memo (cached path).
    assert transform_name in ctx["_unary_result_memo"]
    cached = eval_one_transform(
        disc,
        base,
        transform_name,
        transform,
        base_contexts=base_contexts,
        y_train=y,
        y_screen=y,
        target_col="y",
    )

    # An INDEPENDENT per-base recompute via the impl (bypasses the memo entirely).
    recompute = _eval_mod._eval_one_transform_impl(
        disc,
        base,
        transform_name,
        transform,
        base_contexts=base_contexts,
        y_train=y,
        y_screen=y,
        target_col="y",
    )

    assert len(first) == len(cached) == len(recompute) == 1
    spec_cached = cached[0]["spec"]
    spec_recompute = recompute[0]["spec"]
    assert spec_cached is not None and spec_recompute is not None
    # Bit-identical mi_gain / mi_t / mi_y between cached and per-base recompute.
    assert spec_cached.mi_gain == spec_recompute.mi_gain
    assert spec_cached.mi_t == spec_recompute.mi_t
    assert spec_cached.mi_y == spec_recompute.mi_y
    assert cached[0]["mi_gain_lcb"] == recompute[0]["mi_gain_lcb"]
    # And the cached result equals the first (memo-populating) result too.
    assert first[0]["spec"].mi_gain == spec_cached.mi_gain


def test_unary_memo_returns_independent_copies():
    """Mutating one returned candidate's flags must not leak into a later cached read."""
    cfg = CompositeTargetDiscoveryConfig(
        mi_nbins=10,
        mi_estimator="bin",
        mi_gain_bootstrap_n=0,
        min_valid_domain_frac=0.0,
        random_state=7,
    )
    disc = _Disc(cfg)
    transform = get_transform("cbrt_y")
    ctx, y = _make_unary_ctx()
    base_contexts = {"": ctx}

    r1 = eval_one_transform(
        disc,
        "",
        "cbrt_y",
        transform,
        base_contexts=base_contexts,
        y_train=y,
        y_screen=y,
        target_col="y",
    )
    r1[0]["kept"] = True  # caller-side in-place mutation (mirrors _fit.py FDR pass).
    r2 = eval_one_transform(
        disc,
        "",
        "cbrt_y",
        transform,
        base_contexts=base_contexts,
        y_train=y,
        y_screen=y,
        target_col="y",
    )
    # The second (cached) read must be a fresh copy unaffected by the r1 mutation.
    assert r2[0]["kept"] is False


def test_unary_memo_reduces_full_x_mi_calls(monkeypatch):
    """With the memo ON, the full-X per-feature MI kernel runs ONCE across B re-evaluations, not B times."""
    cfg = CompositeTargetDiscoveryConfig(
        mi_nbins=10,
        mi_estimator="bin",
        mi_gain_bootstrap_n=0,
        min_valid_domain_frac=0.0,
        random_state=7,
    )
    disc = _Disc(cfg)
    transform = get_transform("cbrt_y")
    ctx, y = _make_unary_ctx()
    base_contexts = {"": ctx}

    calls = {"n": 0}
    real = _eval_mod._mi_to_target_prebinned

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(_eval_mod, "_mi_to_target_prebinned", _counting)

    n_bases = 3
    for _ in range(n_bases):
        eval_one_transform(
            disc,
            "",
            "cbrt_y",
            transform,
            base_contexts=base_contexts,
            y_train=y,
            y_screen=y,
            target_col="y",
        )
    # Only the FIRST call computed MI(T_unary, X_full); the next (n_bases-1) hit the memo.
    # The single computing call invokes _mi_to_target_prebinned exactly once (mi_t; mi_y_compare
    # is unused here because valid_screen is full).
    assert calls["n"] == 1, f"expected one full-X MI call, got {calls['n']}"
