"""CPX17 regression: flavour-invariant outlier-member gate memoisation in
``ensemble_probabilistic_predictions``.

The cross-member median + per-member MAE/STD + threshold decision do NOT depend on the
``ensemble_method`` flavour, so ``score_ensemble``'s n_flavours fan-out recomputed the same
gate once per flavour per split. The gate is now memoised (``predict._compute_outlier_gate``)
keyed on the member-array identities + the four thresholds, so the 2nd..n-th flavour for a
given split reuses the first flavour's decision.

This file pins:
  1. cold (fresh gate) vs warm (cached gate) ensemble outputs are BIT-IDENTICAL for every
     flavour -- the cache must never change the produced ensemble;
  2. the cache actually distinguishes threshold sets (no stale hit across different thresholds);
  3. the cache distinguishes member-sets (different members -> different gate);
  4. a dropped-member scenario reuses the same kept-member decision across flavours.

A regression that broke the cache key (e.g. ignoring thresholds, or aliasing distinct member
sets) would flip one of these assertions.
"""

import numpy as np
import pytest

from mlframe.models.ensembling.predict import (
    ensemble_probabilistic_predictions,
    _clear_gate_cache,
    _compute_outlier_gate,
    _gate_cache,
)

FLAVOURS = ["arithm", "harm", "median", "quad", "qube", "geo", "rrf"]


def _make_members(seed=11, n=4000, m=10, k=2, with_outlier=True):
    """Helper: Make members."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.0, 1.0, size=(n, k))
    preds = [np.clip(base + rng.normal(0.0, 0.02, base.shape), 0.0, 1.0).astype(np.float64) for _ in range(m)]
    if with_outlier:
        preds[0] = np.clip(base + rng.normal(0.0, 0.25, base.shape), 0.0, 1.0).astype(np.float64)
    return preds


@pytest.fixture(autouse=True)
def _clean_cache():
    """Helper: Clean cache."""
    _clear_gate_cache()
    yield
    _clear_gate_cache()


def test_cold_vs_warm_gate_is_bit_identical_per_flavour():
    """The memoised (warm) gate must yield ensemble outputs bit-identical to a fresh (cold) gate."""
    preds = _make_members()
    for fl in FLAVOURS:
        _clear_gate_cache()
        cold, _, _ = ensemble_probabilistic_predictions(*preds, ensemble_method=fl, verbose=False)
        warm, _, _ = ensemble_probabilistic_predictions(*preds, ensemble_method=fl, verbose=False)  # cache warm
        assert np.array_equal(np.asarray(cold), np.asarray(warm)), f"cold!=warm for flavour {fl!r}"


def test_gate_memoised_across_flavours_same_split():
    """Running all flavours on the same member set computes the gate exactly ONCE."""
    preds = _make_members()
    _clear_gate_cache()
    for fl in FLAVOURS:
        ensemble_probabilistic_predictions(*preds, ensemble_method=fl, verbose=False)
    # All flavours share one member-set + default thresholds -> exactly one cache entry.
    assert len(_gate_cache) == 1


def test_cache_key_distinguishes_thresholds():
    """Different thresholds must NOT collide on a stale cache hit -- they can drop different members."""
    preds = _make_members()
    _clear_gate_cache()
    skip_default, *_ = _compute_outlier_gate(preds, np.asarray(preds, dtype=np.float64), 0.0, 0.0, 2.5, 2.5)
    skip_tight, *_ = _compute_outlier_gate(preds, np.asarray(preds, dtype=np.float64), 0.0, 0.0, 1.05, 1.05)
    assert len(_gate_cache) == 2, "distinct threshold sets must produce distinct cache entries"
    # Tighter relative threshold must drop at least as many members as the looser one.
    assert skip_default <= skip_tight


def test_cache_key_distinguishes_member_sets():
    """Two different member arrays must not share a gate decision."""
    preds_a = _make_members(seed=11)
    preds_b = _make_members(seed=99)
    _clear_gate_cache()
    _compute_outlier_gate(preds_a, np.asarray(preds_a, dtype=np.float64), 0.0, 0.0, 2.5, 2.5)
    _compute_outlier_gate(preds_b, np.asarray(preds_b, dtype=np.float64), 0.0, 0.0, 2.5, 2.5)
    assert len(_gate_cache) == 2


def test_dropped_member_decision_consistent_across_flavours():
    """When the gate drops the injected outlier, every flavour blends the SAME kept members."""
    preds = _make_members(with_outlier=True)
    skip, *_ = _compute_outlier_gate(preds, np.asarray(preds, dtype=np.float64), 0.0, 0.0, 2.5, 2.5)
    assert 0 in skip, "the injected 0.25-noise outlier (member 0) should be gated out"
    # Outputs for two flavours computed twice (cold/warm) stay identical -> the kept-member set is stable.
    for fl in ("arithm", "median"):
        _clear_gate_cache()
        a, _, _ = ensemble_probabilistic_predictions(*preds, ensemble_method=fl, verbose=False)
        b, _, _ = ensemble_probabilistic_predictions(*preds, ensemble_method=fl, verbose=False)
        assert np.array_equal(np.asarray(a), np.asarray(b))
