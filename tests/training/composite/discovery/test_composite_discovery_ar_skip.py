"""Discovery-phase extreme-AR skip: fires on a group-aware-ACTIVE split with a bounded-only zoo, so a strongly-AR
target stops wasting ~discovery + composite-train wall and ships lag_predict. Distinct from the per-model neural skip.

Regression sensors for:
  * _zoo_is_bounded_only -- neural / plain-linear presence DISABLES the skip; trees + Ridge keep it eligible.
  * _extreme_ar_discovery_skip -- gates on group_aware_ACTIVE (config OR hint), not the analyzer hint alone (the TV6
    bug: a config-only group-aware run did not skip because the gate read the recommendation flag).
"""

from __future__ import annotations

import numpy as np

from mlframe.training.core._phase_composite_discovery import (
    _zoo_is_bounded_only,
    _extreme_ar_discovery_skip,
    _recompute_lag1_ar_per_group,
)


def _grouped_ar_target(n_groups=20, per=300, rho=0.98, seed=0):
    """Grouped ar target."""
    rng = np.random.default_rng(seed)
    ys, gs = [], []
    for gi in range(n_groups):
        y = np.empty(per)
        y[0] = rng.normal()
        for i in range(1, per):
            y[i] = rho * y[i - 1] + rng.normal(0, 0.1)
        ys.append(y)
        gs.append(np.full(per, gi))
    return np.concatenate(ys), np.concatenate(gs)


def test_recompute_lag1_detects_strong_ar_and_ignores_noise():
    """Recompute lag1 detects strong ar and ignores noise."""
    y, gids = _grouped_ar_target(rho=0.98)
    tidx = np.arange(y.size)
    ar = _recompute_lag1_ar_per_group(y, gids, tidx)
    assert ar is not None and ar >= 0.95, f"strong per-group AR(1) must recompute high; got {ar}"
    noise = np.random.default_rng(1).normal(size=y.size)
    ar_n = _recompute_lag1_ar_per_group(noise, gids, tidx)
    assert ar_n is not None and abs(ar_n) < 0.1, f"white noise must recompute ~0; got {ar_n}"


def test_recompute_lag1_robust_to_degenerate_inputs():
    """Recompute lag1 robust to degenerate inputs."""
    y, gids = _grouped_ar_target()
    assert _recompute_lag1_ar_per_group(y, gids, np.array([], dtype=np.int64)) is None  # empty train
    assert _recompute_lag1_ar_per_group(y, gids, np.array([10**9])) is None  # out-of-range index
    assert _recompute_lag1_ar_per_group(y, gids, np.arange(10)) is None  # < 100 finite rows


def test_recomputed_lag1_makes_skip_fire_when_report_empty():
    # The TV6-9 case: report empty -> lag1_report=None, recommended=False, picked=None. With the recompute fallback
    # (target-specific lag1) + group_ids-derived group-aware + recompute-implies-picked, the skip must still fire.
    # The real TVT target is a near-deterministic per-well depth ramp (consecutive depths have near-identical TVT ->
    # lag1=0.9999). A noisy mean-reverting AR(1) caps ~0.986 even at rho~1 (the noise floor), so model the ramp.
    """Recomputed lag1 makes skip fire when report empty."""
    rng = np.random.default_rng(0)
    ys, gs = [], []
    for gi in range(20):
        ramp = np.linspace(rng.uniform(0, 10), rng.uniform(0, 10) + 50, 300)
        ys.append(ramp + rng.normal(0, 0.02, 300))
        gs.append(np.full(300, gi))
    y, gids = np.concatenate(ys), np.concatenate(gs)
    lag1_eff = _recompute_lag1_ar_per_group(y, gids, np.arange(y.size))
    assert lag1_eff is not None and lag1_eff >= 0.99, f"near-1 AR must clear threshold; got {lag1_eff}"
    fired = _extreme_ar_discovery_skip(
        skip_enabled=True,
        group_aware_active=True,  # derived from group_ids is not None
        bounded_only_zoo=True,
        lag1_ar=lag1_eff,  # recomputed, not from the (empty) report
        is_picked_target=True,  # recompute implies target-specific -> picked
        threshold=0.99,
    )
    assert fired is True, "skip must fire on recomputed strong-AR even with an empty target_distribution_report"


def test_bounded_zoo_trees_and_ridge_only():
    """Bounded zoo trees and ridge only."""
    assert _zoo_is_bounded_only(["lgb", "xgb", "cb"]) is True
    assert _zoo_is_bounded_only(["ridge", "lgb"]) is True


def test_unbounded_zoo_neural_or_plain_linear():
    """Unbounded zoo neural or plain linear."""
    assert _zoo_is_bounded_only(["lgb", "mlp"]) is False  # neural
    assert _zoo_is_bounded_only(["linear", "lgb"]) is False  # plain LinearRegression extrapolates
    assert _zoo_is_bounded_only([]) is False  # unknown zoo -> conservative, never auto-skip


def _decide(**kw):
    """Decide."""
    base = dict(
        skip_enabled=True,
        group_aware_active=True,
        bounded_only_zoo=True,
        lag1_ar=1.0,
        is_picked_target=True,
        threshold=0.99,
    )
    base.update(kw)
    return _extreme_ar_discovery_skip(**base)


def test_fires_on_config_only_group_aware_active():
    # The TV6 case: split group-aware via CONFIG (group_aware_active True), strong AR, bounded zoo, picked target.
    """Fires on config only group aware active."""
    assert _decide() is True


def test_does_not_fire_without_active_group_aware():
    """Does not fire without active group aware."""
    assert _decide(group_aware_active=False) is False


def test_does_not_fire_with_unbounded_zoo():
    """Does not fire with unbounded zoo."""
    assert _decide(bounded_only_zoo=False) is False


def test_does_not_fire_below_threshold_or_missing_lag1():
    """Does not fire below threshold or missing lag1."""
    assert _decide(lag1_ar=0.80) is False
    assert _decide(lag1_ar=None) is False


def test_does_not_fire_on_non_picked_target():
    """Does not fire on non picked target."""
    assert _decide(is_picked_target=False) is False


def test_disabled_never_fires():
    """Disabled never fires."""
    assert _decide(skip_enabled=False) is False
