"""Discovery-phase extreme-AR skip: fires on a group-aware-ACTIVE split with a bounded-only zoo, so a strongly-AR
target stops wasting ~discovery + composite-train wall and ships lag_predict. Distinct from the per-model neural skip.

Regression sensors for:
  * _zoo_is_bounded_only -- neural / plain-linear presence DISABLES the skip; trees + Ridge keep it eligible.
  * _extreme_ar_discovery_skip -- gates on group_aware_ACTIVE (config OR hint), not the analyzer hint alone (the TV6
    bug: a config-only group-aware run did not skip because the gate read the recommendation flag).
"""
from __future__ import annotations

from mlframe.training.core._phase_composite_discovery import (
    _zoo_is_bounded_only,
    _extreme_ar_discovery_skip,
)


def test_bounded_zoo_trees_and_ridge_only():
    assert _zoo_is_bounded_only(["lgb", "xgb", "cb"]) is True
    assert _zoo_is_bounded_only(["ridge", "lgb"]) is True


def test_unbounded_zoo_neural_or_plain_linear():
    assert _zoo_is_bounded_only(["lgb", "mlp"]) is False  # neural
    assert _zoo_is_bounded_only(["linear", "lgb"]) is False  # plain LinearRegression extrapolates
    assert _zoo_is_bounded_only([]) is False  # unknown zoo -> conservative, never auto-skip


def _decide(**kw):
    base = dict(
        skip_enabled=True, group_aware_active=True, bounded_only_zoo=True,
        lag1_ar=1.0, is_picked_target=True, threshold=0.99,
    )
    base.update(kw)
    return _extreme_ar_discovery_skip(**base)


def test_fires_on_config_only_group_aware_active():
    # The TV6 case: split group-aware via CONFIG (group_aware_active True), strong AR, bounded zoo, picked target.
    assert _decide() is True


def test_does_not_fire_without_active_group_aware():
    assert _decide(group_aware_active=False) is False


def test_does_not_fire_with_unbounded_zoo():
    assert _decide(bounded_only_zoo=False) is False


def test_does_not_fire_below_threshold_or_missing_lag1():
    assert _decide(lag1_ar=0.80) is False
    assert _decide(lag1_ar=None) is False


def test_does_not_fire_on_non_picked_target():
    assert _decide(is_picked_target=False) is False


def test_disabled_never_fires():
    assert _decide(skip_enabled=False) is False
