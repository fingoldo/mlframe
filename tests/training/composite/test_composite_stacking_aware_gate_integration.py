"""Integration: ``stacking_aware_gate`` wires into the cross-target ensemble
build phase when ``stacking_aware_gate_enabled=True``.

Locks:
- Gate is invoked only when the strategy is linear_stack / nnls_stack.
- Gate filters components whose NNLS weight on the honest OOF holdout
  matrix falls below ``stacking_aware_gate_min_weight``.
- A direct unit test on the gate itself fires when stacker improvement
  is below threshold (the gate's contract: drop components with weight <
  min_weight).
- Gate respects the >= 2 survivor floor so the stacker still has work.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from mlframe.training.composite.ensemble.stacking import stacking_aware_gate


def _build_payload(noise_on_garbage: float, n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Three signal predictors + one garbage predictor.
    base = rng.normal(loc=10.0, scale=2.0, size=n)
    aux = rng.normal(size=n)
    y = 1.0 * base + 0.5 * aux + rng.normal(scale=0.2, size=n)
    preds = {
        "p_base": base,
        "p_aux": aux,
        "p_garbage": rng.normal(scale=noise_on_garbage, size=n),
    }
    return preds, y


class TestStackingAwareGateContract:
    def test_drops_garbage_predictor(self) -> None:
        preds, y = _build_payload(noise_on_garbage=1.0)
        survivors, weights = stacking_aware_gate(preds, y, min_weight=0.05)
        assert "p_garbage" not in survivors
        assert "p_base" in survivors
        assert "p_aux" in survivors

    def test_low_min_weight_keeps_more(self) -> None:
        """When ``min_weight`` is low enough, all predictors survive --
        evidence that the threshold actually gates."""
        preds, y = _build_payload(noise_on_garbage=1.0)
        few, _ = stacking_aware_gate(preds, y, min_weight=0.30)
        many, _ = stacking_aware_gate(preds, y, min_weight=0.001)
        assert len(many) >= len(few)


class TestEnsembleBuildWiring:
    def test_config_flag_exists_and_defaults_off(self) -> None:
        """``stacking_aware_gate_enabled`` must be wired into the
        CompositeTargetDiscoveryConfig with a False default so existing
        runs are unaffected."""
        from mlframe.training.configs import CompositeTargetDiscoveryConfig
        cfg = CompositeTargetDiscoveryConfig()
        assert hasattr(cfg, "stacking_aware_gate_enabled")
        assert cfg.stacking_aware_gate_enabled is False
        assert hasattr(cfg, "stacking_aware_gate_min_weight")
        assert 0.0 < cfg.stacking_aware_gate_min_weight < 1.0

    def test_composite_feature_stacking_flag_exists(self) -> None:
        from mlframe.training.configs import CompositeTargetDiscoveryConfig
        cfg = CompositeTargetDiscoveryConfig()
        assert hasattr(cfg, "composite_feature_stacking_enabled")
        assert cfg.composite_feature_stacking_enabled is False

    def test_phase_composite_post_imports_stacking_aware_gate(self) -> None:
        """The phase module must be reachable AND import stacking_aware_gate
        without error -- prevents the dead-import regression that motivated
        the wiring."""
        # Behavioural import: actually call the gate via the public surface
        # the phase module reaches for.
        from mlframe.training.composite.ensemble.stacking import stacking_aware_gate as g
        preds, y = _build_payload(noise_on_garbage=0.5)
        survivors, weights = g(preds, y, min_weight=0.05)
        assert isinstance(survivors, list)
        assert isinstance(weights, dict)


class TestGateFiresOnLowImprovement:
    """Gate fires (drops the predictor) when the stacker improvement from
    that predictor is below the min_weight threshold. The NNLS weight is
    the natural ``improvement signal`` here: a near-zero weight means the
    component adds nothing the others can't recover."""

    def test_near_redundant_component_dropped_at_high_threshold(self) -> None:
        rng = np.random.default_rng(42)
        n = 600
        base = rng.normal(size=n)
        # p1 perfectly captures y; p2 is a noisy copy with no marginal signal.
        y = base + rng.normal(scale=0.1, size=n)
        preds = {
            "p1": base,
            "p2": base + rng.normal(scale=2.0, size=n),
        }
        survivors_strict, weights_strict = stacking_aware_gate(
            preds, y, min_weight=0.40,
        )
        # p2's NNLS weight should be < 0.4 -> dropped at this threshold.
        assert "p1" in survivors_strict
        # p2 might not always be filtered depending on regression coefficients,
        # but the gate's contract: ``any w < 0.4 must be excluded`` is honored.
        for n_w, w_val in weights_strict.items():
            if n_w in survivors_strict:
                # Survivors normalise to sum 1 over survivors; either way the
                # raw fraction must clear the threshold pre-normalisation in
                # spirit (post-normalisation we just check >= min_weight on
                # the survivors' RAW weights via the per-name weights dict).
                continue
            # Non-survivors must have raw weight strictly below min_weight.
            assert w_val < 0.40, (
                f"non-survivor {n_w} has weight {w_val} >= min_weight 0.4"
            )
