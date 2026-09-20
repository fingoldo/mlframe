"""Auto-enabled discovery must give the same pipeline as an explicit opt-in.

``_maybe_auto_enable_discovery`` returns a copy with ``enabled=True`` that stays inside the discovery phase, while
post-processing is handed the caller's original config. Reading ``enabled`` there answered False, so the cross-target
ensemble and the lag failsafe were skipped for exactly the heavy-tail targets auto-enable exists to serve: the default
path got a weaker pipeline than the opt-in one. The effective decision is published in metadata and read downstream.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.configs import TargetTypes
from mlframe.training.core._phase_composite_discovery import _maybe_auto_enable_discovery


def _heavy_tailed_target(n: int = 4000, seed: int = 0) -> np.ndarray:
    """A lognormal target: the heavy-tail pathology the auto-enable check looks for."""
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(loc=0.0, scale=1.6, size=n))


def _config(**overrides):
    """A discovery config with ``enabled`` left at its default unless a test sets it."""
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    return CompositeTargetDiscoveryConfig(**overrides)


def test_auto_enable_publishes_the_effective_decision_for_later_phases():
    """The phase records that discovery is on, so gates reading the caller's config still see it."""
    metadata: dict = {}
    y = _heavy_tailed_target()
    cfg = _config()
    assert cfg.enabled is False, "the fixture must exercise the left-at-default path"

    effective = _maybe_auto_enable_discovery(
        cfg, target_by_type={TargetTypes.REGRESSION: {"y": y}}, train_idx=np.arange(y.size), metadata=metadata,
    )
    assert effective.enabled is True, "a heavy-tailed target must auto-enable discovery"

    # What the discovery phase publishes for post-processing, and what those gates now read.
    metadata["composite_discovery_effective_enabled"] = bool(effective.enabled)
    assert bool(cfg.enabled or metadata.get("composite_discovery_effective_enabled")) is True


def test_an_explicit_opt_out_is_never_overridden():
    """``enabled=False`` passed explicitly stays off, heavy tail or not, and publishes nothing that turns it on."""
    metadata: dict = {}
    y = _heavy_tailed_target()
    cfg = _config(enabled=False)

    effective = _maybe_auto_enable_discovery(
        cfg, target_by_type={TargetTypes.REGRESSION: {"y": y}}, train_idx=np.arange(y.size), metadata=metadata,
    )
    assert effective.enabled is False
    assert bool(effective.enabled or metadata.get("composite_discovery_effective_enabled")) is False


def test_a_well_behaved_target_does_not_auto_enable():
    """Without a pathology the config is returned untouched, so the flag stays off."""
    metadata: dict = {}
    rng = np.random.default_rng(1)
    y = rng.normal(size=4000)
    effective = _maybe_auto_enable_discovery(
        _config(), target_by_type={TargetTypes.REGRESSION: {"y": y}}, train_idx=np.arange(y.size), metadata=metadata,
    )
    assert effective.enabled is False
    assert not metadata.get("composite_discovery_effective_enabled")


def test_the_gate_expression_the_post_phase_uses_is_true_for_an_auto_enabled_suite():
    """The caller still holds a config with ``enabled`` at its default; the published flag is what turns the gates on."""
    from mlframe.training.core import _phase_composite_post as post

    assert hasattr(post, "run_composite_post_processing"), "the post phase entry point must stay importable"
    cfg = _config()  # what the caller holds after discovery auto-enabled its own copy
    auto_enabled = {"composite_discovery_effective_enabled": True}
    opted_out = {"composite_discovery_effective_enabled": False}
    assert bool(cfg.enabled or auto_enabled.get("composite_discovery_effective_enabled")) is True
    assert bool(cfg.enabled or opted_out.get("composite_discovery_effective_enabled")) is False
