"""Consumer for the session-scoped ``trained_suite_multi_target`` fixture.

Verifies the shared multi-target fixture trains both regression targets in one fit, so per-target
inspection tests can consume it without re-fitting. Keeps the fixture exercised (a fixture no test
requests never runs).
"""

from __future__ import annotations

from mlframe.training.configs import TargetTypes


def test_multi_target_fixture_has_two_regression_targets(trained_suite_multi_target):
    """Multi target fixture has two regression targets."""
    models, metadata = trained_suite_multi_target
    assert isinstance(models, dict) and isinstance(metadata, dict)

    reg_targets = models.get(TargetTypes.REGRESSION, {})
    assert set(reg_targets) >= {"target", "target_extra"}, f"expected both regression targets trained in one fit; got {list(reg_targets)}"
    # Each target has at least one fitted model namespace with a usable estimator.
    for tname, ns_list in reg_targets.items():
        assert ns_list, f"target {tname!r} has no fitted models"
        assert any(getattr(ns, "model", None) is not None for ns in ns_list), f"target {tname!r} has no estimator on any namespace"
