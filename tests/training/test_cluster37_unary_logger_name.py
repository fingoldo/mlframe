"""Cluster 37 regression: the unary transforms log under the composite-transforms logger name (audit 2026-11-20, finding 37-F36)."""

from mlframe.training.composite.transforms import unary


def test_unary_uses_the_shared_composite_logger_name():
    """The module logger carries the same name as the other composite transform modules, so one log filter catches them all."""
    assert unary.logger.name == "mlframe.training.composite_transforms"
