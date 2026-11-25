"""Cluster 25 regression: mlframe.metrics.__all__ exports public names, not module internals (audit 2026-11-20, finding 25-F68).

The list used to carry logger, logging, np, numba, annotations, core and Callable, which leak implementation details into the public namespace.
"""

import mlframe.metrics as metrics

INTERNALS = {"logger", "logging", "np", "numba", "annotations", "core", "Callable"}


def test_all_does_not_export_module_internals():
    """No implementation-detail name appears in metrics.__all__ (25-F68)."""
    assert INTERNALS.isdisjoint(set(metrics.__all__))


def test_all_still_exports_public_metric_names():
    """Control: the list is not empty and every name in it resolves on the module (25-F68)."""
    assert len(metrics.__all__) > 50
    missing = [name for name in metrics.__all__ if not hasattr(metrics, name)]
    assert missing == []
