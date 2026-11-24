"""Cluster 25 regression: the metrics package __all__ exports only its public API (audit 2026-11-20, finding 25-F68).

``__all__`` listed module internals (logger, logging, np, numba, annotations, core, Callable). They are removed; the public metric names remain,
and every name in ``__all__`` is still an attribute of the package.
"""

import mlframe.metrics as metrics

_INTERNALS = {"logger", "logging", "np", "numba", "annotations", "core", "Callable"}


def test_all_has_no_module_internals():
    """No internal module or helper name is exported (25-F68)."""
    assert not _INTERNALS & set(metrics.__all__)


def test_public_names_remain_and_resolve():
    """Every exported name still resolves on the package (25-F68)."""
    assert "fast_brier_score_loss" in metrics.__all__
    for name in metrics.__all__:
        assert hasattr(metrics, name), name
