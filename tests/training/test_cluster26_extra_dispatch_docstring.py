"""Cluster 26 regression: the extra dispatch module's docstring describes this module (audit 2026-11-20, finding 26-F38).

The docstring was copied from ``diagnostics_dispatch`` and described error-analysis and drift wiring. It now describes the model-comparison,
decile, model-card, stability, split and target-overlay renderers this module contains.
"""

from mlframe.reporting import _diagnostics_dispatch_extra as extra


def test_docstring_describes_this_modules_renderers():
    """The docstring names this module's renderers, not the error-analysis and drift copy (26-F38)."""
    doc = extra.__doc__ or ""
    assert "model comparison" in doc
    assert "decile" in doc
    assert "error-analysis + drift" not in doc
    assert "error-analysis" not in doc
