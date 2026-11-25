"""Cluster 13 regression: the core-refine guard is an explicit check, not an assert that python -O strips (audit 2026-11-20, finding 13-F24).

Source-level pin: the module has no ``assert _base_honest_loss_for_auto is not None`` and raises RuntimeError at those points instead.
"""

from pathlib import Path

from mlframe.feature_selection import shap_proxied_fs as pkg

_STEPS = Path(pkg.__file__).resolve().parent / "_shap_proxied_fit_steps.py"


def test_base_honest_loss_guard_is_not_an_assert():
    """The None guard raises RuntimeError rather than relying on assert (13-F24)."""
    text = _STEPS.read_text(encoding="utf-8")
    assert "assert _base_honest_loss_for_auto is not None" not in text
    assert text.count("if _base_honest_loss_for_auto is None:") == 2
    assert "raise RuntimeError('base honest loss missing for core refine" in text
