"""Cluster 13 regression: the shap-proxied package does not export the pandas module, and the prescreen resolver imports once (audit 2026-11-20, findings 13-F23 and 13-F25).

Pinned: "pd" is not in __all__ (so a star import does not pull in pandas), and the hardcoded-ladder branch no longer re-imports
compute_phi_rank_stability inside the function that already imports it.
"""

from pathlib import Path

from mlframe.feature_selection import shap_proxied_fs as pkg

_FIT = Path(pkg.__file__).resolve().parent / "_shap_proxied_fit.py"


def test_pd_is_not_exported():
    """The package's __all__ does not name the pandas module (13-F25)."""
    assert "pd" not in pkg.__all__


def test_compute_phi_rank_stability_imported_once_per_function():
    """The hardcoded-ladder branch does not re-import compute_phi_rank_stability (13-F23)."""
    text = _FIT.read_text(encoding="utf-8")
    assert text.count("import compute_phi_rank_stability") == 1
