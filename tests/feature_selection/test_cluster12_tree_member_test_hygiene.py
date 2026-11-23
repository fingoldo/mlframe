"""Cluster 12 regression: the hybrid tree-member test does not silence warnings for the whole session, and skips when LightGBM is absent (audit 2026-11-20, finding 12-F102).

Source-level pins: the module-level warnings.filterwarnings("ignore") is gone, and the LightGBM test uses importorskip instead of a bare import.
"""

from pathlib import Path

_TEST = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "_benchmarks" / "fs_hybrid" / "test_hybrid_tree_member.py"


def test_no_session_wide_warning_filter():
    """The module no longer ignores all warnings for the whole test session (12-F102)."""
    text = _TEST.read_text(encoding="utf-8")
    assert 'warnings.filterwarnings("ignore")' not in text


def test_lightgbm_test_skips_when_absent():
    """The LightGBM-backed test uses importorskip, so a missing optional dependency skips rather than errors (12-F102)."""
    text = _TEST.read_text(encoding="utf-8")
    assert 'pytest.importorskip("lightgbm")' in text
    assert "    import lightgbm as lgb\n" not in text
