"""Cluster 15 regression: the superseded nested _is_discrete helper is gone and the live _is_discrete_v2 is kept (audit 2026-11-20, finding 15-F12).

Source-level pin: the dead nested definition is removed from the permutation-importance module, and the superseding helper it pointed at
remains in the module.
"""

from pathlib import Path

from mlframe.feature_selection.wrappers import _helpers_importance as hi

_SRC = Path(hi.__file__).resolve()


def test_dead_nested_is_discrete_is_removed():
    """No nested _is_discrete definition remains (15-F12)."""
    text = _SRC.read_text(encoding="utf-8")
    assert "def _is_discrete(col" not in text


def test_superseding_helper_is_still_used():
    """_is_discrete_v2 is still defined and called (15-F12)."""
    text = _SRC.read_text(encoding="utf-8")
    assert "_is_discrete_v2" in text
