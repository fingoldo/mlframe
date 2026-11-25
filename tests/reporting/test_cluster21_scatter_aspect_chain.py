"""Cluster 21 regression: the scatter aspect logic has no no-op branch (audit 2026-11-20, finding 21-F23).

Source-level pin: the ``if not p.equal_aspect: pass`` branch is gone; the explicit-limits and default-limits cases are guarded by
equal_aspect directly, so with the flag off nothing is set, as before.
"""

from pathlib import Path

from mlframe.reporting.renderers import _matplotlib_scatter as ms

_SRC = Path(ms.__file__).resolve()


def test_no_empty_equal_aspect_branch_remains():
    """The no-op ``if not p.equal_aspect: pass`` branch is removed (21-F23)."""
    text = _SRC.read_text(encoding="utf-8")
    assert "if not p.equal_aspect:" not in text


def test_aspect_is_guarded_by_equal_aspect():
    """Both aspect cases are gated on equal_aspect, so the flag off sets nothing (21-F23)."""
    text = _SRC.read_text(encoding="utf-8")
    assert "if p.equal_aspect and (p.xlim is not None or p.ylim is not None):" in text
    assert "        elif p.equal_aspect:\n" in text
