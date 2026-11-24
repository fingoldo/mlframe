"""Cluster 28 regression: the train_eval module defines its logger once and documents the select_target re-export accurately (audit 2026-11-20, finding 28-F63).

Source-level pins: one logger definition, and the comment above the select_target import says it is a re-export.
"""

from pathlib import Path

from mlframe.training import train_eval as te

_SRC = Path(te.__file__).resolve()


def test_logger_is_defined_once():
    """The module assigns its logger exactly once (28-F63)."""
    assert _SRC.read_text(encoding="utf-8").count("\nlogger = logging.getLogger(__name__)\n") == 1


def test_select_target_comment_describes_the_reexport():
    """The comment states the import is a re-export for existing callers (28-F63)."""
    text = _SRC.read_text(encoding="utf-8")
    assert "this import re-exports it so existing ``train_eval`` callers keep working" in text
    assert "carved to ``_train_eval_select_target``; re-exported below." not in text
