"""Cluster 28 regression: helpers.py ends without an empty section header (audit 2026-11-20, finding 28-F68).

The "Callback Classes" header had no code under it. It is removed, and the module still imports with its parser.
"""

from pathlib import Path

from mlframe.training import helpers


def test_no_empty_callback_header_at_the_end():
    """The module does not end on the orphaned section header (28-F68)."""
    text = Path(helpers.__file__).read_text(encoding="utf-8")
    assert "Callback Classes for Training Monitoring" not in text


def test_parser_still_available():
    """Control: the module still exposes its GPU device parser (28-F68)."""
    assert callable(helpers.parse_catboost_devices)
