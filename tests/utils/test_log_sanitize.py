"""Log arguments from mlframe loggers cannot forge or hide log lines (tracker 2026-11-02, LOGF6-1 to LOGF6-6)."""

import logging

import pytest

import mlframe  # noqa: F401 - importing the package installs the record factory
from mlframe.utils.log_sanitize import escape_log_text, install_log_sanitizer


@pytest.fixture
def messages():
    """Capture formatted messages of one mlframe logger and one foreign logger."""
    captured = {"mlframe": [], "other": []}

    class _Handler(logging.Handler):
        """Collect ``getMessage()`` under a key."""

        def __init__(self, key):
            """Remember where to store."""
            super().__init__(level=logging.DEBUG)
            self.key = key

        def emit(self, record):
            """Store the formatted message."""
            captured[self.key].append(record.getMessage())

    ours = logging.getLogger("mlframe.test_log_sanitize")
    foreign = logging.getLogger("not_mlframe_test_log_sanitize")
    handlers = [(ours, _Handler("mlframe")), (foreign, _Handler("other"))]
    for lg, h in handlers:
        lg.addHandler(h)
        lg.setLevel(logging.DEBUG)
        lg.propagate = False
    yield captured, ours, foreign
    for lg, h in handlers:
        lg.removeHandler(h)


def test_a_newline_in_a_column_name_cannot_start_a_fake_log_line(messages):
    """The name's newline is shown as the two characters backslash-n, so the record stays one line."""
    captured, ours, _ = messages
    ours.warning("column %s dropped", "x\n2026-11-02 12:00:00 INFO mlframe: all gates passed")
    (msg,) = captured["mlframe"]
    assert "\n" not in msg
    assert "\\n2026-11-02" in msg


def test_escape_sequences_and_carriage_returns_are_escaped(messages):
    """ESC and CR in a name are written as \\x1b and \\x0d, so they cannot erase or hide the line."""
    captured, ours, _ = messages
    ours.info("target %s", "y\x1b[2K\rfake")
    (msg,) = captured["mlframe"]
    assert "\x1b" not in msg and "\r" not in msg
    assert "\\x1b" in msg and "\\x0d" in msg


def test_exception_text_from_another_library_is_escaped(messages):
    """An exception argument that echoes a hostile value is escaped through its message text."""
    captured, ours, _ = messages
    ours.warning("fit failed: %s", KeyError("col\n::error::forged"))
    (msg,) = captured["mlframe"]
    assert "\n" not in msg


def test_clean_text_is_unchanged_and_the_same_object_flows_through(messages):
    """Names without control characters, and non-string arguments, are passed exactly as given."""
    captured, ours, _ = messages
    ours.info("%s %d %r", "plain name", 3, ["a", "b"])
    assert captured["mlframe"] == ["plain name 3 ['a', 'b']"]
    assert escape_log_text("plain name") == "plain name"


def test_a_long_multi_line_report_keeps_its_line_breaks(messages):
    """A deliberately formatted block of many lines is left readable."""
    captured, ours, _ = messages
    report = "\n".join(f"class_{i}   0.9   0.8   0.85   {100 + i}" for i in range(8))
    ours.info("classification report:\n%s", report)
    (msg,) = captured["mlframe"]
    assert msg.count("\n") == 8


def test_a_mapping_argument_is_escaped_too(messages):
    """A single dict argument, used with %(name)s formatting, has its string values escaped."""
    captured, ours, _ = messages
    ours.info("column %(name)s", {"name": "a\nb"})
    assert captured["mlframe"] == ["column a\\nb"]


def test_records_from_other_loggers_are_not_touched(messages):
    """Only loggers named mlframe* are rewritten; a foreign logger's text is left as it was."""
    captured, _, foreign = messages
    foreign.warning("name %s", "a\nb")
    assert captured["other"] == ["name a\nb"]


def test_installing_twice_is_a_no_op():
    """The factory is wrapped once per process."""
    assert install_log_sanitizer() is False
