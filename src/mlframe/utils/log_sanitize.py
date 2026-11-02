"""Escape control characters that data-derived values carry into mlframe's log records.

Column names, target names, category values and the text of third-party exceptions reach log calls through ``%s`` arguments. A name holding a newline can
start a fake log line (a timestamp, a level, a GitHub Actions ``::error::`` command) and an escape sequence can hide a real one. Rather than editing the
hundreds of call sites, the record factory below rewrites the ARGUMENTS of records created by loggers named ``mlframe*``; the format string, which is
code, is left alone.

Rules, chosen so deliberate multi-line output survives:

* every control character except tab and newline (carriage return, escape, the C0 and C1 ranges) is replaced by its ``\\xNN`` form;
* a newline is replaced by ``\\n`` only in an argument that looks like a name or a short message: at most 3 newlines and 512 characters. Longer, deliberately
  formatted blocks (a classification report) keep their line breaks.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any

__all__ = ["escape_log_text", "install_log_sanitizer", "uninstall_log_sanitizer"]

_CONTROL = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_MAX_NEWLINES = 3
_MAX_LENGTH = 512
_PREFIX = "mlframe"
_MARK = "_mlframe_log_sanitizer"

_original_factory: Any = None


def _hex(match: "re.Match[str]") -> str:
    """The ``\\xNN`` spelling of one control character."""
    return "\\x%02x" % ord(match.group(0))


def escape_log_text(text: str) -> str:
    """``text`` with control characters escaped, and newlines too when it looks like a name or a short message."""
    if _CONTROL.search(text):
        text = _CONTROL.sub(_hex, text)
    if "\n" in text and len(text) <= _MAX_LENGTH and text.count("\n") <= _MAX_NEWLINES:
        text = text.replace("\n", "\\n")
    return text


def _sanitize_one(arg: Any) -> Any:
    """``arg`` itself when it needs nothing, else an escaped string (for an exception, the escaped text of its message)."""
    if isinstance(arg, str):
        clean = escape_log_text(arg)
        return arg if clean == arg else clean
    if isinstance(arg, BaseException):
        text = str(arg)
        clean = escape_log_text(text)
        return arg if clean == text else clean
    return arg


def _sanitize_args(args: Any) -> Any:
    """The record's ``args`` (a tuple, or a single mapping) with every string-like value escaped; the same object when nothing changed."""
    if isinstance(args, tuple):
        new = tuple(_sanitize_one(a) for a in args)
        return args if all(n is o for n, o in zip(new, args)) else new
    if isinstance(args, Mapping):
        new_map = {k: _sanitize_one(v) for k, v in args.items()}
        return args if all(new_map[k] is v for k, v in args.items()) else new_map
    return args


def install_log_sanitizer() -> bool:
    """Wrap the logging record factory so records from ``mlframe*`` loggers get escaped arguments; True when this call installed it."""
    global _original_factory
    current = logging.getLogRecordFactory()
    if getattr(current, _MARK, False):
        return False
    _original_factory = current

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        """Build the record with the previous factory, then escape its arguments when the logger is one of ours."""
        record = _original_factory(*args, **kwargs)
        if record.args and record.name.startswith(_PREFIX):
            record.args = _sanitize_args(record.args)
        return record

    setattr(factory, _MARK, True)
    logging.setLogRecordFactory(factory)
    return True


def uninstall_log_sanitizer() -> bool:
    """Put the previous record factory back; True when ours was installed (used by tests)."""
    global _original_factory
    if not getattr(logging.getLogRecordFactory(), _MARK, False) or _original_factory is None:
        return False
    logging.setLogRecordFactory(_original_factory)
    _original_factory = None
    return True
