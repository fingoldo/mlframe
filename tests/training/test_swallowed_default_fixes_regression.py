"""Regression tests for two handlers that used to hide their input: a falsy chained excepthook and a frame without ``shape``."""

from __future__ import annotations

import logging
import types

from mlframe.training import crash_diagnostics
from mlframe.training.composite import cache


class _FalsyHook:
    """A callable whose truth value is False, like an empty container that also defines ``__call__``."""

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.calls: list = []

    def __bool__(self) -> bool:
        """Falsy on purpose: ``prev or default`` would discard this hook."""
        return False

    def __call__(self, args) -> None:
        """Record the arguments the chain handed over."""
        self.calls.append(args)


def test_a_falsy_chained_thread_hook_is_still_called() -> None:
    """``_prev`` is chosen by ``is not None``, so a falsy but valid hook is chained instead of replaced by the default."""
    hook = _FalsyHook()
    args = types.SimpleNamespace(exc_type=SystemExit, exc_value=None, exc_traceback=None, thread=None)
    crash_diagnostics._thread_excepthook(args, _prev=hook)
    assert hook.calls == [args]


def test_a_frame_without_the_attribute_warns_and_keeps_the_placeholder(caplog) -> None:
    """A frame type that does not expose ``shape`` gives the ``?`` placeholder and an audible warning naming the attribute."""
    with caplog.at_level(logging.WARNING, logger=cache.logger.name):
        part = cache._structure_part(object(), "shape")
    assert part == "?"
    assert any("shape" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
