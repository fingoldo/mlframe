"""mlframe configures its own logger; an embedding application's root logging must survive untouched."""

import logging
import sys

import pytest

from mlframe.training.core._misc_helpers import _ensure_logging_visible


@pytest.fixture(autouse=True)
def _restore_logging():
    root, pkg = logging.getLogger(), logging.getLogger("mlframe")
    saved = (list(root.handlers), root.level, list(pkg.handlers), pkg.level, pkg.propagate)
    yield
    root.handlers, root.level = saved[0], saved[1]
    pkg.handlers, pkg.level, pkg.propagate = saved[2], saved[3], saved[4]


def test_an_applications_root_handler_keeps_its_formatter_and_level():
    root, pkg = logging.getLogger(), logging.getLogger("mlframe")
    root.handlers = []
    app_handler = logging.StreamHandler(stream=sys.stdout)
    app_format = logging.Formatter('{"msg": "%(message)s"}')
    app_handler.setFormatter(app_format)
    root.addHandler(app_handler)
    root.setLevel(logging.WARNING)
    pkg.handlers, pkg.level, pkg.propagate = [], logging.NOTSET, True

    _ensure_logging_visible(logging.INFO)

    assert root.handlers == [app_handler] and app_handler.formatter is app_format, "the application's JSON handler was rewritten"
    assert root.level == logging.WARNING, "the application's root level was lowered"
    assert pkg.level == logging.INFO, "mlframe's own records must still get through to the application's handlers"
    assert pkg.propagate is True, "with an application handler present, our records must keep reaching it"


def test_a_bare_process_gets_a_timestamped_handler_on_the_package_logger():
    root, pkg = logging.getLogger(), logging.getLogger("mlframe")
    root.handlers = []
    pkg.handlers, pkg.level, pkg.propagate = [], logging.NOTSET, True

    _ensure_logging_visible(logging.INFO)

    assert root.handlers == [], "nothing is installed on the root"
    assert len(pkg.handlers) == 1 and "%(asctime)" in pkg.handlers[0].formatter._fmt
    assert pkg.propagate is False, "a later basicConfig must not duplicate every line"


def test_a_second_call_does_not_add_another_handler():
    root, pkg = logging.getLogger(), logging.getLogger("mlframe")
    root.handlers = []
    pkg.handlers, pkg.level, pkg.propagate = [], logging.NOTSET, True

    _ensure_logging_visible(logging.INFO)
    first = list(pkg.handlers)
    _ensure_logging_visible(logging.INFO)
    assert pkg.handlers == first
