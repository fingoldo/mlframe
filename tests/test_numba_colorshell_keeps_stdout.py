"""numba's coloured-error shell must not move ``sys.stdout`` under the test session.

``numba.core.errors.ColorShell.__exit__`` calls colorama's ``deinit()``, which assigns ``sys.stdout = orig_stdout`` once any
``colorama.init()`` has recorded a stream. Inside a doctest (which swaps in its own capture) that sent every later example's
output to the console, so ``test_package_doctests_pass`` failed with "Expected: True / Got nothing" depending on test order.
``tests/conftest.py`` neutralises numba's init/reinit/deinit for the session.
"""

from __future__ import annotations

import io
import sys

import pytest


def test_numba_colorshell_exit_leaves_a_swapped_stdout_in_place(monkeypatch):
    """Entering and leaving numba's ColorShell keeps a swapped-in ``sys.stdout`` installed and writes nothing to colorama's recorded stream."""
    errors = pytest.importorskip("numba.core.errors")
    initialise = pytest.importorskip("colorama.initialise")
    if not hasattr(errors, "ColorShell"):
        pytest.skip("numba built without colorama support: no ColorShell")
    # What an earlier colorama.init() leaves behind: a recorded "original" stream deinit() would restore.
    recorded = io.StringIO()
    monkeypatch.setattr(initialise, "orig_stdout", recorded)
    capture = io.StringIO()
    # Swapped and restored by hand: a monkeypatch of sys.stdout is undone only after the conftest stream-leak guard has checked.
    real = sys.stdout
    sys.stdout = capture
    try:
        with errors.ColorShell():
            print("inside")
        print("after")
        current = sys.stdout
    finally:
        sys.stdout = real

    assert current is capture
    assert capture.getvalue() == "inside\nafter\n"
    assert recorded.getvalue() == ""
