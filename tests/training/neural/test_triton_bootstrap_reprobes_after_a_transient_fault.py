"""An unexpected Triton-bootstrap failure must not disable Triton for the whole process on one fault.

`ensure_triton_loaded` caches its verdict in `_triton_loaded`, and `is_triton_available` just delegates, so
whatever the bootstrap concludes is final. Two outcomes were reaching that cache: "no candidate
libtriton.pyd was found", which is genuinely permanent, and "something raised while looking" -- a file lock
on the .pyd from a concurrent install, a transient OSError walking a network-mounted site-packages -- which
is not. Latching the second pinned every Triton-dependent neural path to its eager fallback for the rest of
the process, the same shape as the `_select_mi_backend` regression this repo documents.
"""

from __future__ import annotations

import sys

import pytest

from mlframe.training.neural import _triton_bootstrap as tb


@pytest.fixture(autouse=True)
def _clean_bootstrap_state():
    """Each test starts unprobed and leaves the module as it found it."""
    prior_loaded, prior_retries = tb._triton_loaded, tb._bootstrap_retries
    tb._triton_loaded = None
    tb._bootstrap_retries = 0
    yield
    tb._triton_loaded = prior_loaded
    tb._bootstrap_retries = prior_retries


@pytest.fixture
def on_windows(monkeypatch):
    """Reach the bootstrap at all: it is a no-op on every other platform."""
    monkeypatch.setattr(sys, "platform", "win32")
    # ...and Triton must not already be importable, or the function returns before the hunt.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _no_triton(name, *args, **kwargs):
        """Pretend Triton is not installed so the WinDLL hunt is reached."""
        if name == "triton":
            raise ImportError("synthetic: triton not importable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _no_triton)


def test_a_transient_fault_is_re_probed_rather_than_latched(on_windows, monkeypatch):
    """The bug: one momentary OSError disabled Triton for the process."""
    attempts = []

    def _flaky_sitepackages():
        """Raise on the first call, then behave."""
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("synthetic transient fault reading site-packages")
        return []

    monkeypatch.setattr("site.getsitepackages", _flaky_sitepackages)

    assert tb.ensure_triton_loaded() is False, "the faulting call should report unavailable"
    assert tb._triton_loaded is None, "the transient fault was cached; the next call will not re-probe"

    assert tb.ensure_triton_loaded() is False  # no .pyd found this time -- the permanent verdict
    assert len(attempts) == 2, f"the bootstrap re-probed {len(attempts) - 1} times after a transient fault, expected 1"
    assert tb._triton_loaded is False, "the genuinely-permanent 'no candidate found' verdict must still be cached"


def test_repeated_faults_stop_re_probing(on_windows, monkeypatch):
    """A genuinely broken install must not pay the WinDLL hunt on every call forever."""
    attempts = []

    def _always_failing():
        """Every attempt raises."""
        attempts.append(1)
        raise OSError("synthetic persistent fault")

    monkeypatch.setattr("site.getsitepackages", _always_failing)

    for _ in range(10):
        assert tb.ensure_triton_loaded() is False
    assert len(attempts) == tb._MAX_BOOTSTRAP_RETRIES + 1, (
        f"the bootstrap ran {len(attempts)} times against a budget of {tb._MAX_BOOTSTRAP_RETRIES}; " "a persistently broken install is not being latched"
    )
    assert tb._triton_loaded is False, "the budget was spent without latching the fallback"


def test_the_permanent_verdict_is_still_cached_on_the_first_call(on_windows, monkeypatch):
    """ "No candidate .pyd anywhere" is permanent and must not be re-probed at all."""
    attempts = []

    def _empty():
        """A clean host with no Triton installed."""
        attempts.append(1)
        return []

    monkeypatch.setattr("site.getsitepackages", _empty)

    for _ in range(5):
        assert tb.ensure_triton_loaded() is False
    assert len(attempts) == 1, f"the permanent 'not installed' verdict was re-probed {len(attempts)} times"
