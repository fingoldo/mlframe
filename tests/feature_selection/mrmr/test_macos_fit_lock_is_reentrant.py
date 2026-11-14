"""A fit nested in another fit on the same thread must not wait for the macOS numba fit lock that its parent holds.

On macOS fit() bodies are serialised by a lock. The target-mix second pass fits a child estimator from inside the residual rescue of the parent, on the parent's thread; with a non-reentrant lock the
child waited for the lock its own parent held, and every macOS fit that reached that stage hung until the job cap (a stack taken from a stuck shard stood on the lock acquisition in the child's fit).
"""

from __future__ import annotations

import sys
import threading

import numpy as np
import pandas as pd


def _small_frame():
    """A tiny frame and target for a fit that finishes in seconds."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.standard_normal(200), "b": rng.standard_normal(200)})
    y = pd.Series((X["a"] > 0).astype(int), name="y")
    return X, y


def test_a_fit_nested_in_a_fit_on_the_same_thread_does_not_deadlock_on_the_macos_lock(monkeypatch):
    """With the lock the code creates on macOS in place, an inner fit started from the outer fit's body returns instead of waiting for its parent."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from mlframe.feature_selection.filters.mrmr import _mrmr_class as module

    with monkeypatch.context() as patched:
        patched.setattr(sys, "platform", "darwin")
        lock = module._macos_numba_parallel_fit_lock()
    assert lock is not None
    monkeypatch.setattr(module, "_MACOS_NUMBA_PARALLEL_FIT_LOCK", lock)

    original_body = MRMR._fit_body
    state = {"inner_done": False}

    def body_that_fits_a_child_first(self, X, y, **kwargs):
        """The outer fit's body: fit another estimator (as the target-mix pass does), then carry on with the real body."""
        if not state["inner_done"]:
            state["inner_done"] = True
            MRMR(verbose=0, fe_max_steps=0).fit(X, y)
        return original_body(self, X, y, **kwargs)

    monkeypatch.setattr(MRMR, "_fit_body", body_that_fits_a_child_first)
    X, y = _small_frame()
    outcome: dict = {}

    def run() -> None:
        """Run the outer fit and record that it came back."""
        MRMR(verbose=0, fe_max_steps=0).fit(X, y)
        outcome["returned"] = True

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=180)
    assert not worker.is_alive(), "the nested fit is waiting for the fit lock its parent holds"
    assert outcome.get("returned") is True
    assert state["inner_done"]
