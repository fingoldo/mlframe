"""Regression: the adaptive-Fourier detector row cap is thread-local, not a process-global os.environ write.

MRMR.fit shrinks the detector's row cap to the fast-search subsample for the fit's duration. It used to do this
by writing os.environ["MLFRAME_FOURIER_DETECT_MAX_N"], which two concurrent fits raced (one fit's restore could
reset another's value), making the detector's sample size non-deterministic across concurrent fits. The cap is now
carried in a thread-local, so each fit's shrink is isolated to its own thread and the env var is only the
cross-process default.
"""
from __future__ import annotations

import os
import threading
import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._fourier_detect_cap import (
    clear_fourier_detect_cap,
    get_fourier_detect_max_n,
    peek_fourier_detect_cap,
    set_fourier_detect_cap,
)
from mlframe.feature_selection.filters.mrmr import MRMR


@pytest.fixture(autouse=True)
def _clear_cap():
    clear_fourier_detect_cap()
    yield
    clear_fourier_detect_cap()


def test_env_fallback_and_default(monkeypatch):
    clear_fourier_detect_cap()
    monkeypatch.delenv("MLFRAME_FOURIER_DETECT_MAX_N", raising=False)
    assert get_fourier_detect_max_n() == 200_000
    monkeypatch.setenv("MLFRAME_FOURIER_DETECT_MAX_N", "50000")
    assert get_fourier_detect_max_n() == 50_000
    monkeypatch.setenv("MLFRAME_FOURIER_DETECT_MAX_N", "0")
    assert get_fourier_detect_max_n() == 0  # 0 disables the cap


def test_thread_local_set_takes_precedence_over_env(monkeypatch):
    monkeypatch.setenv("MLFRAME_FOURIER_DETECT_MAX_N", "200000")
    set_fourier_detect_cap(1234)
    assert get_fourier_detect_max_n() == 1234
    assert peek_fourier_detect_cap() == 1234
    clear_fourier_detect_cap()
    assert peek_fourier_detect_cap() is None
    assert get_fourier_detect_max_n() == 200_000


def test_cap_is_isolated_per_thread(monkeypatch):
    monkeypatch.setenv("MLFRAME_FOURIER_DETECT_MAX_N", "200000")
    set_fourier_detect_cap(777)
    seen = {}

    def _worker():
        # A different thread must NOT see the main thread's per-fit cap; it falls back to env/default.
        seen["peek"] = peek_fourier_detect_cap()
        seen["get"] = get_fourier_detect_max_n()

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    assert seen["peek"] is None, "thread-local cap leaked across threads"
    assert seen["get"] == 200_000
    assert get_fourier_detect_max_n() == 777  # main thread still sees its own cap


def test_fit_does_not_mutate_os_environ_and_clears_cap():
    """A fit with fe_check_pairs_subsample_n set must NOT write os.environ (the old race source) and must leave no thread-local cap behind."""
    before = os.environ.get("MLFRAME_FOURIER_DETECT_MAX_N", "<unset>")
    rng = np.random.default_rng(0)
    X = pd.DataFrame({c: rng.standard_normal(300) for c in ("a", "b", "c", "d")})
    y = pd.Series(((X["a"] + X["b"]) > 0).astype(np.int64), name="targ")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        MRMR(verbose=0, random_seed=0, fe_check_pairs_subsample_n=64).fit(X, y)
    after = os.environ.get("MLFRAME_FOURIER_DETECT_MAX_N", "<unset>")
    assert after == before, "fit mutated the process-global os.environ Fourier cap"
    assert peek_fourier_detect_cap() is None, "fit left a thread-local Fourier cap set"
