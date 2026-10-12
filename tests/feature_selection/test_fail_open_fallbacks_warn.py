"""A fallback taken after a failed probe or fit is logged at WARNING, not DEBUG.

Each test forces the failure on the path that the fail-open gate flags, and asserts the substitution is visible at
WARNING (the level the gate requires). At DEBUG the same record is dropped by the default logging configuration, so
the substitution is silent; these tests fail on that version.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
import pytest


class _Stop(Exception):
    """Raised by a stub after the fallback has been logged, to end a long function early."""


def _warnings_mentioning(caplog, needle: str) -> list[logging.LogRecord]:
    """warnings mentioning stub or test: see module docstring."""
    return [r for r in caplog.records if r.levelno >= logging.WARNING and needle in r.getMessage()]


def test_group_time_series_split_warns_when_pandas_unique_fails(monkeypatch, caplog):
    """test group time series split warns when pandas unique fails."""
    import pandas

    from mlframe.feature_selection._cv_splitters import GroupTimeSeriesSplit

    def _boom(*_a, **_k):
        """boom."""
        raise RuntimeError("pandas unique broken")

    monkeypatch.setattr(pandas, "unique", _boom)
    groups = np.array(["b", "a", "b", "c"])
    with caplog.at_level(logging.DEBUG):
        out = GroupTimeSeriesSplit._ordered_unique(groups)
    # The fallback keeps first-appearance order (its own np.sort over return_index), so the result matches pandas.
    assert list(out) == ["b", "a", "c"]
    assert _warnings_mentioning(caplog, "pandas unique() group-order extraction failed")


def test_ace_pfi_split_warns_when_stratified_split_fails(monkeypatch, caplog):
    """test ace pfi split warns when stratified split fails."""
    import sklearn.model_selection as skms

    from mlframe.feature_selection import ace

    def _boom(*_a, **_k):
        """boom."""
        raise ValueError("train_test_split broken")

    monkeypatch.setattr(skms, "train_test_split", _boom)
    n = 60
    y = np.arange(n, dtype=float)
    with caplog.at_level(logging.DEBUG):
        fit_idx, score_idx = ace._pfi_split(n, y, np.random.default_rng(0), cv_policy=None, split_seed=0)
    assert len(fit_idx) + len(score_idx) == n
    assert _warnings_mentioning(caplog, "ACE held-out PFI split failed")


class _FitRaises:
    """FitRaises stub or test: see module docstring."""
    def fit(self, X, y=None, **_kw):
        """fit."""
        raise RuntimeError("selector cannot fit")


class _NoSupport:
    """NoSupport stub or test: see module docstring."""
    def fit(self, X, y=None, **_kw):
        """fit."""
        return self


def test_compare_selectors_warns_when_a_selector_fit_fails(caplog):
    """test compare selectors warns when a selector fit fails."""
    from mlframe.feature_selection.compare_selectors import compare_selectors

    X = pd.DataFrame(np.random.default_rng(0).normal(size=(40, 3)), columns=["a", "b", "c"])
    y = np.arange(40) % 2
    with caplog.at_level(logging.DEBUG):
        result = compare_selectors(X, y, selectors={"broken": _FitRaises()})
    assert "broken" in result.skipped
    assert _warnings_mentioning(caplog, "selector 'broken' fit failed")


def test_compare_selectors_warns_when_a_fitted_selector_has_no_support(caplog):
    """test compare selectors warns when a fitted selector has no support."""
    from mlframe.feature_selection.compare_selectors import compare_selectors

    X = pd.DataFrame(np.random.default_rng(0).normal(size=(40, 3)), columns=["a", "b", "c"])
    y = np.arange(40) % 2
    with caplog.at_level(logging.DEBUG):
        result = compare_selectors(X, y, selectors={"blank": _NoSupport()})
    assert "blank" in result.skipped
    assert _warnings_mentioning(caplog, "selector 'blank' has no readable support")


def test_cuda_shared_mem_per_block_warns_when_device_query_fails(monkeypatch, caplog):
    """test cuda shared mem per block warns when device query fails."""
    from mlframe.feature_selection.filters import _batch_mi_noise_gate_kernels as mod

    def _boom(*_a, **_k):
        """boom."""
        raise RuntimeError("no device")

    monkeypatch.setattr(mod, "_CUDA_SHARED_BYTES_PER_BLOCK", -1)
    monkeypatch.setattr(mod._nb_cuda, "get_current_device", _boom)
    with caplog.at_level(logging.DEBUG):
        assert mod._cuda_shared_mem_per_block() == 0
    assert _warnings_mentioning(caplog, "querying MAX_SHARED_MEMORY_PER_BLOCK failed")


def test_noise_gate_sweep_warns_when_psutil_is_unavailable(monkeypatch, caplog):
    """test noise gate sweep warns when psutil is unavailable."""
    import pyutilz.dev.benchmarking as bench

    from mlframe.feature_selection.filters import _batch_mi_noise_gate_tuning as mod

    def _stop(*_a, **_k):
        """stop stub or test: see module docstring."""
        raise _Stop()

    monkeypatch.setitem(sys.modules, "psutil", None)  # makes `import psutil` raise ImportError
    monkeypatch.setattr(bench, "sweep_backend_grid", _stop)
    with caplog.at_level(logging.DEBUG), pytest.raises(_Stop):
        mod._run_batch_mi_noise_gate_sweep()
    assert _warnings_mentioning(caplog, "psutil.virtual_memory() probe failed")


def test_hist_kernel_shared_budget_warns_when_device_query_fails(monkeypatch, caplog):
    """test hist kernel shared budget warns when device query fails."""
    from mlframe.feature_selection.filters import _batch_pair_mi_cuda_kernels as mod

    def _boom(*_a, **_k):
        """boom."""
        raise RuntimeError("no device")

    monkeypatch.setattr(mod._nb_cuda, "get_current_device", _boom)
    with caplog.at_level(logging.DEBUG):
        needed = mod._hist_kernel_shared_fits_budget(4, 2)
    assert needed == (4 * 2 + 4) * 4
    assert _warnings_mentioning(caplog, "querying MAX_SHARED_MEMORY_PER_BLOCK failed")


def test_row_chunked_pair_mi_warns_when_cupy_memory_probe_fails(monkeypatch, caplog):
    """test row chunked pair mi warns when cupy memory probe fails."""
    from mlframe.feature_selection.filters import _batch_pair_mi_cuda_kernels as mod

    monkeypatch.setitem(sys.modules, "cupy", None)  # makes `import cupy` raise ImportError

    def _stop(*_a, **_k):
        """stop stub or test: see module docstring."""
        raise _Stop()

    monkeypatch.setattr(mod, "_choose_row_chunk_rows", _stop)
    factors = np.zeros((4, 8), dtype=np.int32)
    pairs = np.zeros(2, dtype=np.int64)
    nbins = np.full(4, 2, dtype=np.int32)
    classes_y = np.zeros(8, dtype=np.int32)
    freqs_y = np.array([0.5, 0.5])
    with caplog.at_level(logging.DEBUG), pytest.raises(_Stop):
        mod.batch_pair_mi_cuda_row_chunked(factors, pairs, pairs, nbins, classes_y, freqs_y)
    assert _warnings_mentioning(caplog, "cp.cuda.runtime.memGetInfo() probe failed")
