"""Falling off the device Fourier / extra-basis paths must log at WARNING."""

from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pandas as pd
import pytest


def test_fourier_block_cuda_failure_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A device failure in the block kernel returns None and says so."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _extra_basis_fourier_cuda as mod

    def _boom(*_a, **_k):
        """Raise like a failing allocation."""
        raise RuntimeError("out of memory")

    col = types.SimpleNamespace(dtype=np.dtype(np.float32), shape=(8,))
    cp = types.SimpleNamespace(empty=_boom)
    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        got = mod.fourier_block_cuda(cp, [col], [{"lo": 0.0, "span": 1.0, "freq": 1.0, "kind": "sin"}])
    assert got is None
    assert any(r.levelno == logging.WARNING and "fourier_block_cuda unavailable" in r.getMessage() for r in caplog.records)


def test_extra_basis_resident_failure_warns(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A device failure while scoring the extra-basis columns returns None and says so."""
    import mlframe.feature_selection.filters._gpu_strict_fe as strict
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _extra_basis_resident as mod

    monkeypatch.setattr(strict, "fe_gpu_device_born_extra_basis_enabled", lambda: True)
    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    monkeypatch.setattr(mod, "extra_basis_device_chosen", lambda n: True)

    def _boom(*_a, **_k):
        """Raise like a failing launch."""
        raise RuntimeError("launch failed")

    monkeypatch.setattr(mod, "_mi_by_name_blocks", _boom)
    raw = pd.DataFrame({"x": np.arange(5.0)})
    eng = pd.DataFrame({"x__f": np.arange(5.0)})
    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        got = mod.extra_basis_eng_mi_resident(raw, eng, np.zeros(5, dtype=np.int64), {"x__f": {"basis": "fourier"}}, nbins=4)
    assert got is None
    assert any(r.levelno == logging.WARNING and "GPU path failed" in r.getMessage() for r in caplog.records)


def test_fourier_batch_device_fault_warns(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A device fault in the batch detector falls back to the per-column detector and says so."""
    import mlframe.feature_selection.filters._fe_env_switch as env_switch
    import mlframe.feature_selection.filters._gpu_strict_fe._entry as entry
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _fourier_batch_gpu as batch_gpu
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _fourier_detect_batch as mod
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_extra_basis_fe as extra

    monkeypatch.setattr(entry, "fe_gpu_strict_resident_enabled", lambda: True)
    monkeypatch.setattr(env_switch, "fused_switch_enabled", lambda _name: True)
    monkeypatch.setattr(mod, "fourier_device_chosen", lambda n: True)

    def _fault(*_a, **_k):
        """Raise a linear-algebra fault."""
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(batch_gpu, "detect_fourier_freqs_batch_gpu", _fault)
    monkeypatch.setattr(extra, "_detect_fourier_freqs_for_col", lambda z, y, **_k: [1.0])
    jobs = [(np.zeros(4), np.zeros(4), [1.0]), (np.zeros(4), np.zeros(4), [1.0])]
    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        got = mod.detect_fourier_freqs_batch(jobs, min_val_corr=0.1)
    assert got == [[1.0], [1.0]]
    assert any(r.levelno == logging.WARNING and "device fault" in r.getMessage() for r in caplog.records)
