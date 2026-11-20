"""Every CPU decline of the batched-CMI VRAM gate leaves a WARNING trail (driven on the CPU with a fake cupy)."""
import logging
import sys
import types

import pytest

from mlframe.feature_selection.filters.info_theory import _cmi_cuda
from mlframe.utils.log_throttle import reset_throttle_counts


def _fake_cupy(free, total):
    """A stand-in cupy module whose memGetInfo reports the given free/total bytes."""
    mod = types.ModuleType("cupy")
    mod.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(memGetInfo=lambda: (free, total)))
    return mod


@pytest.fixture(autouse=True)
def _fresh_throttle():
    """Forget throttle counts so each test sees its first log."""
    reset_throttle_counts()
    yield
    reset_throttle_counts()


def test_vram_cap_exceeded_logs_warning(monkeypatch, caplog):
    """Needing more than the free-VRAM cap declines CUDA and says so."""
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(1 << 30, 4 << 30))
    with caplog.at_level(logging.WARNING, logger=_cmi_cuda.logger.name):
        assert _cmi_cuda._vram_guard_passes(10_000_000, 100, 8) is False
    assert any("over the" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_vram_cushion_declined_logs_warning(monkeypatch, caplog):
    """A declined absolute-cushion check says so."""
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(8 << 30, 8 << 30))
    import mlframe.feature_selection.filters._fe_gpu_vram as vram

    monkeypatch.setattr(vram, "fe_gpu_has_vram_cushion", lambda *a, **k: False)
    with caplog.at_level(logging.WARNING, logger=_cmi_cuda.logger.name):
        assert _cmi_cuda._vram_guard_passes(1000, 10, 8) is False
    assert any("cushion" in r.getMessage() for r in caplog.records)
