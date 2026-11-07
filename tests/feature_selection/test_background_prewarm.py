"""The fit-entry prewarm runs the once-per-process numba warmups on a background thread, and the numba.cuda probe runs once even when several threads ask at once."""

import threading
import time

from mlframe.feature_selection.filters import _internals as I
from mlframe.feature_selection.filters import _numba_warmup as W


def test_probe_runs_once_under_concurrent_callers(monkeypatch):
    """Eight threads asking for the verdict at the same moment trigger one probe and all get its answer."""
    calls = []

    def slow_probe(attempt):
        """A probe slow enough for the callers to overlap."""
        calls.append(attempt)
        time.sleep(0.2)
        return True

    monkeypatch.setattr(I, "_NUMBA_CUDA_CAN_COMPILE", None)
    monkeypatch.setattr(I, "_probe_numba_cuda_once", slow_probe)
    out = []
    threads = [threading.Thread(target=lambda: out.append(I.numba_cuda_can_compile())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls == [0] and out == [True] * 8


def test_background_prewarm_starts_once_and_runs_both_warmups(monkeypatch):
    """The first call starts one thread that runs the typed.Dict warmup and the probe; a second call starts nothing."""
    ran = []
    monkeypatch.setattr(W, "_BACKGROUND_STARTED", False)
    monkeypatch.delenv("MLFRAME_SKIP_NUMBA_WARMUP", raising=False)
    monkeypatch.setenv("MLFRAME_FIT_BACKGROUND_PREWARM", "1")
    monkeypatch.setattr(W, "warmup_typed_dict", lambda: ran.append("dict"))
    monkeypatch.setattr(I, "numba_cuda_can_compile", lambda: ran.append("probe") or True)
    before = threading.active_count()
    W.start_background_prewarm()
    W.start_background_prewarm()
    deadline = time.time() + 5
    while len(ran) < 2 and time.time() < deadline:
        time.sleep(0.01)
    assert ran == ["dict", "probe"]
    assert threading.active_count() <= before + 1


def test_skip_switch_starts_nothing(monkeypatch):
    """``MLFRAME_SKIP_NUMBA_WARMUP=1`` keeps the prewarm off."""
    ran = []
    monkeypatch.setattr(W, "_BACKGROUND_STARTED", False)
    monkeypatch.setenv("MLFRAME_SKIP_NUMBA_WARMUP", "1")
    monkeypatch.setenv("MLFRAME_FIT_BACKGROUND_PREWARM", "1")
    monkeypatch.setattr(W, "warmup_typed_dict", lambda: ran.append("dict"))
    W.start_background_prewarm()
    time.sleep(0.1)
    assert ran == [] and W._BACKGROUND_STARTED is False


def test_off_by_default(monkeypatch):
    """Without the opt-in the fit-entry prewarm starts nothing."""
    ran = []
    monkeypatch.setattr(W, "_BACKGROUND_STARTED", False)
    monkeypatch.delenv("MLFRAME_FIT_BACKGROUND_PREWARM", raising=False)
    monkeypatch.setattr(W, "warmup_typed_dict", lambda: ran.append("dict"))
    W.start_background_prewarm()
    time.sleep(0.1)
    assert ran == [] and W._BACKGROUND_STARTED is False
