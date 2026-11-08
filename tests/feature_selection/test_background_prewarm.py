"""The fit-entry prewarm runs the once-per-process numba warmups on a background thread, and the numba.cuda probe runs once even when several threads ask at once."""

import threading

from mlframe.feature_selection.filters import _internals as I
from mlframe.feature_selection.filters import _numba_warmup as W


def _record_thread_starts(monkeypatch):
    """Record every Thread.start made while the patch is active; the prewarm starts its worker synchronously, so an empty list proves nothing was started."""
    started = []
    real_start = threading.Thread.start

    def recording_start(self):
        """Remember the name of the thread, then start it as usual."""
        started.append(self.name)
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", recording_start)
    return started


def test_probe_runs_once_under_concurrent_callers(monkeypatch):
    """Eight threads asking for the verdict at the same moment trigger one probe and all get its answer."""
    calls = []
    entered = threading.Event()
    release = threading.Event()
    about_to_call = []

    def slow_probe(attempt):
        """A probe held open until every caller has been launched, so the callers overlap."""
        calls.append(attempt)
        entered.set()
        assert release.wait(30)
        return True

    monkeypatch.setattr(I, "_NUMBA_CUDA_CAN_COMPILE", None)
    monkeypatch.setattr(I, "_probe_numba_cuda_once", slow_probe)
    out = []

    def _caller():
        """One concurrent caller of the probe: note that it is about to call, then store the result."""
        about_to_call.append(1)
        out.append(I.numba_cuda_can_compile())

    threads = [threading.Thread(target=_caller) for _ in range(8)]
    for t in threads:
        t.start()
    assert entered.wait(30)
    done = threading.Event()
    while len(about_to_call) < 8 and not done.wait(0.005):
        pass
    release.set()
    for t in threads:
        t.join(30)
    assert not any(t.is_alive() for t in threads)
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
    for t in [t for t in threading.enumerate() if t.name == "mlframe-fit-prewarm"]:
        t.join(30)
    assert ran == ["dict", "probe"]
    assert threading.active_count() <= before + 1


def test_skip_switch_starts_nothing(monkeypatch):
    """``MLFRAME_SKIP_NUMBA_WARMUP=1`` keeps the prewarm off."""
    ran = []
    monkeypatch.setattr(W, "_BACKGROUND_STARTED", False)
    monkeypatch.setenv("MLFRAME_SKIP_NUMBA_WARMUP", "1")
    monkeypatch.setenv("MLFRAME_FIT_BACKGROUND_PREWARM", "1")
    monkeypatch.setattr(W, "warmup_typed_dict", lambda: ran.append("dict"))
    started = _record_thread_starts(monkeypatch)
    W.start_background_prewarm()
    assert started == [] and ran == [] and W._BACKGROUND_STARTED is False


def test_off_by_default(monkeypatch):
    """Without the opt-in the fit-entry prewarm starts nothing."""
    ran = []
    monkeypatch.setattr(W, "_BACKGROUND_STARTED", False)
    monkeypatch.delenv("MLFRAME_FIT_BACKGROUND_PREWARM", raising=False)
    monkeypatch.setattr(W, "warmup_typed_dict", lambda: ran.append("dict"))
    started = _record_thread_starts(monkeypatch)
    W.start_background_prewarm()
    assert started == [] and ran == [] and W._BACKGROUND_STARTED is False
