"""U7: ``register_profiles`` sets its flag only after the registration succeeded."""
import pytest

pytest.importorskip("hypothesis")
from hypothesis import settings

import mlframe.testing.parametric as par


def test_register_profiles_retries_after_transient_failure(monkeypatch):
    """The first registration raises; the flag stays unset and the second call performs the registration."""
    real = settings.register_profile
    calls = {"n": 0}

    def _flaky(*a, **k):
        """Raise on the first call, then delegate."""
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return real(*a, **k)

    monkeypatch.setattr(par, "_PROFILES_REGISTERED", False)

    class _Proxy:
        """Stand-in for ``settings`` (the real class is immutable) with a flaky ``register_profile``."""

        register_profile = staticmethod(_flaky)
        load_profile = staticmethod(settings.load_profile)

    monkeypatch.setattr(par, "settings", _Proxy)
    with pytest.raises(RuntimeError):
        par.register_profiles()
    assert par._PROFILES_REGISTERED is False
    par.register_profiles()
    assert par._PROFILES_REGISTERED is True
    assert calls["n"] >= 4


def test_register_profiles_concurrent_callers_register_once(monkeypatch):
    """A second thread calling while the first is mid-registration waits instead of registering again."""
    import threading

    real = settings.register_profile
    calls = {"n": 0}
    entered = threading.Event()
    release = threading.Event()

    def _slow(*a, **k):
        """Block inside the first registration until released."""
        calls["n"] += 1
        if calls["n"] == 1:
            entered.set()
            release.wait(timeout=3)
        return real(*a, **k)

    class _Proxy:
        """Stand-in for ``settings`` (the real class is immutable) with a slow ``register_profile``."""

        register_profile = staticmethod(_slow)
        load_profile = staticmethod(settings.load_profile)

    waiting = threading.Event()

    class _WatchedLock:
        """The registration lock, reporting a caller that arrives while it is held."""

        def __init__(self):
            self._lock = threading.Lock()

        def __enter__(self):
            if self._lock.locked():
                waiting.set()
            self._lock.acquire()

        def __exit__(self, *exc):
            self._lock.release()

    monkeypatch.setattr(par, "_PROFILES_REGISTERED", False)
    monkeypatch.setattr(par, "_PROFILES_LOCK", _WatchedLock())
    monkeypatch.setattr(par, "settings", _Proxy)
    t1 = threading.Thread(target=par.register_profiles)
    t2 = threading.Thread(target=par.register_profiles)
    t1.start()
    assert entered.wait(timeout=3)
    t2.start()
    assert waiting.wait(timeout=3), "the second caller never reached the held lock"
    release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert calls["n"] == 3
