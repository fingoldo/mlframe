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

    monkeypatch.setattr(par, "_PROFILES_REGISTERED", False)
    monkeypatch.setattr(par, "settings", _Proxy)
    t1 = threading.Thread(target=par.register_profiles)
    t2 = threading.Thread(target=par.register_profiles)
    t1.start()
    assert entered.wait(timeout=3)
    t2.start()
    t2.join(timeout=0.2)
    assert t2.is_alive(), "the second caller registered while the first was mid-registration"
    release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert calls["n"] == 3
