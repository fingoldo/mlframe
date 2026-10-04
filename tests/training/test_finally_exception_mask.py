"""Wave 52 (2026-05-20): finally-block masking in-flight exception.

Audit class: finally block calls a function that can raise (cleanup, release,
psutil.memory_info, GPU free_device, context-manager __exit__) -- if it
raises, the in-flight exception from the try body is silently masked.
Subset: __exit__(None, None, None) passes lying-clean state to inner CM,
breaking exception-aware suppression.

4 P1 + 2 P2 fixes applied:

  P1:
    1. training/feature_handling/locking.py:175 (PIDAwareFileLock.release)
       Wrap self._lock.release() in try/except WARN; only self._held=False
       belongs in finally.

    2. training/composite_cache.py:708 (DiscoveryCache._evict_to_caps)
       Capture sys.exc_info() and forward to _lock_ctx.__exit__; wrap
       __exit__ itself in try/except. (CM contract + cleanup-mask fix.)

    3. training/feature_handling/cache_backend.py:188 (DiskBackend LRU filelock)
       Same pattern as #2.

    4. feature_engineering/transformer/row_attention.py:151 (GPU cleanup)
       Wrap bank.free_device() in try/except. CUDA OOM in attend() often
       breaks the context; free_device on broken context raises again,
       masking the original OOM.

  P2:
    5. training/logging_transformers.py:62 (timing decorator)
       Wrap proc.memory_info().rss read in try/except defaulting 0.0;
       psutil.NoSuchProcess on zombie pool worker would have masked the
       func() exception.

    6. training/pipeline.py:417 (PySR temp column cleanup)
       Wrap train_df.drop in try/except so corrupted-MultiIndex KeyError
       doesn't mask the in-flight exception.

Verified safe (do not refactor): all other 13 finally sites already use
inner try/except (screen.py:116, mrmr.py:1151, registry.py:230, io.py:561)
or only do attribute writes / pre-captured timing / profiler.disable().

NO `return`-in-finally or `raise`-in-finally silent-discard patterns
found across the codebase -- that subclass is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Source-level sensors
# ---------------------------------------------------------------------------


def test_locking_release_wrapped_in_try_except(tmp_path, caplog) -> None:
    """A failing ``release()`` is logged as a warning, never masks the body's own exception, and still clears the held flag."""
    import logging

    import pytest

    pytest.importorskip("filelock")
    from mlframe.training.feature_handling.locking import PIDAwareFileLock

    lock = PIDAwareFileLock(str(tmp_path / "probe.lock"), timeout=5.0)
    real_release = []

    class _BodyError(RuntimeError):
        """The in-flight exception that must survive the failing release."""

    with caplog.at_level(logging.WARNING):
        with pytest.raises(_BodyError):
            with lock:
                def _boom(*args, **kwargs):
                    """Simulate a filelock release failure."""
                    raise OSError("release failed")

                real_release.append(lock._lock.release)  # type: ignore[union-attr]
                lock._lock.release = _boom  # type: ignore[union-attr]
                raise _BodyError("body failure")
    real_release[0]()
    assert any("PIDAwareFileLock.release() failed" in r.getMessage() for r in caplog.records)
    assert lock._held is False


class _RecordingLock:
    """Lock manager that records the arguments ``__exit__`` receives and can fail on release."""

    def __init__(self, exit_error=None):
        """Remember the error ``__exit__`` should raise, if any."""
        self.exit_error = exit_error
        self.exit_args = None

    def __enter__(self):
        """Enter the lock."""
        return self

    def __exit__(self, *exc):
        """Record the exception triple; raise the configured release error."""
        self.exit_args = exc
        if self.exit_error is not None:
            raise self.exit_error
        return False


def _discovery_cache_with_lock(monkeypatch, tmp_path, lock, locked_body):
    """DiscoveryCache whose eviction lock is ``lock`` and whose locked eviction body is ``locked_body``."""
    from mlframe.training.composite.cache_store import DiscoveryCache

    cache = DiscoveryCache(tmp_path, max_entries=1)
    monkeypatch.setattr(cache, "_maybe_filelock", lambda path: lock)
    monkeypatch.setattr(cache, "_evict_to_caps_locked", locked_body)
    return cache


def test_composite_cache_evict_forwards_exc_info(monkeypatch, tmp_path) -> None:
    """The eviction lock's __exit__ sees the body's in-flight exception, and a clean body gives it the all-None triple."""
    body_error = ValueError("eviction body failed")

    def failing_body():
        """Eviction body that fails."""
        raise body_error

    lock = _RecordingLock()
    with pytest.raises(ValueError, match="eviction body failed"):
        _discovery_cache_with_lock(monkeypatch, tmp_path, lock, failing_body)._evict_to_caps()
    assert lock.exit_args is not None
    assert lock.exit_args[0] is ValueError
    assert lock.exit_args[1] is body_error

    clean_lock = _RecordingLock()
    assert _discovery_cache_with_lock(monkeypatch, tmp_path, clean_lock, lambda: 3)._evict_to_caps() == 3
    assert clean_lock.exit_args == (None, None, None)


def test_cache_backend_lru_filelock_forwards_exc_info(monkeypatch, tmp_path) -> None:
    """The LRU sidecar's file lock __exit__ sees the critical section's in-flight exception, and a clean section gives it the all-None triple."""
    from mlframe.training.feature_handling import cache_backend

    locks: list = []

    class RecordingFileLock:
        """Stand-in for PIDAwareFileLock that records its __exit__ arguments."""

        def __init__(self, path, timeout=None):
            """Register this lock."""
            self.exit_args = None
            locks.append(self)

        def __enter__(self):
            """Enter."""
            return self

        def __exit__(self, *exc):
            """Record the exception triple."""
            self.exit_args = exc
            return False

    monkeypatch.setattr(cache_backend, "PIDAwareFileLock", RecordingFileLock)
    backend = cache_backend.LocalDiskBackend(str(tmp_path))
    section_error = ValueError("critical section failed")
    with pytest.raises(ValueError, match="critical section failed"):
        with backend._lru_locked():
            raise section_error
    assert locks[-1].exit_args is not None
    assert locks[-1].exit_args[0] is ValueError
    assert locks[-1].exit_args[1] is section_error

    with backend._lru_locked():
        pass
    assert locks[-1].exit_args == (None, None, None)


def test_row_attention_free_device_wrapped(monkeypatch, caplog) -> None:
    """A free_device() failure after a failed attend() is logged and the attend() error is the one that propagates."""
    import logging

    from mlframe.feature_engineering.transformer import row_attention

    class BrokenBank:
        """Key bank whose device cleanup fails, as after a CUDA error."""

        def to_device(self):
            """Pretend to move to the GPU."""

        def free_device(self):
            """Fail like a broken CUDA context."""
            raise RuntimeError("cuda context is broken")

    def failing_attend(**kwargs):
        """Fail like a CUDA OOM."""
        raise ValueError("attend ran out of memory")

    monkeypatch.setattr(row_attention, "build_key_bank", lambda **kwargs: BrokenBank())
    monkeypatch.setattr(row_attention, "is_gpu_available", lambda: True)
    monkeypatch.setattr(row_attention, "attend", failing_attend)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 16)).astype(np.float32)
    y = rng.normal(size=40).astype(np.float32)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ValueError, match="attend ran out of memory"):
            row_attention.compute_row_attention(X, y, X[:5], None, seed=1, k=4, keep_key_bank_on_gpu=True)
    assert any("free_device() failed" in r.getMessage() and "cuda context is broken" in r.getMessage() for r in caplog.records)


def test_logging_transformers_psutil_wrapped() -> None:
    """``log_resources``'s post-call RSS re-measurement is wrapped in try/except defaulting to 0.0
    (P2 fix #5 above): a ``psutil.NoSuchProcess`` on a zombie pool worker must not mask the wrapped
    call's own exception, and the emitted log record must still carry a usable (zeroed) ``rss_mb`` /
    ``d_rss_mb`` rather than propagating. Behavioural sensor (not a source-text pin): patches
    ``psutil.Process.memory_info`` to raise only on its SECOND call (the post-call re-measurement;
    the pre-call baseline read must still succeed) and asserts the decorated function's own exception
    is what actually propagates, plus the log record's rss_mb defaults to 0.0."""
    import logging

    import psutil
    import pytest

    from mlframe.training.logging_transformers import log_resources

    class _Boom(RuntimeError):
        """The wrapped call's own failure -- must survive the finally-block RSS re-measurement failure."""

    class _Dummy:
        """Bare host object for the ``log_resources`` decorator under test."""

        @log_resources(stage="probe")
        def method(self):
            """Always raise, so the finally block's RSS failure has a real in-flight exception to (not) mask."""
            raise _Boom("inner failure")

    _call_count = {"n": 0}
    _real_memory_info = psutil.Process.memory_info

    def _flaky_memory_info(self):
        """Succeed on the pre-call baseline read, fail on the post-call re-measurement (2nd+ call)."""
        _call_count["n"] += 1
        if _call_count["n"] >= 2:
            raise psutil.NoSuchProcess(pid=0)
        return _real_memory_info(self)

    records: list = []

    class _CapturingHandler(logging.Handler):
        """Collects emitted LogRecords for direct inspection of the ``extra`` payload."""

        def emit(self, record):
            """Append the record; no formatting/output needed for this sensor."""
            records.append(record)

    handler = _CapturingHandler()
    logger = logging.getLogger("mlframe.training.logging_transformers")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        psutil.Process.memory_info = _flaky_memory_info
        with pytest.raises(_Boom):
            _Dummy().method()
    finally:
        psutil.Process.memory_info = _real_memory_info
        logger.removeHandler(handler)

    assert records, "log_resources must still emit a record when post-call RSS read fails"
    # rss1 (the post-call re-measurement) defaults to 0.0 on failure; rss0 (the pre-call baseline)
    # still succeeded (a real, positive process RSS), so d_rss_mb = rss1 - rss0 = -rss0 is negative,
    # not necessarily 0.0 -- only rss1 itself is pinned to the 0.0 default.
    assert records[0].rss_mb == 0.0
    assert records[0].d_rss_mb < 0.0


def test_pipeline_temp_target_drop_wrapped(monkeypatch, caplog) -> None:
    """A failing temp-target drop is logged at debug and does not mask the PySR fit failure that is being handled."""
    import logging
    import sys
    import types

    import numpy as np
    import pandas as pd

    from mlframe.training.configs import PreprocessingExtensionsConfig
    from mlframe.training.pipeline import _apply_pysr_fe

    def failing_pysr(*args, **kwargs):
        """PySR run that fails."""
        raise RuntimeError("pysr search failed")

    fake_module = types.ModuleType("mlframe.feature_engineering.bruteforce")
    fake_module.run_pysr_feature_engineering = failing_pysr
    monkeypatch.setitem(sys.modules, "mlframe.feature_engineering.bruteforce", fake_module)

    def failing_drop(self, *args, **kwargs):
        """Fail like a corrupted-MultiIndex frame."""
        raise KeyError("corrupted column index")

    monkeypatch.setattr(pd.DataFrame, "drop", failing_drop)
    frame = pd.DataFrame({"x1": np.arange(20.0), "x2": np.arange(20.0)[::-1]})
    cfg = PreprocessingExtensionsConfig(pysr_enabled=True, random_seed=1)
    with caplog.at_level(logging.DEBUG, logger="mlframe.training.pipeline"):
        added = _apply_pysr_fe(train_df=frame, val_df=None, test_df=None, y_train=np.arange(20.0), config=cfg, verbose=0)
    assert added == []
    messages = [r.getMessage() for r in caplog.records]
    assert any("PySR fit failed" in m and "pysr search failed" in m for m in messages)
    assert any("temp_target_col drop failed" in m and "corrupted column index" in m for m in messages)


# ---------------------------------------------------------------------------
# Behavioural sensor: in-flight exception is preserved through finally.
# ---------------------------------------------------------------------------


def test_finally_with_raising_cleanup_does_not_mask_original_exception(monkeypatch, tmp_path, caplog) -> None:
    """A lock release that raises inside the eviction finally block is logged, and the body's own exception is the one that propagates."""
    import logging

    def failing_body():
        """Eviction body that fails."""
        raise ValueError("real bug")

    lock = _RecordingLock(exit_error=OSError("simulated filelock release failure"))
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ValueError, match="real bug"):
            _discovery_cache_with_lock(monkeypatch, tmp_path, lock, failing_body)._evict_to_caps()
    assert lock.exit_args is not None
    assert any("filelock __exit__ failed" in r.getMessage() and "simulated filelock release failure" in r.getMessage() for r in caplog.records)
