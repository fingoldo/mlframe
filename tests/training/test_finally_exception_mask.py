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

from pathlib import Path

MLFRAME_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe"


def _read(rel: str) -> str:
    """Read a source file. A flat module that became a subpackage
    (``X.py`` -> ``X/__init__.py`` + submodules) is read as the package
    __init__ plus every submodule so structural source pins still match."""
    _path = MLFRAME_ROOT / rel
    if not _path.exists() and _path.suffix == ".py":
        _pkg = _path.with_suffix("")
        _init = _pkg / "__init__.py"
        if _init.exists():
            parts = [_init.read_text(encoding="utf-8")]
            for _sub in sorted(_pkg.glob("*.py")):
                if _sub.name != "__init__.py":
                    parts.append(_sub.read_text(encoding="utf-8"))
            return "\n".join(parts)
    return _path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Source-level sensors
# ---------------------------------------------------------------------------


def test_locking_release_wrapped_in_try_except() -> None:
    """Locking release wrapped in try except."""
    src = _read("training/feature_handling/locking.py")
    # The pre-fix bare release() inside the outer try is gone.
    # The post-fix wraps it explicitly.
    assert "PIDAwareFileLock.release() failed for" in src
    # And the finally still only sets _held = False.
    assert "self._held = False" in src


def test_composite_cache_evict_forwards_exc_info() -> None:
    """Composite cache evict forwards exc info."""
    src = _read("training/composite/cache_store.py")
    # The pre-fix `__exit__(None, None, None)` is replaced with `__exit__(*_exc)`.
    assert "_lock_ctx.__exit__(None, None, None)" not in src
    assert "_lock_ctx.__exit__(*_exc)" in src
    # Wrapped in try/except so __exit__ failure doesn't propagate.
    assert "DiscoveryCache eviction filelock __exit__ failed" in src


def test_cache_backend_lru_filelock_forwards_exc_info() -> None:
    """Cache backend lru filelock forwards exc info."""
    src = _read("training/feature_handling/cache_backend.py")
    assert "file_lock.__exit__(None, None, None)" not in src
    assert "file_lock.__exit__(*_exc)" in src
    assert "DiskBackend LRU filelock __exit__ failed" in src


def test_row_attention_free_device_wrapped() -> None:
    """Row attention free device wrapped."""
    src = _read("feature_engineering/transformer/row_attention.py")
    # The fix wraps free_device in try/except WARN.
    assert "bank.free_device() failed (likely after upstream CUDA error)" in src


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


def test_pipeline_temp_target_drop_wrapped() -> None:
    """Pipeline temp target drop wrapped."""
    src = _read("training/pipeline.py")
    # The fix wraps drop in try/except DEBUG.
    assert "pipeline: temp_target_col drop failed in finally" in src


# ---------------------------------------------------------------------------
# Behavioural sensor: in-flight exception is preserved through finally.
# ---------------------------------------------------------------------------


def test_finally_with_raising_cleanup_does_not_mask_original_exception() -> None:
    """Validate the bug-class invariant: a finally that catches its own cleanup
    error preserves the original exception from the try body."""
    seen = []

    class FakeLock:
        """Groups tests covering fake lock."""
        def release(self):
            """Release."""
            raise OSError("simulated filelock release failure")

    fl = FakeLock()
    try:
        try:
            raise ValueError("real bug")
        finally:
            # Mirrors the locking.py:175 fix pattern.
            try:
                fl.release()
            except Exception as _rel_err:
                seen.append(("release_failed", _rel_err))
    except ValueError as ve:
        seen.append(("propagated", str(ve)))
    # The release error was logged; the original ValueError propagated.
    assert ("release_failed",) == (seen[0][0],)
    assert seen[1] == ("propagated", "real bug")
