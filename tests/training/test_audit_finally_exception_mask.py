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

import pytest


MLFRAME_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe"


def _read(rel: str) -> str:
    """Read a source file. For ``training/pipeline.py`` (carved into
    ``_pipeline_extensions.py`` / ``_pipeline_fit_transform.py`` siblings),
    concat the siblings so the source-grep boundary check still matches the
    relocated code."""
    primary = (MLFRAME_ROOT / rel).read_text(encoding="utf-8")
    if rel == "training/pipeline.py":
        _dir = MLFRAME_ROOT / "training"
        for nm in ("_pipeline_extensions.py", "_pipeline_fit_transform.py"):
            _sib = _dir / nm
            if _sib.exists():
                primary = primary + "\n" + _sib.read_text(encoding="utf-8")
    return primary


# ---------------------------------------------------------------------------
# Source-level sensors
# ---------------------------------------------------------------------------


def test_locking_release_wrapped_in_try_except() -> None:
    src = _read("training/feature_handling/locking.py")
    # The pre-fix bare release() inside the outer try is gone.
    # The post-fix wraps it explicitly.
    assert "PIDAwareFileLock.release() failed for" in src
    # And the finally still only sets _held = False.
    assert "self._held = False" in src


def test_composite_cache_evict_forwards_exc_info() -> None:
    src = _read("training/composite_cache.py")
    # The pre-fix `__exit__(None, None, None)` is replaced with `__exit__(*_exc)`.
    assert "_lock_ctx.__exit__(None, None, None)" not in src
    assert "_lock_ctx.__exit__(*_exc)" in src
    # Wrapped in try/except so __exit__ failure doesn't propagate.
    assert "DiscoveryCache eviction filelock __exit__ failed" in src


def test_cache_backend_lru_filelock_forwards_exc_info() -> None:
    src = _read("training/feature_handling/cache_backend.py")
    assert "file_lock.__exit__(None, None, None)" not in src
    assert "file_lock.__exit__(*_exc)" in src
    assert "DiskBackend LRU filelock __exit__ failed" in src


def test_row_attention_free_device_wrapped() -> None:
    src = _read("feature_engineering/transformer/row_attention.py")
    # The fix wraps free_device in try/except WARN.
    assert "bank.free_device() failed (likely after upstream CUDA error)" in src


def test_logging_transformers_psutil_wrapped() -> None:
    src = _read("training/logging_transformers.py")
    # The fix wraps rss read in try/except defaulting to 0.0.
    assert "rss1 = proc.memory_info().rss / 1024 ** 2\n                except Exception:\n                    rss1 = 0.0" in src


def test_pipeline_temp_target_drop_wrapped() -> None:
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
        def release(self):
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
    assert ("release_failed", ) == (seen[0][0],)
    assert seen[1] == ("propagated", "real bug")
