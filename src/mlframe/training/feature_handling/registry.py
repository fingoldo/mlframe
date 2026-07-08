"""
Provider lifecycle registry: weakref + LRU strong-keep tier + per-key
locks + Future-based pre-warm.

Round-3 architecture R2-2 + performance F1: the naive
``_PROVIDER_REGISTRY: dict[sig, weakref]`` had a TOCTOU race in lazy
load (two threads both pass the ``not in`` check, both load 2x VRAM).
F1 also flagged that pure weakref churn is a ~5s cold-load per
target-loop iteration; an LRU strong-keep tier of N=2 providers (or
``"auto"``-derived from free VRAM) eliminates the churn for the
common 50-target sweep.

Concurrency model:
  * ``_REGISTRY_LOCK`` serialises registry mutations (creation,
    LRU promotion, eviction).
  * Per-key ``_ProviderEntry.lock`` serialises actual ``acquire()`` /
    ``release()`` work on a single provider.
  * Double-checked locking on creation: re-check after acquiring the
    registry lock so the "first thread loads, second thread reuses"
    path is correct.

Public API (consumed via the FeatureHandlingConfig cache layer):
  * :func:`acquire_provider(provider, fhc)` -- context manager that
    bumps refcount on enter, drops on exit. Banned naked acquire/
    release so callers can't leak.
  * :func:`shutdown_all()` -- drop everything; useful for notebook
    reload.

Module-level state (per-process):
  * :data:`_REGISTRY` -- ``WeakValueDictionary[signature, _ProviderEntry]``
  * :data:`_LRU_HARD` -- ``OrderedDict[signature, _ProviderEntry]``
    strong-keep tier capped by ``cache.keep_n_providers``.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import weakref
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mlframe.training.feature_handling.protocols import FrozenFeaturizerProvider


# =====================================================================
# Provider entry
# =====================================================================


@dataclass
class _ProviderEntry:
    """Per-signature registry entry."""
    provider: FrozenFeaturizerProvider
    refcount: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    is_loaded: bool = False  # True after first successful acquire()


# =====================================================================
# Module-level state
# =====================================================================

_REGISTRY: weakref.WeakValueDictionary[str, _ProviderEntry] = weakref.WeakValueDictionary()
_LRU_HARD: OrderedDict[str, _ProviderEntry] = OrderedDict()
_REGISTRY_LOCK = threading.Lock()
_PREWARM_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mlframe-prewarm")
_PREWARM_FUTURES: "dict[str, Future]" = {}  # signature -> Future


def shutdown_prewarm_executor(wait: bool = False) -> None:
    """Explicitly shut down the module-level prewarm ThreadPoolExecutor.

    Long-running services that hot-reload mlframe accumulate 2 zombie worker
    threads per reload because the executor is created at import time and
    only torn down at interpreter exit (Python's default ``atexit``). Test
    suites and service code that want to release threads earlier can call
    this helper.
    """
    try:
        _PREWARM_EXECUTOR.shutdown(wait=wait)
    except (RuntimeError, OSError):
        # Already shut down or interpreter mid-teardown; either way the threads
        # are or will be gone soon.
        pass


# Register the shutdown so interpreter exit reliably reclaims the executor's
# worker threads even if a hot-reload path didn't call shutdown_prewarm_executor
# explicitly. ``wait=False`` so the exit isn't delayed by in-flight prewarms.
import atexit as _atexit  # noqa: E402 -- module-level state block
_atexit.register(shutdown_prewarm_executor, wait=False)


# =====================================================================
# Registry mutators
# =====================================================================


def _resolve_keep_n(cache_cfg) -> int:
    """Resolve ``cache.keep_n_providers`` to a concrete int.

    "auto" derivation: ``min(2, max(1, free_vram_gb // 1))`` on GPU,
    else 2 (CPU-only, RAM is cheap).
    """
    val = getattr(cache_cfg, "keep_n_providers", "auto")
    if isinstance(val, int):
        return val
    # "auto"
    try:
        import torch
        if not torch.cuda.is_available():
            return 2
        free, _ = torch.cuda.mem_get_info()
        free_gb = free / 1e9
        return min(2, max(1, int(free_gb)))
    except ImportError:
        return 2


def _register_or_get(
    signature: str,
    provider_factory,
) -> _ProviderEntry:
    """Atomic get-or-create. Double-checked locking avoids the TOCTOU
    race where two threads both create the same provider.

    ``provider_factory`` is a zero-arg callable returning a
    :class:`FrozenFeaturizerProvider`. Called only when no entry
    exists yet.
    """
    # Fast path: entry exists.
    entry = _REGISTRY.get(signature)
    if entry is not None:
        return entry

    # Slow path: take the lock and re-check.
    with _REGISTRY_LOCK:
        entry = _REGISTRY.get(signature)
        if entry is not None:
            return entry
        provider = provider_factory()
        entry = _ProviderEntry(provider=provider)
        _REGISTRY[signature] = entry
        return entry


def _bump_lru(signature: str, entry: _ProviderEntry, keep_n: int) -> None:
    """Promote ``signature`` in the strong-keep tier; evict the
    oldest if tier exceeds ``keep_n``. Caller holds REGISTRY_LOCK.
    """
    if signature in _LRU_HARD:
        _LRU_HARD.move_to_end(signature)
    else:
        _LRU_HARD[signature] = entry
    while len(_LRU_HARD) > keep_n:
        _, evicted = _LRU_HARD.popitem(last=False)
        # Evict from strong-tier doesn't unload immediately -- the
        # weakref still holds it if anything else refs it. If refcount
        # is zero, unload now.
        with evicted.lock:
            if evicted.refcount == 0 and evicted.is_loaded:
                try:
                    evicted.provider.release()
                    evicted.is_loaded = False
                except Exception as e:  # pragma: no cover
                    logger.warning("evicting provider release() raised: %s", e)


# =====================================================================
# Context-manager acquire (the only public entry point)
# =====================================================================


@contextlib.contextmanager
def acquire_provider(
    provider: FrozenFeaturizerProvider,
    cache_cfg,
) -> Iterator[FrozenFeaturizerProvider]:
    """Context-manager around provider load + refcount.

    Usage::

        with acquire_provider(provider, fhc.cache) as p:
            embeddings = p.transform(["hello", "world"])

    On enter:
      1. Look up registry entry by ``provider.signature`` (or create).
      2. Lock entry, increment refcount, ``acquire()`` if first ref.
      3. Promote in LRU strong-keep tier.
    On exit:
      1. Lock entry, decrement refcount.
      2. If refcount==0 AND not in LRU tier, ``release()``.

    Naked ``provider.acquire()`` is banned -- the refcount semantic is
    too easy to mess up otherwise.
    """
    signature = provider.signature
    entry = _register_or_get(signature, lambda: provider)

    # Bump the LRU strong-keep tier BEFORE the refcount goes back to
    # zero on the matching release-side path. The pre-fix order was:
    #   (a) entry.lock -> refcount += 1
    #   (b) drop entry.lock
    #   (c) _REGISTRY_LOCK -> bump LRU
    # Between (b) and (c) a sibling acquire-then-release could drop
    # refcount to zero and, finding ``signature not in _LRU_HARD``,
    # call ``release()`` on the very provider we still intend to use.
    # By inserting into ``_LRU_HARD`` first, the release-side guard
    # ``signature not in _LRU_HARD`` keeps the provider alive across
    # the window. The double-checked-lock for is_loaded follows.
    keep_n = _resolve_keep_n(cache_cfg)
    with _REGISTRY_LOCK:
        _bump_lru(signature, entry, keep_n)

    with entry.lock:
        if entry.refcount == 0 and not entry.is_loaded:
            entry.provider.acquire()
            entry.is_loaded = True
        entry.refcount += 1

    try:
        yield entry.provider
    finally:
        with entry.lock:
            entry.refcount -= 1
            if entry.refcount == 0 and signature not in _LRU_HARD:
                # Not in strong-keep tier and refcount dropped to zero
                # -> release immediately. (LRU eviction also calls
                # release; this branch handles refcount drops
                # outside the LRU.)
                try:
                    entry.provider.release()
                    entry.is_loaded = False
                except Exception as e:  # pragma: no cover
                    logger.warning("provider release() raised: %s", e)


# =====================================================================
# Pre-warm
# =====================================================================


def prewarm(provider: FrozenFeaturizerProvider) -> Future:
    """Schedule provider weight load on a background thread.

    Round-3 chaos C3: the previous naked ``threading.Thread`` swallowed
    exceptions to stderr and the main thread never knew. Returning a
    ``Future`` means the first ``acquire_provider`` call can do
    ``fut.result(timeout=...)`` to surface the error with full
    traceback.

    Idempotent per signature -- the Future is cached for the suite
    lifetime, so two prewarm calls return the same Future. The cache
    lookup + submit is serialised under ``_REGISTRY_LOCK`` so two
    threads racing on a fresh signature don't both submit a load job
    (which on a GPU provider doubles the VRAM peak).
    """
    signature = provider.signature

    def _do_load():
        # acquire_provider is a context manager; for prewarm we want
        # to load weights but NOT bump refcount (we have no consumer
        # yet). Call provider.acquire() under the registry lock
        # directly.
        entry = _register_or_get(signature, lambda: provider)
        try:
            with entry.lock:
                if not entry.is_loaded:
                    entry.provider.acquire()
                    entry.is_loaded = True
        except BaseException:
            # Prewarm failure leaves the entry in a half-broken state:
            # ``refcount=0``, ``is_loaded=False``, but the registry
            # still has a weakref to the provider object whose
            # ``acquire()`` we know to fail. Pre-fix the next
            # ``acquire_provider`` blindly re-called ``acquire()`` and
            # got the same failure - without the Future visible to
            # surface the original traceback. Drop the entry so the
            # next caller constructs a fresh provider (or fails fast
            # with the original exception once they call
            # ``wait_prewarm``).
            with _REGISTRY_LOCK:
                cur = _REGISTRY.get(signature)
                if cur is entry:
                    try:
                        del _REGISTRY[signature]
                    except KeyError:
                        pass
                _LRU_HARD.pop(signature, None)
            raise
        return signature

    with _REGISTRY_LOCK:
        fut = _PREWARM_FUTURES.get(signature)
        if fut is not None and not fut.done():
            return fut
        if fut is not None and fut.exception() is None:
            # Already loaded successfully -- hand back the resolved future.
            return fut
        new_fut = _PREWARM_EXECUTOR.submit(_do_load)
        # Wave 43 (2026-05-20): if a future internal caller invokes prewarm() but
        # forgets the wait_prewarm() pair, exceptions stored on the cached future
        # would be silently retained. Attach a done-callback so an unawaited failure
        # at least logs once.
        def _log_unhandled(_fut: "Future") -> None:
            try:
                exc = _fut.exception(timeout=0)
            except Exception:
                return
            if exc is not None:
                logger.warning(
                    "prewarm(%r) failed; caller did not call wait_prewarm to surface it.",
                    signature, exc_info=exc,
                )
        new_fut.add_done_callback(_log_unhandled)
        _PREWARM_FUTURES[signature] = new_fut
        return new_fut


def wait_prewarm(provider: FrozenFeaturizerProvider, timeout: float = 600.0) -> None:
    """Block until the prewarm Future for ``provider.signature``
    finishes. Surfaces any exception (with full traceback) to the
    caller. No-op if no prewarm was scheduled.
    """
    signature = provider.signature
    fut = _PREWARM_FUTURES.get(signature)
    if fut is None:
        return
    fut.result(timeout=timeout)  # raises on prewarm failure


# =====================================================================
# Shutdown -- notebook reload safety
# =====================================================================


def shutdown_all() -> None:
    """Drop every provider, release VRAM, clear the strong-keep tier
    and the prewarm executor.

    Round-3 chaos C18: ``importlib.reload(mlframe.feature_handling)``
    rebinds module-level globals; old providers' weakref entries
    vanish but their VRAM is still held. Calling this between
    notebook reloads frees VRAM cleanly. Also useful at suite-end
    for explicit cleanup in CI.
    """
    with _REGISTRY_LOCK:
        for signature, entry in list(_LRU_HARD.items()):
            with entry.lock:
                if entry.is_loaded:
                    try:
                        entry.provider.release()
                    except Exception as e:  # pragma: no cover
                        logger.warning("shutdown: release(%s) raised: %s", signature, e)
                    entry.is_loaded = False
        _LRU_HARD.clear()

        # Walk live weakref entries too (covers entries that aren't
        # in the LRU because they were just created or recently
        # evicted).
        for signature in list(_REGISTRY.keys()):
            weak_entry = _REGISTRY.get(signature)
            if weak_entry is None:
                continue
            with weak_entry.lock:
                if weak_entry.is_loaded:
                    try:
                        weak_entry.provider.release()
                    except Exception as e:  # pragma: no cover
                        logger.warning("shutdown: release(%s) raised: %s", signature, e)
                    weak_entry.is_loaded = False

    _PREWARM_FUTURES.clear()


def provider_status() -> dict:
    """Snapshot of the registry for ``fhc.describe()`` and tests.

    Round-3 U-R2-24: background warmup errors must be surface-able
    via the FHC API.
    """
    out: dict[str, dict[str, Any]] = {}
    for signature, entry in list(_LRU_HARD.items()):
        out[signature] = {
            "loaded": entry.is_loaded,
            "refcount": entry.refcount,
            "in_lru_tier": True,
        }
    for signature in list(_REGISTRY.keys()):
        if signature in out:
            continue
        weak_entry = _REGISTRY.get(signature)
        if weak_entry is None:
            continue
        out[signature] = {
            "loaded": weak_entry.is_loaded,
            "refcount": weak_entry.refcount,
            "in_lru_tier": False,
        }
    # Surface pre-warm errors
    for signature, fut in list(_PREWARM_FUTURES.items()):
        if fut.done() and fut.exception() is not None:
            out.setdefault(signature, {})["prewarm_error"] = repr(fut.exception())
    return out


__all__ = [
    "acquire_provider",
    "prewarm",
    "wait_prewarm",
    "shutdown_all",
    "provider_status",
]
