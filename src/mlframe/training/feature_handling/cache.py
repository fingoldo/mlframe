"""
Multi-tier feature-handling cache.

Architecture:

* **In-memory tier**: cheap session-keyed lookup. Within one
  ``train_mlframe_models_suite()`` call, ``id(train_df)`` is safe
  (we hold a strong ref) so no content hashing is done in hot path.
  This avoids the 30s/fit overhead from arrow-buffer materialisation.

* **Disk tier (opt-in)**: ``cache.persistence != "off"`` activates
  disk-backed storage with content-fingerprint-based keys (computed
  ONCE per suite). Stores fp16 numpy arrays via the existing
  :func:`mlframe.training.io.atomic_write_bytes` helper -- atomic
  rename, tempfile cleanup on exception. Reads via memmap so
  10 GB embedding caches don't double-allocate.

Reuse semantics:
  * ACROSS MODELS in same target iter: SAME ``InMemoryKey`` for all models (key is independent of
    consumer model).
  * ACROSS TARGETS in target loop: DIFFERENT ``InMemoryKey`` for target-encoder handlers -- the
    ``train_idx_token`` slot carries a blake2b digest over the target content
    (see ``apply._target_content_token``) so multi-target suites don't collide on the same encoder slot.
    Text handlers fold a column-content token instead so OD-masked vs full-train fits are distinct
    (the legacy literal-zero token collided across different row masks).
  * ACROSS WEIGHT SCHEMAS: SAME ``InMemoryKey`` (params / provider / target content all unchanged).
  * ACROSS sklearn.clone() boundaries: cache survives because the cache lives at the FHC layer, not
    the model.

Eviction:
  * In-memory: LRU within ``cache.ram_max_gb``. ``cache.ram_reserve_gb``
    floor on remaining system RAM via :func:`psutil.virtual_memory`.
  * Disk: oldest-file-first (by mtime -- reads don't touch mtime, so this is write-order, not a true
    access-order LRU) within ``cache.disk_evict_when_free_below_gb`` / ``cache.disk_min_free_gb``
    thresholds, checked after every disk write.

Eviction strategies:
  * "lru" -- straight LRU (default for in-memory).
  * "size_weighted" -- score = ``size_gb / (recompute_time_s * (1 +
    access_count))``; prefers evicting big-cheap entries.
  * "lfu" -- least-frequently-used; ties broken by LRU.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
)

import numpy as np
import psutil

from mlframe.training.feature_handling.fingerprint import (
    ContentFingerprint,
    DiskKey,
    InMemoryKey,
)
from mlframe.training.feature_handling.system import long_path_safe
from mlframe.training.io import atomic_write_bytes

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import scipy.sparse  # noqa: F401


# =====================================================================
# Cache entry metadata
# =====================================================================


@dataclass
class _CacheEntry:
    """In-memory cache entry. Tracks size + recompute time + access
    count for eviction scoring."""
    value: Any
    size_bytes: int
    recompute_time_s: float
    access_count: int = 0
    inserted_at: float = field(default_factory=time.monotonic)


# =====================================================================
# In-memory + disk cache
# =====================================================================


class FeatureCache:
    """Two-tier cache. One instance per :class:`FeatureHandlingConfig`
    (constructed lazily when the first handler runs).

    The cache is process-local. Cross-process locking on the disk
    tier is not implemented (single-process workload).

    Public surface:
      * :meth:`get_or_compute(key, compute_fn)` -- the only consumer
        API. Hits in-memory tier first, then disk, otherwise calls
        ``compute_fn()``, stores at both tiers (or in-memory only
        if persistence="off"), returns the value.
      * :meth:`stats()` -- snapshot for ``fhc.describe()``.
      * :meth:`clear()` -- wipe in-memory state (disk untouched).
    """

    def __init__(
        self,
        cache_cfg,
        content_fingerprint: Optional[ContentFingerprint] = None,
    ):
        self._cfg = cache_cfg
        self._content = content_fingerprint  # required only for disk tier
        self._mem: OrderedDict[InMemoryKey, _CacheEntry] = OrderedDict()
        # Map InMemoryKey -> matching DiskKey so we know where to look
        # / write on the disk tier without re-materialising the full
        # DiskKey from the InMemoryKey on every access.
        self._key_xref: Dict[InMemoryKey, DiskKey] = {}
        self._lock = threading.Lock()
        self._hits_mem = 0
        self._hits_disk = 0
        self._misses = 0
        self._evictions = 0

    def __getstate__(self) -> dict:
        """Drop the unpicklable ``threading.Lock`` (a fresh one is created in ``__setstate__``); no
        pickling path currently reaches this class, but it's the same "live object survives only
        because nobody pickles the parent yet" precondition that bit ``training/neural/ranker.py``'s
        ``trainer_``/CUDA-tensor exclusion once already -- defensive guard against a future caller."""
        state = self.__dict__.copy()
        state["_lock"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state and re-create the ``threading.Lock`` dropped by ``__getstate__``."""
        self.__dict__.update(state)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Snapshot of current cache size/hit-rate counters, consumed by ``fhc.describe()``."""
        with self._lock:
            ram_bytes = sum(e.size_bytes for e in self._mem.values())
            return {
                "n_keys": len(self._mem),
                "ram_bytes": ram_bytes,
                "ram_gb": ram_bytes / 1e9,
                "hits_mem": self._hits_mem,
                "hits_disk": self._hits_disk,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": ((self._hits_mem + self._hits_disk) / max(1, self._hits_mem + self._hits_disk + self._misses)),
                "persistence": self._cfg.persistence,
            }

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get_or_compute(
        self,
        in_mem_key: InMemoryKey,
        compute_fn: Callable[[], Any],
        *,
        disk_key: Optional[DiskKey] = None,
        size_estimator: Optional[Callable[[Any], int]] = None,
    ) -> Any:
        """Two-tier lookup with fall-through to ``compute_fn``.

        ``disk_key`` is required only when ``cache.persistence != "off"``;
        the caller (phase E concat layer) builds it from the suite
        :class:`ContentFingerprint`.

        ``size_estimator`` defaults to ``_default_size_estimator`` --
        sufficient for ndarrays / sparse matrices / dense lists. Pass
        a custom estimator only when the value is an opaque object.
        """
        # 1. In-memory tier
        with self._lock:
            entry = self._mem.get(in_mem_key)
            if entry is not None:
                entry.access_count += 1
                self._mem.move_to_end(in_mem_key)
                self._hits_mem += 1
                return entry.value

        # 2. Disk tier (opt-in)
        if self._cfg.persistence != "off" and disk_key is not None:
            disk_value = self._read_from_disk(disk_key)
            if disk_value is not None:
                # FH-XREF-NO-EVICT: write xref under the same lock that
                # owns ``_mem`` insertion + eviction so a concurrent
                # eviction can't strand the xref while the mem entry is
                # gone. Pre-fix the xref write was outside the lock and
                # ran AFTER ``_insert_in_memory`` already released the
                # lock -- a parallel get_or_compute could trigger
                # eviction in the gap.
                self._insert_in_memory(in_mem_key, disk_value, recompute_time_s=0.5, size_estimator=size_estimator, disk_key=disk_key)
                with self._lock:
                    self._hits_disk += 1
                return disk_value

        # 3. Compute fresh
        with self._lock:
            self._misses += 1
        t0 = time.monotonic()
        value = compute_fn()
        recompute_time_s = max(0.001, time.monotonic() - t0)

        # 4. Store
        write_xref = self._cfg.persistence in ("auto", "read_write") and disk_key is not None
        if write_xref:
            assert disk_key is not None  # guaranteed by the write_xref construction above
            self._write_to_disk(disk_key, value)
        self._insert_in_memory(
            in_mem_key, value, recompute_time_s=recompute_time_s,
            size_estimator=size_estimator,
            disk_key=disk_key if write_xref else None,
        )

        return value

    # ------------------------------------------------------------------
    # In-memory tier internals
    # ------------------------------------------------------------------

    def _insert_in_memory(
        self,
        key: InMemoryKey,
        value: Any,
        recompute_time_s: float,
        size_estimator: Optional[Callable[[Any], int]],
        disk_key: Optional[DiskKey] = None,
    ) -> None:
        """Wrap ``value`` in a ``_CacheEntry``, insert/refresh it at the MRU end, and evict if the tier is now over budget."""
        size = (size_estimator or _default_size_estimator)(value)
        entry = _CacheEntry(
            value=value,
            size_bytes=size,
            recompute_time_s=recompute_time_s,
        )
        with self._lock:
            self._mem[key] = entry
            self._mem.move_to_end(key)
            if disk_key is not None:
                self._key_xref[key] = disk_key
            self._evict_if_needed_locked()

    def _evict_if_needed_locked(self) -> None:
        """Evict from in-memory until under ``cache.ram_max_gb`` AND
        system RAM headroom > ``cache.ram_reserve_gb``. Caller holds
        ``self._lock``.
        """
        ram_max_bytes = self._cfg.ram_max_gb * 1e9 if self._cfg.ram_max_gb else float("inf")
        reserve_bytes = self._cfg.ram_reserve_gb * 1e9 if self._cfg.ram_reserve_gb else 0.0
        strategy = self._cfg.eviction_strategy

        while True:
            cache_size = sum(e.size_bytes for e in self._mem.values())
            try:
                avail = psutil.virtual_memory().available
            except Exception:  # pragma: no cover
                avail = float("inf")
            over_size = cache_size > ram_max_bytes
            under_reserve = avail < reserve_bytes
            if not over_size and not under_reserve:
                return
            if not self._mem:
                return

            evict_key = self._select_eviction_victim(strategy)
            if evict_key is None:
                return
            self._mem.pop(evict_key)
            # FH-XREF-NO-EVICT: drop the matching ``_key_xref`` entry alongside the ``_mem`` entry
            # so the xref table can't grow monotonically while ``_mem`` evicts. Pre-fix the xref
            # outlived the evicted mem entry, and a future ``get_or_compute`` hitting the disk tier
            # would replay the orphan entry's stale ``DiskKey`` mapping.
            self._key_xref.pop(evict_key, None)
            self._evictions += 1

    def _select_eviction_victim(
        self,
        strategy: Literal["lru", "lfu", "size_weighted"],
    ) -> Optional[InMemoryKey]:
        """Pick the next entry to drop. Caller holds ``self._lock``."""
        if not self._mem:
            return None
        if strategy == "lru":
            return next(iter(self._mem))  # OrderedDict head = oldest
        if strategy == "lfu":
            return min(self._mem.items(), key=lambda kv: (kv[1].access_count, kv[1].inserted_at))[0]
        # size_weighted: score = size_gb / (rt * (1 + access_count))
        # higher score = first to evict (big + cheap to recompute + rarely used)
        best_key = None
        best_score = -1.0
        for k, e in self._mem.items():
            score = (e.size_bytes / 1e9) / max(0.001, e.recompute_time_s * (1 + e.access_count))
            if score > best_score:
                best_score = score
                best_key = k
        return best_key

    # ------------------------------------------------------------------
    # Disk tier internals
    # ------------------------------------------------------------------

    def _disk_dir(self) -> str:
        """Resolve the cache directory. ``cache.dir`` (None -> error
        when persistence != "off"); ensures dir exists with mode 0o700 (cross-tenant leakage defence).
        """
        d = self._cfg.dir
        if d is None:
            raise RuntimeError(
                "FeatureCache disk tier requires cache.dir to be set when "
                "cache.persistence != 'off'. Set FeatureHandlingConfig("
                "cache=CacheConfig(dir='/path/to/cache', persistence='auto'))."
            )
        d = long_path_safe(os.path.abspath(d))
        os.makedirs(d, mode=0o700, exist_ok=True)
        return d

    def _path_for(self, disk_key: DiskKey) -> str:
        """Resolve the on-disk cache file path for a given ``DiskKey``."""
        return os.path.join(self._disk_dir(), disk_key.filename())

    def _read_from_disk(self, disk_key: DiskKey) -> Optional[Any]:
        """Read and deserialise the disk-tier entry for ``disk_key``, returning None on any miss (missing file or corrupt payload)."""
        path = self._path_for(disk_key)
        # No exists-then-deserialize precheck: the except Exception below already handles missing-file,
        # and a precheck would only reopen a TOCTOU race window between the check and the read.
        allow_pickle = bool(getattr(self._cfg, "allow_pickle", False))
        try:
            return _deserialize(path, allow_pickle=allow_pickle)
        except FileNotFoundError:
            return None
        except CachePickleRefusedError:
            # Surface the refusal up to the caller -- silent miss would
            # hide a security policy decision the operator must see.
            raise
        except Exception as e:  # pragma: no cover
            logger.warning("disk cache read of %s failed: %s; treating as miss", path, e)
            return None

    def _write_to_disk(self, disk_key: DiskKey, value: Any) -> None:
        """Atomically serialise ``value`` to the disk tier; a write failure is logged and swallowed since the in-memory tier already has the value."""
        path = self._path_for(disk_key)
        allow_pickle = bool(getattr(self._cfg, "allow_pickle", False))
        try:
            atomic_write_bytes(path, lambda f: _serialize(value, f, allow_pickle=allow_pickle))
        except CachePickleRefusedError:
            raise
        except Exception as e:  # pragma: no cover
            logger.warning("disk cache write to %s failed: %s; in-memory still populated", path, e)
            return
        # Write the sha256 sidecar so safe_load consumers can verify subsequent reads. Only matters
        # for the pickle-fallback path (allow_pickle=True); the npz path is numeric-only and not
        # an RCE vector, but the sidecar is cheap so we write it unconditionally to keep parity
        # with composite_cache.
        if allow_pickle:
            from mlframe.utils.safe_pickle import write_sidecar as _swrite
            try:
                _swrite(path)
            except OSError as _sc_err:
                logger.debug("FeatureCache sidecar write failed (value written OK): %s", _sc_err)
        self._maybe_evict_disk()

    def _maybe_evict_disk(self) -> None:
        """LRU-by-mtime disk-tier eviction against ``cache.disk_evict_when_free_below_gb`` /
        ``cache.disk_min_free_gb`` -- documented since this module's introduction but never
        implemented, so a long-running suite with disk persistence enabled grew the cache directory
        without bound. Triggered after every disk write (cheap no-op check when free space is
        already comfortably above the threshold); evicts the OLDEST ``.bin`` entries (by mtime,
        since the disk tier has no separate access-order sidecar) until free space is back above
        ``disk_min_free_gb``, or there is nothing left to evict.
        """
        evict_below_gb = getattr(self._cfg, "disk_evict_when_free_below_gb", None)
        if not evict_below_gb:
            return
        try:
            d = self._disk_dir()
            free_bytes = shutil.disk_usage(d).free
        except Exception:
            return
        if free_bytes >= evict_below_gb * 1e9:
            return
        target_free_bytes = max(0.0, getattr(self._cfg, "disk_min_free_gb", 0.0) or 0.0) * 1e9
        try:
            entries = [(os.path.join(d, fn), os.path.getmtime(os.path.join(d, fn))) for fn in os.listdir(d) if fn.endswith(".bin")]
        except Exception as exc:
            logger.warning("FeatureCache disk eviction: failed to list %s: %s", d, exc)
            return
        entries.sort(key=lambda t: t[1])  # oldest mtime first
        n_evicted = 0
        for path, _mtime in entries:
            if free_bytes >= target_free_bytes:
                break
            try:
                size = os.path.getsize(path)
                os.remove(path)
                free_bytes += size
                n_evicted += 1
            except OSError as exc:
                logger.debug("FeatureCache disk eviction: could not remove %s: %s", path, exc)
        if n_evicted:
            with self._lock:
                self._evictions += n_evicted
            logger.info(
                "FeatureCache disk eviction: removed %d stale entry(s) from %s (free space was below %.1f GB)",
                n_evicted, d, evict_below_gb,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Wipe in-memory state. Disk left intact -- use the disk
        eviction routine for that."""
        with self._lock:
            self._mem.clear()
            self._key_xref.clear()

    def purge_by_df_token(self, df_token: int) -> int:
        """Drop every in-memory entry whose ``InMemoryKey.df_token`` matches ``df_token``.

        Must be called from ``_release_ctx_polars_frames`` (or any other code path that releases
        the strong reference to a frame): once Python is free to recycle ``id(frame)``, a future
        ``InMemoryKey`` lookup using the recycled id would silently collide with a cached entry
        that belonged to the dropped frame and replay stale state. The session_id rotation in
        ``reset_session`` already protects across *suite* boundaries, but mid-suite releases (one
        suite, multiple tier transitions) live inside the same session and need this scrub.

        Returns the count of dropped entries (for diagnostic logging).
        """
        with self._lock:
            stale_keys = [k for k in self._mem.keys() if k.df_token == df_token]
            for k in stale_keys:
                self._mem.pop(k, None)
                self._key_xref.pop(k, None)
            return len(stale_keys)


# =====================================================================
# Helpers: size estimator + serialise / deserialise
# =====================================================================


def _default_size_estimator(value: Any) -> int:
    """Best-effort size in bytes. Specialised for numpy arrays and
    scipy.sparse matrices; falls back to ``sys.getsizeof`` (which is
    often wrong for nested objects, but a sane lower bound)."""
    if isinstance(value, np.ndarray):
        return value.nbytes
    try:
        import scipy.sparse as sp
        if sp.issparse(value):
            return int(value.data.nbytes + value.indices.nbytes + value.indptr.nbytes)
    except ImportError:  # pragma: no cover
        pass
    return sys.getsizeof(value)


class CachePickleRefusedError(RuntimeError):
    """Raised when a payload would require pickle (de)serialisation but
    ``CacheConfig.allow_pickle`` is False. Cache directories with write
    access by other principals make ``pickle.load`` an arbitrary-code
    execution vector; we refuse by default and require explicit opt-in.
    """


def _serialize(value: Any, fileobj: Any, *, allow_pickle: bool = False) -> None:
    """Serialise to disk via numpy ``.npz``, dtype
    preserved exactly via ``np.save`` so fp16 stored = fp16 loaded
    (no silent fp32 promotion via pickle pathways).

    When ``allow_pickle=False`` (default) and ``value`` is not ndarray /
    scipy-sparse, refuses rather than falling through to pickle. Set
    ``allow_pickle=True`` via :class:`CacheConfig` only when the cache
    directory is trusted (no concurrent write access by other
    principals)."""
    if isinstance(value, np.ndarray):
        # Encode the "kind" sentinel as a uint8 array of ASCII bytes
        # rather than an object-dtype array -- object dtype requires
        # ``allow_pickle=True`` on the load side, which we now refuse
        # by default for security. ``np.frombuffer`` keeps it numeric.
        kind_bytes = np.frombuffer(b"ndarray", dtype=np.uint8)
        np.savez(fileobj, kind=kind_bytes, value=value)
        return
    try:
        import scipy.sparse as sp
        if sp.issparse(value):
            csr = value.tocsr()
            kind_bytes = np.frombuffer(b"csr", dtype=np.uint8)
            np.savez(
                fileobj,
                kind=kind_bytes,
                data=csr.data,
                indices=csr.indices,
                indptr=csr.indptr,
                shape=np.array(csr.shape, dtype=np.int64),
            )
            return
    except ImportError:  # pragma: no cover
        pass
    if not allow_pickle:
        raise CachePickleRefusedError(
            f"FeatureCache: cannot serialise {type(value).__name__!r} without "
            "pickle, and CacheConfig.allow_pickle is False. Pickle is an RCE "
            "vector on attacker-controlled cache files; set allow_pickle=True "
            "only when the cache directory is trusted."
        )
    # Opt-in pickle path.
    import pickle  # nosec B403 - pickle used only for trusted same-process/dev-local round-trips, see call sites in this file
    pickle.dump(value, fileobj, protocol=5)


def _deserialize(path: str, *, allow_pickle: bool = False) -> Any:
    """Inverse of :func:`_serialize`. Memmap on read for ndarray
    payloads (fp16 memmap saves 5-10 GB peak RAM).

    ``np.load`` is opened with ``allow_pickle=allow_pickle``; the
    pickle-file fallback (legacy from the old write path) is only
    reached when ``allow_pickle=True``. With the default False, a
    pickle-only payload on disk raises :class:`CachePickleRefusedError`
    instead of executing the pickle stream."""
    try:
        # np.load returns an NpzFile wrapping a zipfile + mmap-backed handle
        # when the payload is npz. Not closing it leaks one OS handle per
        # cache read; on Windows this blocks later overwrite/eviction
        # (PermissionError), on Linux it eventually hits EMFILE in long
        # CV/RFECV loops. Use the with-block and materialise arrays via
        # np.array(...) before exiting so the caller gets owned buffers
        # (mmap views would go invalid on close).
        loaded = np.load(path, allow_pickle=allow_pickle, mmap_mode="r")
    except (OSError, zipfile.BadZipFile):
        # OSError (PermissionError from a locked file on Windows, a genuine disk-read failure, ...) and
        # zipfile.BadZipFile (a truncated npz from a crashed write) are real I/O/corruption failures, not
        # a "this is a legacy pickle payload" signal -- let them propagate as themselves instead of being
        # funnelled into an attempted-unpickle-of-non-pickle-bytes fallback below, which would previously
        # raise a confusing UnpicklingError that hides the real underlying error.
        raise
    except Exception:
        if not allow_pickle:
            raise CachePickleRefusedError(
                f"FeatureCache: failed to read {path!r} via numpy and " "CacheConfig.allow_pickle is False; refusing pickle fallback."
            )
        # Pickle fallback: route through safe_pickle.safe_load so a sidecar-digest mismatch
        # surfaces as PickleVerificationError instead of executing tampered bytes. legacy
        # entries written before the sidecar landed have no .sha256, so allow_unverified=True
        # keeps them readable; strict mode is opt-in via MLFRAME_FEATURE_CACHE_STRICT=1.
        from mlframe.utils.safe_pickle import safe_load as _sload
        _strict = os.environ.get("MLFRAME_FEATURE_CACHE_STRICT", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        return _sload(path, allow_unverified=not _strict)
    # ``np.load`` with ``allow_pickle=True`` on a pure-pickle file (no
    # npy / npz magic header) returns the unpickled object directly.
    # NpzFile exposes ``.files``; anything else is already the value.
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        return loaded
    # NpzFile path: materialise then close.
    with loaded as npz:
        kind_arr = npz.get("kind") if "kind" in npz.files else None
        if kind_arr is None:
            # Older format without sentinel -- assume single array stored
            # as 'value'.
            if "value" in npz.files:
                return np.array(npz["value"])
            # fall through to first key
            return np.array(npz[npz.files[0]])
        # ``kind`` is a uint8 ASCII-byte vector (post-audit format) OR a
        # legacy object-dtype length-1 array. Decode both.
        # Handle legacy bytes-typed object arrays too -- `str(b"ndarray") ==
        # "b'ndarray'"` would silently miss the kind== checks below and
        # raise ValueError("unknown serialised kind ...").
        if kind_arr.dtype == np.uint8:
            kind = bytes(kind_arr).decode("ascii")
        else:
            raw = kind_arr[0]
            kind = raw.decode("ascii") if isinstance(raw, (bytes, bytearray)) else str(raw)
        if kind == "ndarray":
            return np.array(npz["value"])
        if kind == "csr":
            try:
                import scipy.sparse as sp
            except ImportError:  # pragma: no cover
                raise
            return sp.csr_matrix(
                (np.array(npz["data"]), np.array(npz["indices"]), np.array(npz["indptr"])),
                shape=tuple(npz["shape"]),
            )
        raise ValueError(f"unknown serialised kind {kind!r} at {path!r}")


__all__ = ["FeatureCache"]
