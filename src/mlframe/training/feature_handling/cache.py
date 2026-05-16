"""
Multi-tier feature-handling cache.

User-confirmed architecture (round-3 simplification):

* **In-memory tier**: cheap session-keyed lookup. Within one
  ``train_mlframe_models_suite()`` call, ``id(train_df)`` is safe
  (we hold a strong ref) so no content hashing is done in hot path.
  This avoids the 30s/fit overhead from arrow-buffer materialisation
  the round-2 perf agent flagged.

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
  * Disk: LRU within ``cache.disk_evict_when_free_below_gb`` /
    ``cache.disk_min_free_gb`` thresholds.

Eviction strategies:
  * "lru" -- straight LRU (default for in-memory).
  * "size_weighted" -- score = ``size_gb / (recompute_time_s × (1 +
    access_count))``; prefers evicting big-cheap entries (round-3 A6).
  * "lfu" -- least-frequently-used; ties broken by LRU.

Phase D ships the foundation. Vocab caches (TF-IDF / cat codebook)
in phase D.4 sit on top of this primitive.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
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
    tier comes later (round-3 plan) but isn't needed for the v1
    solo-greenfield workload.

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

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
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
                "hit_rate": (
                    (self._hits_mem + self._hits_disk)
                    / max(1, self._hits_mem + self._hits_disk + self._misses)
                ),
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
                self._insert_in_memory(in_mem_key, disk_value, recompute_time_s=0.5,
                                        size_estimator=size_estimator)
                self._key_xref[in_mem_key] = disk_key
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
        self._insert_in_memory(
            in_mem_key, value, recompute_time_s=recompute_time_s,
            size_estimator=size_estimator,
        )
        if self._cfg.persistence in ("auto", "read_write") and disk_key is not None:
            self._write_to_disk(disk_key, value)
            self._key_xref[in_mem_key] = disk_key

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
    ) -> None:
        size = (size_estimator or _default_size_estimator)(value)
        entry = _CacheEntry(
            value=value,
            size_bytes=size,
            recompute_time_s=recompute_time_s,
        )
        with self._lock:
            self._mem[key] = entry
            self._mem.move_to_end(key)
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
        # size_weighted (round-3 A6): score = size_gb / (rt * (1 + access_count))
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
        when persistence != "off"); ensures dir exists with mode 0o700.
        Round-3 S11: cross-tenant leakage defence.
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
        return os.path.join(self._disk_dir(), disk_key.filename())

    def _read_from_disk(self, disk_key: DiskKey) -> Optional[Any]:
        path = self._path_for(disk_key)
        if not os.path.exists(path):
            return None
        try:
            return _deserialize(path)
        except Exception as e:  # pragma: no cover
            logger.warning("disk cache read of %s failed: %s; treating as miss", path, e)
            return None

    def _write_to_disk(self, disk_key: DiskKey, value: Any) -> None:
        path = self._path_for(disk_key)
        try:
            atomic_write_bytes(path, lambda f: _serialize(value, f))
        except Exception as e:  # pragma: no cover
            logger.warning("disk cache write to %s failed: %s; in-memory still populated", path, e)

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


def _serialize(value: Any, fileobj: Any) -> None:
    """Serialise to disk via numpy ``.npz``. Round-3 R3-10: dtype
    preserved exactly via ``np.save`` so fp16 stored = fp16 loaded
    (no silent fp32 promotion via pickle pathways)."""
    if isinstance(value, np.ndarray):
        # Wrap a dict so we can detect the format on load.
        np.savez(fileobj, kind=np.array(["ndarray"], dtype=object), value=value)
        return
    try:
        import scipy.sparse as sp
        if sp.issparse(value):
            csr = value.tocsr()
            np.savez(
                fileobj,
                kind=np.array(["csr"], dtype=object),
                data=csr.data,
                indices=csr.indices,
                indptr=csr.indptr,
                shape=np.array(csr.shape, dtype=np.int64),
            )
            return
    except ImportError:  # pragma: no cover
        pass
    # Last resort: pickle. Round-3 chaos C24 says joblib pickle is
    # fragile, but for a cache miss we'd rather have it work than
    # always fail.
    import pickle
    pickle.dump(value, fileobj, protocol=5)


def _deserialize(path: str) -> Any:
    """Inverse of :func:`_serialize`. Memmap on read for ndarray
    payloads (round-3 P8: fp16 memmap saves 5-10 GB peak RAM)."""
    # Open .npz once -- numpy figures out the format from header.
    try:
        npz = np.load(path, allow_pickle=True, mmap_mode="r")
    except Exception:
        # Could be a pickle file from the last-resort path above.
        with open(path, "rb") as f:
            import pickle
            return pickle.load(f)
    kind_arr = npz.get("kind") if "kind" in npz.files else None
    if kind_arr is None:
        # Older format without sentinel -- assume single array stored
        # as 'value'.
        if "value" in npz.files:
            return npz["value"]
        # fall through to first key
        return npz[npz.files[0]]
    kind = str(kind_arr[0])
    if kind == "ndarray":
        return npz["value"]
    if kind == "csr":
        try:
            import scipy.sparse as sp
        except ImportError:  # pragma: no cover
            raise
        return sp.csr_matrix((npz["data"], npz["indices"], npz["indptr"]), shape=tuple(npz["shape"]))
    raise ValueError(f"unknown serialised kind {kind!r} at {path!r}")


__all__ = ["FeatureCache"]
