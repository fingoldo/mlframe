"""Cross-process / cross-run suite-level artefact cache (Wave 8 / A5 architectural).

mlframe today has multiple in-process caches (``_PRE_PIPELINE_CACHE``, ``PipelineCache``, ``DiscoveryCache``, ``MRMR._FIT_CACHE``, ``FeatureCache``) but no single facade that turns deterministic SUITE-LEVEL artefacts (``fit_and_transform_pipeline`` output, ``apply_preprocessing_extensions`` output, ``trainset_features_stats``, dummy baselines, composite target specs) into a cross-process disk cache. PySR symbolic FE alone can take minutes per call; on a CI / multi-worker setup every fresh process pays that full cost.

This module provides ``SuiteArtefactCache`` -- a thin facade over ``joblib.Memory`` (used for its proven bytes-limit + LRU eviction story) keyed by a stable ``SuiteKeyBuilder`` digest that folds ``(df_fingerprint, config_canonical, mlframe_models, lib_versions, random_seed)`` via ``orjson.dumps(..., OPT_SORT_KEYS)`` + blake2b -- mirroring ``_full_y_content_hash`` and ``make_discovery_cache_key`` in spirit.

Critical CLAUDE.md gates honoured:
* default ``bytes_limit=2_000_000_000`` (2 GB) -- the project-wide ceiling for in-RAM cached artefacts. Above this the cache REFUSES to store, returning the caller's computed value untouched so the hot path keeps running. Operators with a different storage budget set ``MLFRAME_SUITE_CACHE_MAX_BYTES`` or pass ``bytes_limit=`` explicitly.
* size_estimate-aware ``put`` -- callers that already know the artefact byte size (frames via ``df.estimated_size("b")`` / ndarrays via ``.nbytes``) pass it in to avoid the cache having to re-walk the structure with ``sys.getsizeof``.
* disk-persistent via ``safe_pickle.safe_dump`` -- every cached artefact gets the same sha256 sidecar fail-closed verification that the rest of the package uses for pickle on disk. Direct ``pickle.dump`` is NOT used here: the surrounding cache directory is attacker-reachable on shared CI hosts.
* ``pyutilz.performance.kernel_tuning.cache``-style env-var override pattern -- no hardcoded thresholds outside the documented defaults, every cap is operator-tunable.

Public surface:
* :class:`SuiteKeyBuilder` -- ``.build(...)`` returns a 32-char hex digest.
* :class:`SuiteArtefactCache` -- ``get`` / ``put`` / ``evict_lru`` / ``total_bytes`` / ``clear``.
* :func:`get_default_cache` -- process-wide lazy singleton honouring env vars.
* :func:`cache_artefact` -- decorator that binds a function to ``get_default_cache()`` and keys per-call from positional / kwarg inputs (mirrors ``functools.lru_cache`` shape but persists to disk).

Wire-in proofs-of-concept (this commit):
* ``mlframe.training.core._phase_helpers_fit_pipeline._cached_fit_and_transform_pipeline`` -- wraps ``fit_and_transform_pipeline``.
* ``mlframe.training.core._phase_helpers_fit_pipeline._cached_apply_preprocessing_extensions`` -- wraps ``apply_preprocessing_extensions``.

Remaining producers (TODO; tracked in pipeline-cache-critique.md A5 architectural proposal):
* ``trainset_features_stats`` (already opt-in via precompute bundle -- finish the disk layer per #1)
* ``composite_target_specs`` (#1, producer side missing in ``_precompute.py``)
* ``dummy_baselines`` (#1)
* ``_PRE_PIPELINE_CACHE`` promoted to disk (#7 + #11 -- needs random_seed + lib_version fold first)
"""
from __future__ import annotations

import functools
import hashlib
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import orjson  # type: ignore[import-not-found]
    _HAS_ORJSON = True
except ImportError:  # pragma: no cover - orjson is a project dep but allow fallback
    import json as _json_fallback
    orjson = None  # type: ignore[assignment]
    _HAS_ORJSON = False

from mlframe.utils.safe_pickle import (
    PickleVerificationError,
    safe_dump as _safe_pickle_dump,
    safe_load as _safe_pickle_load,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SuiteKeyBuilder",
    "SuiteArtefactCache",
    "get_default_cache",
    "cache_artefact",
    "DEFAULT_BYTES_LIMIT",
    "DEFAULT_CACHE_DIR_ENV",
    "DEFAULT_BYTES_LIMIT_ENV",
]

# Project-wide CLAUDE.md ceiling for in-RAM cached artefacts. Above this the cache refuses to store; the value is returned unchanged so the call still completes. Operators on hosts with more storage set ``MLFRAME_SUITE_CACHE_MAX_BYTES`` to override.
DEFAULT_BYTES_LIMIT: int = 2_000_000_000

DEFAULT_CACHE_DIR_ENV: str = "MLFRAME_SUITE_CACHE_DIR"
DEFAULT_BYTES_LIMIT_ENV: str = "MLFRAME_SUITE_CACHE_MAX_BYTES"


def _default_cache_dir() -> str:
    """Operator-overridable cache directory; default ``~/.mlframe/cache/suite/``.

    Honours the same env-var convention as ``pyutilz.performance.kernel_tuning.cache``. The directory is created lazily on first ``put``; ``get`` against a non-existent dir is a clean miss.
    """
    override = os.environ.get(DEFAULT_CACHE_DIR_ENV, "").strip()
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".mlframe", "cache", "suite")


def _default_bytes_limit() -> int:
    raw = os.environ.get(DEFAULT_BYTES_LIMIT_ENV, "").strip()
    if not raw:
        return DEFAULT_BYTES_LIMIT
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "SuiteArtefactCache: %s=%r is not an integer; falling back to default %d bytes",
            DEFAULT_BYTES_LIMIT_ENV, raw, DEFAULT_BYTES_LIMIT,
        )
        return DEFAULT_BYTES_LIMIT


def _canonical_dump(payload: Any) -> bytes:
    """Sort-keyed JSON bytes -- the input to the cache key blake2b. Mirrors the JSON_HASH_MUST_SORT_KEYS rule already enforced in ``compute_config_signature_v1`` / ``_full_y_content_hash``.

    ``orjson`` wins by ~3-5x over stdlib on dict-of-strings payloads (the common config-shape) but falls back to stdlib if absent. Non-string keys are stringified by ``default=str`` so the call NEVER raises on unhashable / non-JSON-serialisable subfields -- a stable digest beats a precise one for cache keys.
    """
    if _HAS_ORJSON:
        return orjson.dumps(payload, default=str, option=orjson.OPT_SORT_KEYS)
    return _json_fallback.dumps(payload, sort_keys=True, default=str).encode("utf-8")


class SuiteKeyBuilder:
    """Builds the stable cache key for a suite-level artefact.

    Pure-static; instances would be wasteful since every method is a function over its inputs. The class wraps the methods only for namespacing + extensibility (subclasses can override the digest size or add a new contributing field without monkey-patching a module-level free function).

    Key contract: a cache hit IS a hit on the EXACT same ``(df_fp, config_canonical, mlframe_models, lib_versions, random_seed)`` tuple. Any drift in any one component lands a new key -- the cost of a wrong-cache-hit on a 100+ GB frame is intolerable, the cost of an extra miss is one cold compute.

    The 32-char hex digest matches the existing ``data_signature`` / ``make_discovery_cache_key`` length so all three caches stay visually consistent in logs.
    """

    DIGEST_SIZE: int = 16  # 16 bytes -> 32-char hex

    @classmethod
    def build(
        cls,
        *,
        df_fp: str,
        config_canonical: Any,
        mlframe_models: Optional[Any] = None,
        lib_versions: Optional[dict] = None,
        random_seed: Optional[int] = None,
        extra: Optional[dict] = None,
    ) -> str:
        """Compose a stable cache key.

        Parameters
        ----------
        df_fp
            Content fingerprint of the input frame -- caller-provided so the cache lookup does NOT re-walk a 100+ GB frame. Use ``mlframe.training.feature_handling.fingerprint.ContentFingerprint.short()`` (project canonical) or any deterministic hex digest.
        config_canonical
            Anything ``orjson.dumps(..., OPT_SORT_KEYS)``-serialisable: a dict, a pydantic ``model_dump(mode="json")`` output, a dataclass ``__dict__``, or a tuple. Falls through ``default=str`` for exotic types so the call never raises.
        mlframe_models
            Set / list / tuple of model names ("cb", "lgb", ...). Sorted by ``frozenset`` before digest so order is irrelevant.
        lib_versions
            ``{lib_name: version_str}`` map; folded to invalidate cache when a key dep (sklearn / polars / lightgbm / catboost) changes major.
        random_seed
            Folded explicitly so two structurally identical configs with different seeds do NOT collide. ``None`` is treated as "seed not specified" (folded literally as the string "None") -- two cache hits sharing ``None`` is fine because they're agreeing the artefact is seed-independent.
        extra
            Free-form dict for caller-specific discriminators (per-target name, per-fold index, etc.).
        """
        payload: dict[str, Any] = {
            "df_fp": str(df_fp),
            "config": config_canonical,
            "random_seed": "None" if random_seed is None else int(random_seed),
        }
        if mlframe_models is not None:
            try:
                payload["models"] = sorted(str(m) for m in mlframe_models)
            except TypeError:
                payload["models"] = repr(mlframe_models)
        if lib_versions:
            payload["versions"] = dict(sorted(lib_versions.items()))
        if extra:
            payload["extra"] = extra
        blob = _canonical_dump(payload)
        return hashlib.blake2b(blob, digest_size=cls.DIGEST_SIZE).hexdigest()


class SuiteArtefactCache:
    """Thin LRU + size-aware facade over a disk directory of sidecar-verified pickles.

    Why not ``joblib.Memory`` directly?
    * ``joblib.Memory`` does NOT verify a sha256 sidecar on load -- the surrounding directory is attacker-reachable on shared CI hosts, so we route every load through :func:`mlframe.utils.safe_pickle.safe_load`.
    * ``joblib.Memory.reduce_size`` exists but operates on its own LRU concept; we want per-key explicit size-gating so caller-provided ``size_estimate`` short-circuits the walk on known-small artefacts.

    Concurrency: in-process thread-safe via ``_lock``. Cross-process safety is delegated to the file-system: ``safe_dump`` writes atomically (open + close + rename); a torn write can never publish under the visible key. Two processes writing the same key race on the rename -- last writer wins, which is fine because the contents are deterministic by construction (same key implies same artefact).

    Bytes budget: a "soft" cap by design. ``put`` refuses entries whose ``size_estimate`` exceeds the budget (the value is returned to the caller unchanged so the hot path keeps running). Total-bytes accounting walks the directory lazily; ``evict_lru`` runs on every ``put`` so the on-disk footprint stays within budget across restarts.
    """

    # Sentinel for ``get`` miss path. ``None`` is a valid cached value (e.g. "no recommendation"), so we distinguish missing-from-cache from cached-as-None.
    _MISS = object()

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        *,
        bytes_limit: int = DEFAULT_BYTES_LIMIT,
        max_entries: Optional[int] = None,
    ) -> None:
        self.cache_dir = cache_dir or _default_cache_dir()
        self.bytes_limit = int(bytes_limit)
        self.max_entries = max_entries
        self._lock = threading.Lock()
        # In-memory LRU access ledger for the eviction order; the on-disk footprint
        # is authoritative for total bytes. The ledger is populated lazily on first
        # access -- a cold cache directory does NOT pay the directory-walk cost at
        # construction time.
        self._lru: "OrderedDict[str, float]" = OrderedDict()
        self._lru_initialised: bool = False

    # ----- internal helpers ----------------------------------------------------

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.pkl")

    def _ensure_dir(self) -> None:
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def _file_size(self, path: str) -> int:
        try:
            return os.path.getsize(path)
        except OSError:
            return 0

    def _init_lru_from_disk(self) -> None:
        """Populate ``_lru`` from on-disk mtimes the first time the cache is touched.

        Called under ``self._lock``. Skips silently if the cache directory does not yet exist -- a cold cache has nothing to initialise.
        """
        if self._lru_initialised:
            return
        try:
            entries = []
            with os.scandir(self.cache_dir) as it:
                for de in it:
                    if not de.is_file() or not de.name.endswith(".pkl"):
                        continue
                    key = de.name[:-4]
                    try:
                        mt = de.stat().st_mtime
                    except OSError:
                        mt = 0.0
                    entries.append((key, mt))
            entries.sort(key=lambda kv: kv[1])
            for k, mt in entries:
                self._lru[k] = mt
        except FileNotFoundError:
            pass
        self._lru_initialised = True

    def _total_bytes_locked(self) -> int:
        # Authoritative byte count from disk; the in-memory LRU only tracks access order. Counts both .pkl and the .pkl.sha256 sidecar so the budget check reflects the FULL on-disk footprint. Pre-fix this summed only .pkl, leaving the per-entry ~64-byte sidecar unaccounted -- at N=1000 entries that's a 64 KB blind spot and the operator's budget check was systematically optimistic.
        total = 0
        try:
            with os.scandir(self.cache_dir) as it:
                for de in it:
                    if not de.is_file():
                        continue
                    if not (de.name.endswith(".pkl") or de.name.endswith(".pkl.sha256")):
                        continue
                    try:
                        total += de.stat().st_size
                    except OSError:
                        pass
        except FileNotFoundError:
            return 0
        return total

    def _evict_lru_locked(self) -> int:
        """Evict oldest entries until under both byte + entry caps. Returns count evicted.

        Both the .pkl and its .pkl.sha256 sidecar are dropped together and BOTH sizes count against ``total`` -- pre-fix the sidecar removal failure path was a silent ``except OSError: pass`` which would leave an orphan sidecar on disk that no later eviction would ever clean up. ``_total_bytes_locked`` also now counts sidecars so the budget figure is honest.
        """
        removed = 0
        # Take a snapshot of LRU keys (oldest first; OrderedDict insertion order = touch order).
        keys_ordered = list(self._lru.keys())
        total = self._total_bytes_locked()
        for key in keys_ordered:
            over_bytes = self.bytes_limit > 0 and total > self.bytes_limit
            over_entries = self.max_entries is not None and len(self._lru) > self.max_entries
            if not over_bytes and not over_entries:
                break
            path = self._path(key)
            sidecar = path + ".sha256"
            pkl_size = self._file_size(path)
            sidecar_size = self._file_size(sidecar)
            try:
                os.remove(path)
            except FileNotFoundError:
                # Already gone (concurrent process / parallel agent). Still try to drop the orphan sidecar below.
                pkl_size = 0
            except OSError as exc:
                logger.debug("SuiteArtefactCache: eviction of %s failed: %s", key, exc)
                break
            # Drop the sha256 sidecar too so a future put of the same key starts fresh AND so the on-disk footprint actually shrinks (orphan sidecars accumulate at ~64-byte each, ignored by every later eviction sweep).
            try:
                os.remove(sidecar)
            except FileNotFoundError:
                sidecar_size = 0
            except OSError as exc:
                # Surface this -- a permission / lock failure here is the canonical source of the "total_bytes reports under cap but on-disk footprint is larger" symptom.
                logger.warning("SuiteArtefactCache: failed to drop sidecar %s during eviction: %s", sidecar, exc)
                sidecar_size = 0
            self._lru.pop(key, None)
            total -= (pkl_size + sidecar_size)
            removed += 1
        return removed

    # ----- public API ---------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Return the cached value, or ``default`` if absent / corrupt / unverified.

        A successful read bumps the in-memory LRU ledger so subsequent ``evict_lru`` picks the least-recently-USED (not just least-recently-WRITTEN) entry. We DELIBERATELY do not bump the disk mtime -- cross-process atime semantics on NTFS are unreliable, and a network-mounted cache directory would pay per-read I/O for no benefit. The in-memory ledger covers single-process LRU correctness; cross-process callers see write-order eviction which is the conservative behaviour.
        """
        path = self._path(key)
        try:
            value = _safe_pickle_load(path)
        except (FileNotFoundError, OSError, EOFError, PickleVerificationError):
            return default
        except Exception as exc:
            logger.debug("SuiteArtefactCache.get %s failed: %s", key, exc)
            return default
        with self._lock:
            self._init_lru_from_disk()
            # move_to_end on existing -- O(1) -- promotes to MRU.
            if key in self._lru:
                self._lru.move_to_end(key)
            else:
                # Disk-only entry we didn't know about (other process wrote it). Record it now.
                try:
                    self._lru[key] = os.path.getmtime(path)
                except OSError:
                    self._lru[key] = 0.0
        return value

    def put(
        self,
        key: str,
        value: Any,
        *,
        size_estimate: Optional[int] = None,
    ) -> bool:
        """Persist ``value`` under ``key`` if size is within budget.

        Returns True on successful store, False on refusal (size_estimate > bytes_limit) or write failure. Callers MUST treat False as "value not cached, but still the correct return value" -- the hot path keeps running with the caller's value.

        ``size_estimate``: pass the upstream-known byte size when available -- ``df.estimated_size('b')`` for polars, ``df.memory_usage(deep=True).sum()`` for pandas, ``arr.nbytes`` for ndarrays. When ``None``, the cache writes the file first and then measures via ``os.path.getsize``; the file is rolled back if it exceeds the budget. This keeps the "size_estimate not known" path correct at the cost of one wasted write -- acceptable because the unknown-size path is the exception, not the rule.
        """
        if size_estimate is not None and size_estimate > self.bytes_limit:
            logger.debug(
                "SuiteArtefactCache: refusing to cache %s (estimated %d bytes > limit %d)",
                key, size_estimate, self.bytes_limit,
            )
            return False
        self._ensure_dir()
        path = self._path(key)
        try:
            _safe_pickle_dump(value, path)
        except OSError as exc:
            logger.warning("SuiteArtefactCache.put %s write failed: %s", key, exc)
            return False
        except Exception as exc:  # pickle failures (unpicklable object): not fatal
            logger.debug("SuiteArtefactCache.put %s pickle failed: %s", key, exc)
            # Best-effort cleanup of any partial file safe_dump may have left behind.
            for p in (path, path + ".sha256"):
                try:
                    os.remove(p)
                except OSError:
                    pass
            return False
        actual_size = self._file_size(path)
        if actual_size > self.bytes_limit:
            # File written but oversized -- roll back; caller still gets the value.
            for p in (path, path + ".sha256"):
                try:
                    os.remove(p)
                except OSError:
                    pass
            logger.debug(
                "SuiteArtefactCache: rolled back oversize entry %s (%d bytes > limit %d)",
                key, actual_size, self.bytes_limit,
            )
            return False
        with self._lock:
            self._init_lru_from_disk()
            self._lru[key] = os.path.getmtime(path) if os.path.exists(path) else 0.0
            self._lru.move_to_end(key)
            self._evict_lru_locked()
        return True

    def evict_lru(self) -> int:
        """Force an eviction sweep. Returns count evicted. Idempotent."""
        with self._lock:
            self._init_lru_from_disk()
            return self._evict_lru_locked()

    def total_bytes(self) -> int:
        """Total on-disk bytes across all cached entries. O(N) directory scan; not cheap, intended for diagnostics / tests rather than the hot path."""
        with self._lock:
            return self._total_bytes_locked()

    def clear(self) -> int:
        """Remove every cached entry. Returns count removed. Used by tests for isolation."""
        with self._lock:
            self._init_lru_from_disk()
            count = 0
            for key in list(self._lru.keys()):
                path = self._path(key)
                for p in (path, path + ".sha256"):
                    try:
                        os.remove(p)
                        if p == path:
                            count += 1
                    except OSError:
                        pass
                self._lru.pop(key, None)
            # Also catch any orphan .pkl that slipped past the LRU ledger.
            try:
                with os.scandir(self.cache_dir) as it:
                    for de in it:
                        if de.is_file() and (de.name.endswith(".pkl") or de.name.endswith(".pkl.sha256")):
                            try:
                                os.remove(de.path)
                                if de.name.endswith(".pkl"):
                                    count += 1
                            except OSError:
                                pass
            except FileNotFoundError:
                pass
            return count

    def __contains__(self, key: str) -> bool:
        return os.path.exists(self._path(key))

    def __len__(self) -> int:
        with self._lock:
            self._init_lru_from_disk()
            return len(self._lru)


# ----- process-wide singleton + decorator -------------------------------------

_DEFAULT_CACHE_LOCK = threading.Lock()
_DEFAULT_CACHE: Optional[SuiteArtefactCache] = None


def get_default_cache() -> SuiteArtefactCache:
    """Lazy process-wide singleton; honours env vars at FIRST construction.

    Subsequent env-var changes do NOT rebuild the singleton -- pass an explicit ``SuiteArtefactCache(...)`` to overrides scoped to one call site. Tests that need to relocate the cache use ``set_default_cache`` (below) which is explicit about the reseat.
    """
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is not None:
        return _DEFAULT_CACHE
    with _DEFAULT_CACHE_LOCK:
        if _DEFAULT_CACHE is None:
            _DEFAULT_CACHE = SuiteArtefactCache(
                cache_dir=_default_cache_dir(),
                bytes_limit=_default_bytes_limit(),
            )
    return _DEFAULT_CACHE


def set_default_cache(cache: Optional[SuiteArtefactCache]) -> None:
    """Replace (or clear) the process-wide singleton. Tests only; production code should pass an explicit ``SuiteArtefactCache`` to its callees rather than mutate the singleton."""
    global _DEFAULT_CACHE
    with _DEFAULT_CACHE_LOCK:
        _DEFAULT_CACHE = cache


def cache_artefact(
    name: str,
    *,
    cache: Optional[SuiteArtefactCache] = None,
    key_builder: Optional[Callable[..., str]] = None,
    size_estimate: Optional[Callable[[Any], int]] = None,
) -> Callable:
    """Decorator wrapping a function with disk-backed memoisation through a :class:`SuiteArtefactCache`.

    Parameters
    ----------
    name
        Stable artefact name (folded into the cache key alongside the call-args digest). Two functions cached under the same ``name`` share the cache slot -- by design, so a refactored ``foo_v2`` can serve cached ``foo_v1`` artefacts when their signatures match.
    cache
        Cache instance; defaults to :func:`get_default_cache`. Tests pass an isolated ``SuiteArtefactCache(tmp_path)``.
    key_builder
        Function receiving ``(*args, **kwargs)`` and returning a hex digest. Defaults to ``SuiteKeyBuilder.build(df_fp=blake2b(repr(args)+repr(kwargs)), config_canonical={'name': name})`` which is the safe-but-coarse fallback -- callers that want fine-grained cache hits provide a tailored builder that pulls ``ContentFingerprint.short()`` from the actual frame arg.
    size_estimate
        Function receiving the function's RETURN value and returning a byte size. Defaults to ``None`` (cache writes-then-checks; see :meth:`SuiteArtefactCache.put`).

    The decorator preserves the function signature via ``functools.wraps`` so callers see the original docstring / argspec.
    """

    def _default_key_builder(*args: Any, **kwargs: Any) -> str:
        # Coarse fallback: hash the args/kwargs repr. Adequate for cheap-to-rebuild artefacts; callers needing precise content-keying provide their own builder.
        try:
            blob = _canonical_dump({"args": [repr(a) for a in args], "kwargs": {k: repr(v) for k, v in sorted(kwargs.items())}})
        except Exception:
            blob = repr((args, sorted(kwargs.items()))).encode("utf-8")
        df_fp = hashlib.blake2b(blob, digest_size=SuiteKeyBuilder.DIGEST_SIZE).hexdigest()
        return SuiteKeyBuilder.build(df_fp=df_fp, config_canonical={"name": name})

    def _decorator(fn: Callable) -> Callable:
        kb = key_builder or _default_key_builder

        @functools.wraps(fn)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            c = cache if cache is not None else get_default_cache()
            try:
                key = kb(*args, **kwargs)
            except Exception as exc:
                # Key-builder failure: degrade to no-cache rather than crash the caller's hot path.
                logger.debug("cache_artefact(%s): key_builder failed (%s); bypassing cache", name, exc)
                return fn(*args, **kwargs)
            # Use the sentinel so cached-None values are distinguishable from a miss.
            cached = c.get(key, default=SuiteArtefactCache._MISS)
            if cached is not SuiteArtefactCache._MISS:
                return cached
            value = fn(*args, **kwargs)
            est = None
            if size_estimate is not None:
                try:
                    est = int(size_estimate(value))
                except Exception:
                    est = None
            try:
                c.put(key, value, size_estimate=est)
            except Exception as exc:  # cache.put failure must NEVER break the caller
                logger.debug("cache_artefact(%s): cache.put failed (%s)", name, exc)
            return value

        _wrapper.__wrapped_artefact_name__ = name  # type: ignore[attr-defined]
        return _wrapper

    return _decorator
