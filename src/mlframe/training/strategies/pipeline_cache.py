"""``PipelineCache`` + byte-budget helpers."""
from __future__ import annotations

import logging
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("mlframe.training.strategies")


_DEFAULT_PIPELINE_CACHE_BYTES_LIMIT = 2_000_000_000


_DEFAULT_PIPELINE_CACHE_RAM_FRACTION = 0.15


def _resolve_pipeline_cache_budget(fraction: Optional[float] = None) -> int:
    """Decide the PipelineCache byte budget for this process.

    Priority:
      1. ``MLFRAME_PIPELINE_CACHE_BYTES_LIMIT`` env var (absolute override).
      2. A FRACTION of TOTAL host RAM (``fraction`` arg >
         ``MLFRAME_PIPELINE_CACHE_RAM_FRACTION`` env > 0.15 default), then
         clamped to currently-available RAM minus a 16 GB floor so the cache
         never grabs more than is actually free RIGHT NOW. Default 0.15 (was
         0.4): on hosts where training itself takes 60-80% of RAM, the prior
         0.4 fraction (54 GB cap on a 137 GB host) collided with a transient
         peak during composite-discovery + pipeline-fit and triggered OOM.
         0.15 keeps the cache useful (~20 GB on a 137 GB host) without
         starving training; users with cache-bound workloads (re-fit-heavy
         multi-model suites on small data) can opt up via
         ``train_mlframe_models_suite(pipeline_cache_ram_budget_fraction=...)``.
      3. Fallback ``_DEFAULT_PIPELINE_CACHE_BYTES_LIMIT`` if psutil is
         unavailable or raises.
    """
    env_raw = os.environ.get("MLFRAME_PIPELINE_CACHE_BYTES_LIMIT")
    if env_raw:
        try:
            return int(env_raw)
        except (TypeError, ValueError):
            pass
    if fraction is None:
        _frac_env = os.environ.get("MLFRAME_PIPELINE_CACHE_RAM_FRACTION")
        if _frac_env:
            try:
                fraction = float(_frac_env)
            except (TypeError, ValueError):
                fraction = None
    if fraction is None:
        fraction = _DEFAULT_PIPELINE_CACHE_RAM_FRACTION
    # Clamp the fraction to a sane band (0 disables caching effectively; >0.9
    # would starve training).
    fraction = min(0.9, max(0.0, float(fraction)))
    try:
        import psutil as _psutil
        vm = _psutil.virtual_memory()
        total = int(vm.total)
        available = int(vm.available)
        # 16 GB floor (was 4 GB): on big-frame workloads (4M+ rows x 500+ cols)
        # the next transient step (polars->pandas materialise, composite-discovery
        # leak-corr matrix, mini-HPT analyzers) can easily allocate 10-20 GB
        # transient. A 4 GB floor let the cache budget approach available RAM and
        # the next transient step OOM'd the process.
        floor = 16 * 1024 * 1024 * 1024
        budget = int(total * fraction)
        # Never exceed what is free right now (minus the headroom floor):
        # training frames + transient transforms must still fit.
        budget = min(budget, max(0, available - floor))
        budget = max(2 * 1024 * 1024 * 1024, budget)
        return int(budget)
    except Exception:
        return _DEFAULT_PIPELINE_CACHE_BYTES_LIMIT


def _estimate_slot_nbytes(slot: Any) -> int:
    """Best-effort byte-size estimate for a cached frame / array slot.

    Returns 0 for None. Recognises pandas DataFrame (``memory_usage(deep=False).sum()``), polars DataFrame (``estimated_size()``), numpy arrays (``nbytes``), and falls back to ``sys.getsizeof`` for anything else. ``deep=False`` on pandas avoids walking object-dtype columns (which can be 10-100x slower than the buffer-only sum) -- the cache-size signal is needed on the hot insert path and the precision loss on object columns is acceptable for an LRU gate.
    """
    if slot is None:
        return 0
    try:
        nbytes_attr = getattr(slot, "nbytes", None)
        if isinstance(nbytes_attr, int):
            return nbytes_attr
    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
        logger.debug("suppressed in pipeline_cache.py:88: %s", e)
        pass
    try:
        import pandas as _pd
        if isinstance(slot, _pd.DataFrame):
            return int(slot.memory_usage(deep=False).sum())
        if isinstance(slot, _pd.Series):
            return int(slot.memory_usage(deep=False))
    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
        logger.debug("suppressed in pipeline_cache.py:96: %s", e)
        pass
    try:
        import polars as _pl
        if isinstance(slot, _pl.DataFrame):
            return int(slot.estimated_size())
    except Exception:  # nosec B110 - optional dependency import guard
        pass
    try:
        return int(sys.getsizeof(slot))
    except Exception:
        return 0


def _estimate_entry_nbytes(entry: Tuple[Any, Any, Any]) -> int:
    return sum(_estimate_slot_nbytes(s) for s in entry)


class PipelineCache:
    """Bounded LRU cache for transformed DataFrames.

    Different model types that require the same preprocessing can share cached DataFrames, improving efficiency when training multiple models.

    Size discipline (CLAUDE.md "Caching and batching: use both, but never assume a frame fits in RAM"): the cache enforces a byte-budget (default 2 GB; override via ``MLFRAME_PIPELINE_CACHE_BYTES_LIMIT`` env var). On insert overflow, least-recently-used entries are evicted until under the cap; each ``get`` promotes the touched key to most-recently-used. Per-entry size is estimated via ``_estimate_slot_nbytes`` (pandas ``memory_usage``, polars ``estimated_size``, numpy ``nbytes``, ``sys.getsizeof`` fallback) so the cap reflects actual buffer occupancy, not container overhead.

    Not thread-safe; designed for sequential use within a single training run.
    """

    def __init__(self, verbose: bool = True, bytes_limit: Optional[int] = None):
        """Construct a pre-pipeline cache.

        ``verbose=True`` is the new default: HIT/MISS lines are emitted at ``logger.info`` and routinely needed when triaging "why-did-this-suite-re-fit" tickets. The lines are throttled by the per-call HIT vs MISS branch (one log per get) and add no measurable overhead vs the dict lookup itself, so the cost of leaving them on by default is negligible against the diagnostic value of having them already on when the operator wants them. Pass ``verbose=False`` to silence in tight unit-test loops.

        ``bytes_limit=None`` (default) reads ``MLFRAME_PIPELINE_CACHE_BYTES_LIMIT`` from env, falling back to 2_000_000_000 (2 GB). Pass an explicit int to override per-instance (useful in tests).
        """
        # OrderedDict so ``move_to_end`` (LRU promotion on get) and ``popitem(last=False)`` (LRU eviction on overflow) are explicit. Plain dict happens to preserve insertion order in CPython 3.7+ but the LRU contract demands the explicit type.
        self._cache: "OrderedDict[str, Tuple[Any, Any, Any]]" = OrderedDict()
        self._entry_sizes: Dict[str, int] = {}
        self._total_bytes: int = 0
        self.n_hits: int = 0
        self.n_misses: int = 0
        self.n_evicted: int = 0
        self.verbose: bool = bool(verbose)
        if bytes_limit is None:
            bytes_limit = _resolve_pipeline_cache_budget()
        self._bytes_limit: int = int(bytes_limit)

    def get(self, cache_key: str) -> Optional[Tuple[Any, Any, Any]]:
        """Get cached DataFrames for a cache key; promote to MRU on hit."""
        val = self._cache.get(cache_key)
        if val is None:
            self.n_misses += 1
            if self.verbose:
                logger.info("PipelineCache MISS key=%s (hits=%d misses=%d size=%d)", cache_key, self.n_hits, self.n_misses, len(self._cache))
        else:
            # LRU promotion: the just-accessed key moves to the end of the OrderedDict, so the front always contains the genuinely least-recently-used candidates for the next eviction.
            self._cache.move_to_end(cache_key)
            self.n_hits += 1
            if self.verbose:
                logger.info("PipelineCache HIT  key=%s (hits=%d misses=%d size=%d)", cache_key, self.n_hits, self.n_misses, len(self._cache))
        return val

    def set(self, cache_key: str, train_df: Any, val_df: Any, test_df: Any) -> None:
        """Cache transformed DataFrames; evict LRU entries if the byte budget is exceeded.

        The actively-inserted key is NEVER evicted (an oversized single entry stays in cache; eviction stops at ``len(self._cache) == 1`` even if still over budget) -- pinning the freshly-set key preserves the suite's progress; the operator can raise ``MLFRAME_PIPELINE_CACHE_BYTES_LIMIT`` to accommodate.
        """
        entry = (train_df, val_df, test_df)
        # Replace-in-place semantics: if the key already exists, subtract its old size first so the total accounting stays consistent.
        if cache_key in self._cache:
            self._total_bytes -= self._entry_sizes.get(cache_key, 0)
        entry_size = _estimate_entry_nbytes(entry)
        self._cache[cache_key] = entry
        self._cache.move_to_end(cache_key)
        self._entry_sizes[cache_key] = entry_size
        self._total_bytes += entry_size
        if self.verbose:
            logger.info("PipelineCache SET  key=%s size=%d total=%d/%d (size=%d)", cache_key, entry_size, self._total_bytes, self._bytes_limit, len(self._cache))
        self._evict_until_under_limit(active_key=cache_key)

    def _evict_until_under_limit(self, *, active_key: Optional[str]) -> None:
        """LRU-evict entries until total bytes <= limit; never evict ``active_key`` (the just-inserted entry stays pinned even if a single oversized entry exceeds the cap on its own).

        Re-checks current available RAM on every call: if free memory has
        dropped since init (other processes loaded), the effective limit
        tightens dynamically so the cache never pushes the system into
        swap. Operator-set ``MLFRAME_PIPELINE_CACHE_BYTES_LIMIT`` is
        respected as an upper bound.
        """
        try:
            self._bytes_limit = min(self._bytes_limit, _resolve_pipeline_cache_budget())
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed in pipeline_cache.py:187: %s", e)
            pass
        if self._total_bytes <= self._bytes_limit:
            return
        evicted_count = 0
        bytes_freed = 0
        # Iterate over a snapshot of keys in LRU-first order; OrderedDict iteration preserves insertion order so the front is genuinely the LRU.
        for key in list(self._cache.keys()):
            if self._total_bytes <= self._bytes_limit:
                break
            if key == active_key:
                continue
            slot_bytes = self._entry_sizes.pop(key, 0)
            self._cache.pop(key, None)
            self._total_bytes -= slot_bytes
            bytes_freed += slot_bytes
            evicted_count += 1
        if evicted_count:
            self.n_evicted += evicted_count
            logger.info(
                "PipelineCache evicted %d entries freeing %d bytes (now %d/%d bytes, %d entries)",
                evicted_count, bytes_freed, self._total_bytes, self._bytes_limit, len(self._cache),
            )

    def has(self, cache_key: str) -> bool:
        """Check if a cache key exists (does NOT promote to MRU)."""
        return cache_key in self._cache

    def clear(self) -> None:
        """Clear all cached DataFrames and reset the byte-budget accounting."""
        self._cache.clear()
        self._entry_sizes.clear()
        self._total_bytes = 0

    def cache_size_bytes(self) -> int:
        """Total estimated byte-size across every cached frame slot, summed at insert time and maintained on get/set/evict.

        Returns the running ``_total_bytes`` accumulator (the LRU eviction policy keeps it bounded under ``_bytes_limit``) so the value reflects actual buffer occupancy as estimated by ``_estimate_slot_nbytes`` (pandas ``memory_usage(deep=False)``, polars ``estimated_size``, numpy ``nbytes``, ``sys.getsizeof`` fallback) so the cap reflects actual buffer occupancy, not container overhead.
        """
        return int(self._total_bytes)

    def __repr__(self) -> str:
        return f"PipelineCache(keys={len(self._cache)}, hits={self.n_hits}, misses={self.n_misses})"
