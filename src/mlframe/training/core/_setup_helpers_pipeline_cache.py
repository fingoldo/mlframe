"""Polars-ds Pipeline JSON-roundtrip cache for ``_setup_helpers``.

Carved from ``_setup_helpers.py`` to keep the parent below the LOC
budget. The module-level globals (``_PIPELINE_JSON_ROUNDTRIP_CACHE``,
``_PIPELINE_JSON_DISK_CACHE_PATH``, ``_PIPELINE_JSON_DISK_CACHE_LOADED``)
live here and the parent re-exports them; tests that monkeypatch the
parent facade are forwarded back here at call time so behaviour stays
identical to the pre-split monolith.
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._training_context import TrainingContext  # noqa: F401


_PARENT_MODULE = "mlframe.training.core._setup_helpers"


def pipeline_json_cache_key(pipeline_json: str) -> str:
    """Stable, content-only cache key for a pipeline ``to_json()`` payload.

    The builtin ``hash()`` is salted per process by PYTHONHASHSEED, so keying the
    disk cache on it produced a 100% cross-process miss -- defeating the whole point
    of persisting the round-trip verdict. blake2b of the UTF-8 bytes depends only on
    the content, so every process derives the same key for the same JSON.
    """
    return hashlib.blake2b(pipeline_json.encode("utf-8"), digest_size=16).hexdigest()


def _parent_attr(name: str, default):
    """Resolve a state attribute from the parent facade when loaded; fall back to local module global.

    Pre-split callers monkeypatched globals on the parent module (e.g.
    ``monkeypatch.setattr(sh, "_PIPELINE_JSON_DISK_CACHE_PATH", path)``).
    After the carve, the source of truth lives in this sibling; the helper
    routes reads through the parent so those tests stay green without
    forcing every caller to update.
    """
    parent = sys.modules.get(_PARENT_MODULE)
    if parent is not None and hasattr(parent, name):
        return getattr(parent, name)
    return default


def _parent_set(name: str, value) -> None:
    parent = sys.modules.get(_PARENT_MODULE)
    if parent is not None:
        setattr(parent, name, value)


# iter193 (2026-05-23): per-process cache for the polars-ds Pipeline
# from_json() roundtrip validation. ``_finalize_and_save_metadata`` calls
# ``Pipeline.from_json(_js)`` on EVERY fit to verify the JSON roundtrips;
# c0141 profile attributed 5.413s wall to this single call. The validation
# is deterministic in the input JSON (same JSON -> same parse result), so
# we cache the result keyed by ``hash(_js)``. First-fit pays the 5s, subse-
# quent fits within the same process do a 1us dict lookup. Mirrors the
# _PROBE_PRECISION_CACHE (iter181), _CB_GPU_USABLE_CACHE, and
# _mlframe_callback_cache_installed (iter189) patterns for process-stable
# costs. Falls back to live validation on cache miss.
_PIPELINE_JSON_ROUNDTRIP_CACHE: dict[int, bool] = {}


# iter275 (2026-05-23): cross-process file cache. The in-memory cache above
# only helps repeated fits within ONE process. Fresh-process workflows
# (fuzz combo re-runs, pytest-xdist workers, CI, dev iteration) pay the
# full 8.5s validation on every first call. c0141 iter275 profile
# attributed 8.57s to a single ``from_json`` (20 polars Exprs x 428ms each).
# Across 150 fuzz combos that's ~21 min wasted on deterministic work.
#
# The file cache is keyed by ``hash(_js) + polars_ds version + polars
# version``; a polars-ds upgrade invalidates the whole cache (different
# version tag) so a wheel that suddenly fails roundtrip on previously-OK
# JSONs surfaces immediately rather than silently using stale "safe"
# verdicts. Single-file JSON layout for simple atomic-replace; cap at 1000
# entries with FIFO eviction so the file stays small + readable.
_PIPELINE_JSON_DISK_CACHE_PATH: str | None = None
_PIPELINE_JSON_DISK_CACHE_LOADED: bool = False
_PIPELINE_JSON_DISK_CACHE_MAX_ENTRIES: int = 1000


def _pipeline_disk_cache_path() -> str:
    """Resolve the per-version disk cache path on first use.

    Honour an out-of-band override set on the parent facade
    (tests / debug tooling do ``monkeypatch.setattr(parent,
    "_PIPELINE_JSON_DISK_CACHE_PATH", path)``); otherwise initialise the
    sibling global lazily under the system temp dir.
    """
    override = _parent_attr("_PIPELINE_JSON_DISK_CACHE_PATH", None)
    if override is not None:
        return override
    global _PIPELINE_JSON_DISK_CACHE_PATH
    if _PIPELINE_JSON_DISK_CACHE_PATH is not None:
        _parent_set("_PIPELINE_JSON_DISK_CACHE_PATH", _PIPELINE_JSON_DISK_CACHE_PATH)
        return _PIPELINE_JSON_DISK_CACHE_PATH
    import tempfile
    cache_dir = os.path.join(tempfile.gettempdir(), "mlframe_caches")
    os.makedirs(cache_dir, exist_ok=True)
    _PIPELINE_JSON_DISK_CACHE_PATH = os.path.join(cache_dir, "polars_ds_pipeline_roundtrip.json")
    _parent_set("_PIPELINE_JSON_DISK_CACHE_PATH", _PIPELINE_JSON_DISK_CACHE_PATH)
    return _PIPELINE_JSON_DISK_CACHE_PATH


def _pipeline_disk_cache_version_tag() -> str:
    """Cache tag derived from the polars-ds + polars wheel versions.

    A wheel upgrade invalidates every prior verdict by changing the tag,
    so a wheel that newly fails roundtrip on previously-OK JSONs does NOT
    silently inherit a stale 'safe' verdict. Best-effort: when version
    introspection fails, return ``"unknown"`` (cache still works within
    one process; cross-process lookups just collide on the unknown tag).
    """
    parts = []
    for mod_name in ("polars_ds", "polars"):
        try:
            mod = __import__(mod_name)
            parts.append(f"{mod_name}={getattr(mod, '__version__', 'unknown')}")
        except Exception:
            parts.append(f"{mod_name}=unknown")
    return "|".join(parts)


def _load_pipeline_disk_cache_into_memory() -> None:
    """One-shot disk -> in-memory rehydrate; safe to call repeatedly."""
    global _PIPELINE_JSON_DISK_CACHE_LOADED
    # Re-read the loaded flag from the parent facade in case a test reset it via monkeypatch.
    if _parent_attr("_PIPELINE_JSON_DISK_CACHE_LOADED", _PIPELINE_JSON_DISK_CACHE_LOADED):
        _PIPELINE_JSON_DISK_CACHE_LOADED = True
        return
    _PIPELINE_JSON_DISK_CACHE_LOADED = True  # set early so partial failures don't retry
    _parent_set("_PIPELINE_JSON_DISK_CACHE_LOADED", True)
    path = _pipeline_disk_cache_path()
    if not os.path.exists(path):
        return
    try:
        try:
            import orjson as _orjson  # 3-10x faster on the hot read path
            with open(path, "rb") as fh:
                data = _orjson.loads(fh.read())
        except ImportError:
            import json as _json
            with open(path, "r", encoding="utf-8") as fh:
                data = _json.load(fh)
    except Exception:
        return  # corrupt file: ignore, will be overwritten on next save
    if not isinstance(data, dict) or data.get("version_tag") != _pipeline_disk_cache_version_tag():
        return  # version mismatch: ignore stale entries
    entries = data.get("entries")
    if not isinstance(entries, dict):
        return
    for hash_str, ok in entries.items():
        try:
            _PIPELINE_JSON_ROUNDTRIP_CACHE[str(hash_str)] = bool(ok)
        except (ValueError, TypeError):
            continue  # skip malformed entries


def _persist_pipeline_disk_cache() -> None:
    """Snapshot the in-memory cache to disk via atomic replace.

    Best-effort: any IO / serialization failure is swallowed so the live
    training path is never broken by cache persistence trouble. FIFO-cap
    keeps the on-disk file small enough to read in microseconds.
    """
    try:
        path = _pipeline_disk_cache_path()
        # Defensive: monkeypatched paths in tests (and prod first-fit on a
        # fresh box) may point at a directory the producer hasn't created yet.
        # ``os.replace`` raises ``FileNotFoundError`` when the dir is missing,
        # which our outer ``except Exception`` swallows -- leaving callers to
        # wonder why ``persist + reload`` round-trips return nothing. One-line
        # ``makedirs`` keeps the contract that ``persist`` always lands when
        # the parent ``cache_file`` is in a writeable location.
        _parent_dir = os.path.dirname(path)
        if _parent_dir:
            os.makedirs(_parent_dir, exist_ok=True)
        entries = _PIPELINE_JSON_ROUNDTRIP_CACHE
        if len(entries) > _PIPELINE_JSON_DISK_CACHE_MAX_ENTRIES:
            # FIFO eviction: keep the most recent N (dict preserves insertion order)
            keep_keys = list(entries.keys())[-_PIPELINE_JSON_DISK_CACHE_MAX_ENTRIES:]
            entries = {k: entries[k] for k in keep_keys}
        payload = {
            "version_tag": _pipeline_disk_cache_version_tag(),
            "entries": {str(k): bool(v) for k, v in entries.items()},
        }
        tmp_path = path + ".tmp"
        try:
            import orjson as _orjson
            with open(tmp_path, "wb") as fh:
                fh.write(_orjson.dumps(payload))
        except ImportError:
            import json as _json
            with open(tmp_path, "w", encoding="utf-8") as fh:
                _json.dump(payload, fh)
        os.replace(tmp_path, path)
    except Exception:
        pass


class _PolarsDsPipelineJsonProxy:
    """Pickle-fast wrapper around a polars-ds ``Pipeline``.

    Default pickle of a polars-ds Pipeline descends through every internal
    ``pl.Expr``. On complex pipelines (Yeo-Johnson + dim_reducer + ordinal
    encoding over many categories) the polars Rust deserializer can take
    100-200ms PER expression during load; 19 such expressions produced a
    2.21s load wall on the 100k binary_classification x lgb profile
    (seed=20260522, 2026-05-19). The proxy serialises via the Pipeline's
    own ``to_json()`` API on save and reconstructs via ``from_json()`` on
    load -- ~0.2ms regardless of expression complexity, ~5000x faster on
    the worst case. Transparent ``transform()`` + attribute forwarding
    means consumers that pulled ``metadata["pipeline"]`` keep working
    unchanged.
    """
    __slots__ = ("_pipeline",)

    def __init__(self, pipeline):
        self._pipeline = pipeline

    @property
    def pipeline(self):
        return self._pipeline

    def __reduce__(self):
        # Use ``__reduce__`` instead of __getstate__/__setstate__ so the
        # reconstruction goes through ``_polars_ds_pipeline_from_json`` which
        # imports polars_ds lazily on the load process (predict-time
        # consumers may not have polars-ds installed; the import-error path
        # then falls through with the JSON string carried as a string blob
        # for late reconstruction).
        return (_polars_ds_pipeline_from_json, (self._pipeline.to_json(),))

    def transform(self, df):
        return self._pipeline.transform(df)

    def __getattr__(self, name):
        # Transparent forwarding so existing code paths that touch attributes
        # like ``.feature_names_out`` / ``.steps`` / ``.blueprint`` keep
        # working. ``__getattr__`` (not ``__getattribute__``) only fires on
        # attributes the proxy itself doesn't define, so the
        # ``self._pipeline`` field stays accessible without recursion.
        return getattr(self._pipeline, name)

    def __repr__(self):
        return f"_PolarsDsPipelineJsonProxy(pipeline={self._pipeline!r})"


def _polars_ds_pipeline_from_json(json_str):
    """Module-level reconstructor used by ``_PolarsDsPipelineJsonProxy.__reduce__``.

    Lives at module scope so it is picklable (``__reduce__`` callables must
    be discoverable at import time, not nested in a class body).
    """
    from polars_ds.pipeline import Pipeline as _PdsPipeline
    return _PolarsDsPipelineJsonProxy(_PdsPipeline.from_json(json_str))
