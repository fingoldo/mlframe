"""Regression sensors for the cache.py FUTURE items of the 2026-06-10 composite audit.

- S25: ``compute_config_signature_v1`` UNCONDITIONALLY folds the on-disk schema
  epoch + ``mlframe.__version__`` into every config signature. Pre-fix the
  version component was opt-in (only present when ``library_versions`` was
  passed), so a direct R&D caller following the module recipe built keys with
  no code-version component and replayed stale specs across mlframe upgrades.
- S18: eviction byte accounting counts the ``.pkl.sha256`` sidecar (so the cap
  agrees with ``_discovery_cache_bytes_total``), and orphan ``*.tmp`` /
  ``.lru.lock`` files are swept by eviction and ``clear()``.
"""

from __future__ import annotations

import json
import os
from unittest import mock


from mlframe.training.composite.cache import (
    DiscoveryCache,
    _DISCOVERY_CACHE_SCHEMA_VERSION,
    _discovery_cache_bytes_total,
    compute_config_signature_v1,
)


class _Cfg:
    """Minimal pydantic-shaped config (model_dump route)."""

    def model_dump(self, mode="json"):
        """Model dump."""
        return {"some_field": 1}


class TestS25VersionFoldedUnconditionally:
    """Groups tests covering s25 version folded unconditionally."""
    def test_signature_folds_mlframe_version_without_library_versions(self) -> None:
        """The whole point of S25: a caller passing NO library_versions still
        gets a code-version component, so the digest moves on an mlframe bump."""
        # Snapshot the JSON payload that gets hashed, without depending on the
        # opaque digest. Intercept stdlib json.dumps (production hashes via it).
        captured = {}
        real_dumps = json.dumps

        def capturing_dumps(obj, *a, **kw):
            """Capturing dumps."""
            if isinstance(obj, dict) and "_schema" in obj:
                captured["payload"] = obj
            return real_dumps(obj, *a, **kw)

        with mock.patch.object(json, "dumps", side_effect=capturing_dumps):
            compute_config_signature_v1(_Cfg())  # NO library_versions

        assert "payload" in captured, "_schema block never folded into the digest"
        schema = captured["payload"]["_schema"]
        assert schema["discovery_cache_schema_version"] == _DISCOVERY_CACHE_SCHEMA_VERSION
        assert "mlframe" in schema, "mlframe version not folded unconditionally"
        # Must be a real version string, not the absent sentinel, in a healthy install.
        assert schema["mlframe"] not in ("", None)

    def test_signature_changes_when_mlframe_version_bumps_no_library_versions(self) -> None:
        """Pre-fix this was the bug: without library_versions the signature was
        version-blind, so two mlframe versions shared a cache entry."""
        sig_real = compute_config_signature_v1(_Cfg())  # NO library_versions
        with mock.patch("mlframe.__version__", "999.999"):
            sig_bumped = compute_config_signature_v1(_Cfg())  # NO library_versions
        assert sig_real != sig_bumped, "mlframe version bump did not change the config signature without library_versions -- stale spec replay hazard (S25)"

    def test_signature_changes_when_schema_version_bumps(self) -> None:
        # Patch to a value DIFFERENT from the current default (hardcoding "2" collided once the default itself was
        # bumped to 2); the digest must change whenever the schema-version component changes.
        """Signature changes when schema version bumps."""
        from mlframe.training.composite import cache as _cache_mod

        sig_v1 = compute_config_signature_v1(_Cfg())
        with mock.patch(
            "mlframe.training.composite.cache._DISCOVERY_CACHE_SCHEMA_VERSION",
            _cache_mod._DISCOVERY_CACHE_SCHEMA_VERSION + 1,
        ):
            sig_v2 = compute_config_signature_v1(_Cfg())
        assert sig_v1 != sig_v2, "schema-version bump did not invalidate the digest"

    def test_library_versions_still_folded_on_top(self) -> None:
        """The richer override must still discriminate (regression: S25 is additive)."""
        sig_a = compute_config_signature_v1(_Cfg(), library_versions={"polars": "1.0"})
        sig_b = compute_config_signature_v1(_Cfg(), library_versions={"polars": "2.0"})
        assert sig_a != sig_b


class TestS18EvictionAccountingAndSweep:
    """Groups tests covering s18 eviction accounting and sweep."""
    def test_eviction_size_cap_counts_sidecar(self, tmp_path) -> None:
        """S18: the eviction byte cap must agree with _discovery_cache_bytes_total
        (which counts .pkl + .pkl.sha256).

        Discriminating construction: with TINY values the ~100 B sidecar dominates
        the ~25 B pkl, so a cap set strictly between the pkl-ONLY total and the
        pkl+sidecar total makes the pre-fix (pkl-only) accounting measure UNDER
        budget and evict nothing, while the post-fix (sidecar-inclusive) accounting
        measures OVER budget and evicts. Pre-fix leaves all 3 entries; post-fix
        evicts down to <= the cap."""
        cache = DiscoveryCache(str(tmp_path), max_entries=None, max_size_mb=float("inf"))
        # Tiny values: the fixed-size .sha256 sidecar is the dominant per-entry cost.
        for k in ("a", "b", "c"):
            cache.set(k * 32, {"v": k})
        d = cache.cache_dir
        pkl_only = sum(os.path.getsize(os.path.join(d, n)) for n in os.listdir(d) if n.endswith(".pkl"))
        reported = _discovery_cache_bytes_total(cache)
        assert reported > pkl_only, "sidecars must contribute to the reported footprint"
        # Cap strictly between pkl-only and pkl+sidecar totals. Pre-fix accounting
        # (pkl_only <= cap) is under budget -> no eviction; post-fix (reported > cap)
        # is over budget -> eviction fires.
        cap_bytes = (pkl_only + reported) / 2.0
        assert pkl_only <= cap_bytes < reported  # the discriminating band
        cache.max_size_mb = cap_bytes / (1024 * 1024)
        # Re-touch eviction without adding bytes: invalidate+re-set the same key is
        # noisy; instead call the private evictor directly (it is what set() runs).
        removed = cache._evict_to_caps()
        n_pkl = len([f for f in os.listdir(d) if f.endswith(".pkl")])
        post_total = _discovery_cache_bytes_total(cache)
        assert removed >= 1 and post_total <= cap_bytes, (
            "eviction did not fire under a sidecar-inclusive size cap (S18): "
            f"removed={removed}, {n_pkl} entries remain, post_total={post_total} > cap={cap_bytes}"
        )

    def test_clear_sweeps_orphan_tmp_and_lock(self, tmp_path) -> None:
        """S18: clear() must reclaim orphan *.tmp (interrupted writes) and the
        .lru.lock marker -- pre-fix neither was swept and they accumulated."""
        cache = DiscoveryCache(str(tmp_path))
        cache.set("d" * 32, {"x": 1})
        # Simulate an interrupted write (orphan tmp) + a lingering lock marker.
        orphan = os.path.join(cache.cache_dir, "leftover.tmp")
        with open(orphan, "wb") as f:
            f.write(b"partial pickle")
        lock = cache._lock_path()
        with open(lock, "w", encoding="utf-8") as f:
            f.write("")
        cache.clear()
        assert not os.path.exists(orphan), "clear() left an orphan *.tmp"
        assert not os.path.exists(lock), "clear() left the .lru.lock marker"

    def test_eviction_sweeps_aged_orphan_tmp(self, tmp_path) -> None:
        """S18: an aged orphan tmp must be reclaimed by the eviction sweep so the
        cap accounting is not under-counted by invisible *.tmp files."""
        cache = DiscoveryCache(str(tmp_path), max_entries=1, max_size_mb=None)
        orphan = os.path.join(cache.cache_dir, "aged.tmp")
        with open(orphan, "wb") as f:
            f.write(b"x" * 1024)
        # Age it past the min-age gate via the env override (set to 0 so any age qualifies).
        with mock.patch.dict(os.environ, {"MLFRAME_DISCOVERY_CACHE_TMP_AGE_S": "0"}):
            # set() triggers _evict_to_caps -> _sweep_orphan_tmp_files.
            cache.set("e" * 32, {"x": 1})
            cache.set("f" * 32, {"x": 1})
        assert not os.path.exists(orphan), "eviction did not sweep the aged orphan tmp"

    def test_eviction_preserves_fresh_inflight_tmp(self, tmp_path) -> None:
        """A fresh (in-flight) *.tmp must NOT be yanked by the sweep -- only aged
        orphans are reclaimed, so a concurrent write is never corrupted."""
        cache = DiscoveryCache(str(tmp_path), max_entries=1, max_size_mb=None)
        fresh = os.path.join(cache.cache_dir, "fresh.tmp")
        with open(fresh, "wb") as f:
            f.write(b"in flight")
        # Default min-age is 1h; a just-created tmp is far younger.
        cache.set("g" * 32, {"x": 1})
        cache.set("h" * 32, {"x": 1})
        assert os.path.exists(fresh), "sweep removed a fresh in-flight tmp (race hazard)"
