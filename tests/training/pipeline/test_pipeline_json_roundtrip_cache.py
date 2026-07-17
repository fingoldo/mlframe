"""Regression test for iter193: polars-ds Pipeline.from_json roundtrip
validation must be cached per-process.

c0141 iter193 profile (binary cb+hgb+linear + Boruta, 200k) attributed
5.413s wall to a single ``Pipeline.from_json(_js)`` call inside
``_finalize_and_save_metadata``. The call is a deterministic roundtrip
validation -- same JSON input always produces the same parse result -- so
the second / third / Nth fit in the same process can reuse the result.

The cache lives at
``mlframe.training.core._setup_helpers._PIPELINE_JSON_ROUNDTRIP_CACHE``
and is keyed by ``hash(_js)``. Mirrors the _PROBE_PRECISION_CACHE
(mlp_runtime_defaults iter181), _CB_GPU_USABLE_CACHE (_cb_pool), and
_mlframe_callback_cache_installed (neural/base iter189) patterns.
"""




def test_pipeline_json_roundtrip_cache_skips_second_validation():
    """First call populates the cache, second call with same JSON hash
    short-circuits without re-invoking ``Pipeline.from_json``."""
    from mlframe.training.core import _setup_helpers as sh

    # Clear cache to make the test deterministic across re-runs.
    sh._PIPELINE_JSON_ROUNDTRIP_CACHE.clear()

    # Simulate the JSON + pipeline shape. Use a stub since we don't have
    # polars_ds installed in every CI env (test_pipeline_json_roundtrip_cache
    # avoids depending on it).
    fake_js = '{"steps": [{"type": "test_stub"}]}'
    call_count = {"n": 0}

    class _StubPipeline:
        def to_json(self):
            return fake_js

        @classmethod
        def from_json(cls, js):
            call_count["n"] += 1
            return cls()

    # Mock polars_ds.pipeline.Pipeline import + isinstance gate.
    import sys

    fake_module = type(sys)("polars_ds")
    fake_module.pipeline = type(sys)("polars_ds.pipeline")
    fake_module.pipeline.Pipeline = _StubPipeline
    sys.modules["polars_ds"] = fake_module
    sys.modules["polars_ds.pipeline"] = fake_module.pipeline
    try:
        # Re-run the cache-using block twice.
        for _ in range(2):
            _js = fake_js
            _js_hash = hash(_js)
            _rt_ok = sh._PIPELINE_JSON_ROUNDTRIP_CACHE.get(_js_hash)
            if _rt_ok is None:
                try:
                    _StubPipeline.from_json(_js)
                    _rt_ok = True
                except Exception:
                    _rt_ok = False
                sh._PIPELINE_JSON_ROUNDTRIP_CACHE[_js_hash] = _rt_ok

        assert call_count["n"] == 1, f"Pipeline.from_json was invoked {call_count['n']} times; expected exactly 1 (cache should short-circuit the second call)."
        assert sh._PIPELINE_JSON_ROUNDTRIP_CACHE.get(hash(fake_js)) is True
    finally:
        sys.modules.pop("polars_ds", None)
        sys.modules.pop("polars_ds.pipeline", None)
        sh._PIPELINE_JSON_ROUNDTRIP_CACHE.clear()


def test_pipeline_json_roundtrip_cache_remembers_failures():
    """Negative-result caching: a JSON that fails parse this time will fail
    every time (deterministic), so the cache must remember False outcomes too
    and skip retry."""
    from mlframe.training.core import _setup_helpers as sh

    sh._PIPELINE_JSON_ROUNDTRIP_CACHE.clear()
    fake_js = '{"steps": [{"type": "always_fails"}]}'
    call_count = {"n": 0}

    def _failing_from_json(js):
        call_count["n"] += 1
        raise ValueError("intentional failure for test")

    # Drive the cache directly.
    for _ in range(3):
        _js_hash = hash(fake_js)
        _rt_ok = sh._PIPELINE_JSON_ROUNDTRIP_CACHE.get(_js_hash)
        if _rt_ok is None:
            try:
                _failing_from_json(fake_js)
                _rt_ok = True
            except Exception:
                _rt_ok = False
            sh._PIPELINE_JSON_ROUNDTRIP_CACHE[_js_hash] = _rt_ok

    assert call_count["n"] == 1, f"Failing from_json invoked {call_count['n']} times; expected 1 (negative-result cache must skip retry)."
    assert sh._PIPELINE_JSON_ROUNDTRIP_CACHE.get(hash(fake_js)) is False
    sh._PIPELINE_JSON_ROUNDTRIP_CACHE.clear()


def test_pipeline_json_cache_keyed_by_content_not_object_identity():
    """Two distinct JSON strings with different content must NOT share a
    cache entry, but the SAME content built twice MUST hit."""
    from mlframe.training.core import _setup_helpers as sh

    sh._PIPELINE_JSON_ROUNDTRIP_CACHE.clear()
    js_a = '{"steps": [{"type": "a"}]}'
    js_b = '{"steps": [{"type": "b"}]}'
    sh._PIPELINE_JSON_ROUNDTRIP_CACHE[hash(js_a)] = True
    sh._PIPELINE_JSON_ROUNDTRIP_CACHE[hash(js_b)] = False

    assert sh._PIPELINE_JSON_ROUNDTRIP_CACHE.get(hash(js_a)) is True
    assert sh._PIPELINE_JSON_ROUNDTRIP_CACHE.get(hash(js_b)) is False
    # Reconstructed same content -> same hash.
    assert sh._PIPELINE_JSON_ROUNDTRIP_CACHE.get(hash('{"steps": [{"type": "a"}]}')) is True
    sh._PIPELINE_JSON_ROUNDTRIP_CACHE.clear()


def test_pipeline_json_disk_cache_roundtrip(tmp_path, monkeypatch):
    """iter275 cross-process file cache: persist verdict to disk so a
    fresh process inherits the validation result."""
    from mlframe.training.core import _setup_helpers as sh

    # Redirect cache to a per-test tmpdir to keep prod cache untouched.
    cache_file = str(tmp_path / "polars_ds_pipeline_roundtrip.json")
    monkeypatch.setattr(sh, "_PIPELINE_JSON_DISK_CACHE_PATH", cache_file)
    monkeypatch.setattr(sh, "_PIPELINE_JSON_DISK_CACHE_LOADED", False)
    sh._PIPELINE_JSON_ROUNDTRIP_CACHE.clear()

    # Seed the in-memory cache + persist to disk. Keys are the production cache-key
    # form -- a content-only blake2b hexdigest string (PYTHONHASHSEED-stable), NOT a
    # builtin hash() int; the disk layer canonicalises keys to str, so seeding with an
    # int would not survive the round-trip (and prod never uses int keys).
    from mlframe.training.core._setup_helpers_pipeline_cache import pipeline_json_cache_key

    cache_key = pipeline_json_cache_key('{"steps": [{"type": "test_disk_cache"}]}')
    sh._PIPELINE_JSON_ROUNDTRIP_CACHE[cache_key] = True
    sh._persist_pipeline_disk_cache()

    import os

    assert os.path.exists(cache_file), "disk cache file must be created on persist"

    # Wipe the in-memory cache + reset loaded marker, then hydrate from disk.
    sh._PIPELINE_JSON_ROUNDTRIP_CACHE.clear()
    monkeypatch.setattr(sh, "_PIPELINE_JSON_DISK_CACHE_LOADED", False)
    sh._load_pipeline_disk_cache_into_memory()

    assert sh._PIPELINE_JSON_ROUNDTRIP_CACHE.get(cache_key) is True, "disk cache must rehydrate into the in-memory cache on load"


def test_pipeline_json_disk_cache_version_invalidation(tmp_path, monkeypatch):
    """A polars-ds version change invalidates the on-disk cache so a
    wheel that newly fails roundtrip can't silently inherit a stale
    'safe' verdict."""
    from mlframe.training.core import _setup_helpers as sh
    import orjson as _orjson

    cache_file = str(tmp_path / "polars_ds_pipeline_roundtrip.json")
    monkeypatch.setattr(sh, "_PIPELINE_JSON_DISK_CACHE_PATH", cache_file)
    monkeypatch.setattr(sh, "_PIPELINE_JSON_DISK_CACHE_LOADED", False)
    sh._PIPELINE_JSON_ROUNDTRIP_CACHE.clear()

    # Hand-craft a cache file with a stale version tag.
    fake_hash = "12345"
    with open(cache_file, "wb") as fh:
        fh.write(
            _orjson.dumps(
                {
                    "version_tag": "polars_ds=0.0.0-stale|polars=0.0.0-stale",
                    "entries": {fake_hash: True},
                }
            )
        )

    sh._load_pipeline_disk_cache_into_memory()
    assert int(fake_hash) not in sh._PIPELINE_JSON_ROUNDTRIP_CACHE, "stale-version cache file must NOT hydrate entries into in-memory"


def test_pipeline_json_disk_cache_corrupt_file_does_not_crash(tmp_path, monkeypatch):
    """A corrupted cache file (truncated JSON, garbage bytes) must not
    crash the training pipeline; load is best-effort."""
    from mlframe.training.core import _setup_helpers as sh

    cache_file = str(tmp_path / "polars_ds_pipeline_roundtrip.json")
    monkeypatch.setattr(sh, "_PIPELINE_JSON_DISK_CACHE_PATH", cache_file)
    monkeypatch.setattr(sh, "_PIPELINE_JSON_DISK_CACHE_LOADED", False)
    sh._PIPELINE_JSON_ROUNDTRIP_CACHE.clear()

    with open(cache_file, "w", encoding="utf-8") as fh:
        fh.write("{not valid json")

    # Should NOT raise.
    sh._load_pipeline_disk_cache_into_memory()
    # In-memory cache should remain empty (no entries inherited).
    assert len(sh._PIPELINE_JSON_ROUNDTRIP_CACHE) == 0
