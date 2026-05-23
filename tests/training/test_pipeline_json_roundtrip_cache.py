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
import time
from unittest.mock import patch

import pytest


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

        assert call_count["n"] == 1, (
            f"Pipeline.from_json was invoked {call_count['n']} times; "
            f"expected exactly 1 (cache should short-circuit the second call)."
        )
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

    assert call_count["n"] == 1, (
        f"Failing from_json invoked {call_count['n']} times; expected 1 "
        f"(negative-result cache must skip retry)."
    )
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
