"""Regression: the pipeline JSON-roundtrip disk cache must key on a stable content hash.

Pre-fix the key was builtin ``hash(_js)``, which is salted per process by PYTHONHASHSEED,
so every fresh process missed the on-disk verdict 100% of the time -- defeating the disk
cache entirely. The key must be a pure function of the JSON content, identical across runs
with different hash seeds.
"""

from __future__ import annotations

import hashlib

from mlframe.training.core._setup_helpers_pipeline_cache import pipeline_json_cache_key


def test_pipeline_cache_key_is_content_only_not_builtin_hash():
    js = '{"steps": [{"name": "yeo_johnson"}, {"name": "ordinal_encode"}]}'
    # Stable across calls in the same process.
    assert pipeline_json_cache_key(js) == pipeline_json_cache_key(js)
    # NOT the builtin hash (that is the salted value the pre-fix code used).
    assert pipeline_json_cache_key(js) != str(hash(js))


def test_pipeline_cache_key_is_pure_blake2b_of_content():
    """The key must be a pure content hash (PYTHONHASHSEED-independent by construction).

    blake2b is unsalted and identical across processes/hash-seeds; the pre-fix builtin
    ``hash(_js)`` was salted per process, causing 100% cross-process cache misses.
    """
    js = '{"steps":[{"name":"x"}]}'
    expected = hashlib.blake2b(js.encode("utf-8"), digest_size=16).hexdigest()
    assert pipeline_json_cache_key(js) == expected
