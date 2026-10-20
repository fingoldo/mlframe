"""The fit-transform populate path must store the pre-pipeline result through `_pre_pipeline_cache_set`.

The populate block used to write the LRU slot inline, so `_pre_pipeline_cache_set` (the function the cache's own
contract is written around, and the one that honours `key=`) had no caller and the eviction logic lived in two places.
"""

from __future__ import annotations

import pandas as pd

from mlframe.training.pipeline import _pipeline_helpers_apply as apply_mod
from mlframe.training.pipeline._pipeline_cache import _PRE_PIPELINE_CACHE, _PRE_PIPELINE_CACHE_LOCK, _pre_pipeline_cache_get, _pre_pipeline_cache_key


class _FittedStub:
    """A pre-pipeline stand-in; the populate path only stores it and checks it is not None."""


def test_populate_routes_through_cache_set_and_is_retrievable_by_entry_key(monkeypatch):
    """The populate call goes through the set helper with the entry key, and the stored entry is found by that key."""
    raw_train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    raw_val = pd.DataFrame({"a": [4.0]})
    pipeline = _FittedStub()
    key = _pre_pipeline_cache_key(raw_train, raw_val, pipeline)
    out_train = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
    out_val = pd.DataFrame({"a": [3.0]})

    calls = []
    real_set = apply_mod._pre_pipeline_cache_set

    def _recording_set(*args, **kwargs):
        """Record the keyword arguments of each populate call, then delegate to the real cache write."""
        calls.append(kwargs)
        return real_set(*args, **kwargs)

    monkeypatch.setattr(apply_mod, "_pre_pipeline_cache_set", _recording_set)
    with _PRE_PIPELINE_CACHE_LOCK:
        _PRE_PIPELINE_CACHE.pop(key, None)
    try:
        apply_mod._apply_pre_pipeline_p_step2_maybe_clean_ram(
            False, out_train, 0.0, False, False, None, out_val, pipeline, key,
        )
        assert calls, "populate wrote the cache inline instead of through _pre_pipeline_cache_set"
        assert calls[0].get("key") is key
        hit = _pre_pipeline_cache_get(raw_train, raw_val, pipeline, key=key)
        assert hit is not None and hit[2] is pipeline
    finally:
        with _PRE_PIPELINE_CACHE_LOCK:
            _PRE_PIPELINE_CACHE.pop(key, None)
