"""Regression test for audit2 leaks-F1: the per-instance CUDA-graph predict cache grew unbounded across
distinct batch shapes (one retained graph + 2 device buffers per shape, forever). It is now LRU-capped;
assert the graph count is bounded, that eviction reclaims the entry (no lingering ref -> VRAM freed), that
False capture-failure sentinels are NOT evicted, and that the most-recently-used shape survives.

The leaking resource is the retained Python-side tuple of device tensors; this targets the eviction
bookkeeping directly (deterministic, no VRAM contention with the real GPU capture/replay path).
"""

import gc
import weakref


from mlframe.training.neural._flat_torch_module import _flat_torch_predict_accel as accel


class _Stub(accel._PredictAccelMixin):
    """Groups tests covering stub."""
    def __init__(self):
        self._cuda_graph_predict_cache = {}


class _Dummy:
    """Weakref-able stand-in (bare object() cannot be weak-referenced)."""


def _entry():
    # Stand-in for (graph, static_in, static_out); real ones are device-resident torch objects.
    """Entry."""
    return (_Dummy(), _Dummy(), _Dummy())


def test_graph_cache_is_lru_capped_and_frees_evicted_entries(monkeypatch):
    """Graph cache is lru capped and frees evicted entries."""
    monkeypatch.setattr(accel, "_CUDA_GRAPH_PREDICT_CACHE_MAX", 3)
    m = _Stub()

    refs = {}
    for i in range(6):
        key = ((i, 4), "float32", "cuda:0")
        entry = _entry()
        refs[i] = weakref.ref(entry[0])  # track the "graph" object
        m._cuda_graph_predict_cache[key] = entry
        m._evict_cuda_graph_cache_if_needed()
        del entry

    graph_entries = [v for v in m._cuda_graph_predict_cache.values() if v is not False]
    assert len(graph_entries) == 3, "captured-graph count must be bounded by the cap"

    gc.collect()
    # The 3 oldest (i=0,1,2) were evicted -> their graph objects must be collectable (no cache ref).
    assert all(refs[i]() is None for i in (0, 1, 2)), "evicted graphs still referenced -> VRAM not reclaimed"
    # The 3 newest (i=3,4,5) survive.
    surviving_keys = {k[0][0] for k in m._cuda_graph_predict_cache}
    assert surviving_keys == {3, 4, 5}


def test_false_sentinels_are_not_evicted(monkeypatch):
    """A shape whose capture failed is cached as False (permanent eager fallback). Evicting it would cause a
    capture-retry storm, so the cap must count only real graphs and leave sentinels in place."""
    monkeypatch.setattr(accel, "_CUDA_GRAPH_PREDICT_CACHE_MAX", 2)
    m = _Stub()
    # Two failed-capture sentinels first...
    m._cuda_graph_predict_cache[((0, 4), "float32", "cuda:0")] = False
    m._cuda_graph_predict_cache[((1, 4), "float32", "cuda:0")] = False
    # ...then 4 real graphs.
    for i in range(2, 6):
        m._cuda_graph_predict_cache[((i, 4), "float32", "cuda:0")] = _entry()
        m._evict_cuda_graph_cache_if_needed()

    sentinels = [v for v in m._cuda_graph_predict_cache.values() if v is False]
    graphs = [v for v in m._cuda_graph_predict_cache.values() if v is not False]
    assert len(sentinels) == 2, "False capture-failure sentinels must survive (no retry storm)"
    assert len(graphs) == 2, "real graphs bounded by the cap"


def test_lru_move_to_end_keeps_the_hot_shape(monkeypatch):
    """Simulate the replay-hit reorder: touching a key moves it to the end so eviction spares it."""
    monkeypatch.setattr(accel, "_CUDA_GRAPH_PREDICT_CACHE_MAX", 2)
    m = _Stub()
    hot = ((0, 4), "float32", "cuda:0")
    m._cuda_graph_predict_cache[hot] = _entry()
    m._cuda_graph_predict_cache[((1, 4), "float32", "cuda:0")] = _entry()
    # Replay-hit on the oldest shape moves it to the end (this is what _maybe_cuda_graph_forward does).
    m._cuda_graph_predict_cache[hot] = m._cuda_graph_predict_cache.pop(hot)
    # A new capture arrives, triggering eviction of the now-coldest (the (1,4) shape), not the hot one.
    m._cuda_graph_predict_cache[((2, 4), "float32", "cuda:0")] = _entry()
    m._evict_cuda_graph_cache_if_needed()

    surviving = {k[0][0] for k in m._cuda_graph_predict_cache}
    assert 0 in surviving, "the recently-replayed (hot) shape must NOT be evicted"
    assert 1 not in surviving, "the cold shape should be evicted"
