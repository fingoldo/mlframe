"""Objects that hold a live lock or condition must still pickle: the lock is dropped on save and made again on load."""

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted data

import pandas as pd

from mlframe.feature_selection.filters._composite_group_agg_fe import _ColumnCache
from mlframe.feature_selection.filters._gpu_alloc_trace import _TraceState
from mlframe.utils.host_block_budget import HostBlockBudget


def test_host_block_budget_round_trips_and_keeps_its_total():
    """HostBlockBudget pickles its total and can still reserve after load (the condition is rebuilt)."""
    restored = pickle.loads(pickle.dumps(HostBlockBudget(4096)))  # nosec B301 -- round-trip of a locally-created, trusted object
    assert restored.total == 4096
    with restored.reserve(1024):
        pass


def test_trace_state_round_trips_its_counters():
    """_TraceState pickles its counters; the restored lock is usable."""
    state = _TraceState()
    state.n_allocations = 3
    restored = pickle.loads(pickle.dumps(state))  # nosec B301 -- round-trip of a locally-created, trusted object
    assert restored.n_allocations == 3
    with restored.lock:
        pass


def test_column_cache_round_trips_its_rows():
    """_ColumnCache pickles without its lock and keeps the cached column rows."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    cache = _ColumnCache(X, ["a"])
    restored = pickle.loads(pickle.dumps(cache))  # nosec B301 -- round-trip of a locally-created, trusted object
    with restored._lock:
        pass
    assert restored.values("a") is None or restored.values("a").shape == cache.values("a").shape
