"""Regression sensor for A5 P2 #8.

Pre-fix the ``_pandas_view_cache`` was a plain dict with ``popitem`` FIFO eviction at a hard count cap of 4 and no byte gate. On a 100 GB workload the FIFO eviction sometimes dropped the most-recently used view (when older entries were hit but not re-marked), and a single oversized view could pin GB-scale blockmgr buffers across targets. The fix promotes the cache to OrderedDict + move_to_end on hit + byte budget gate (env override ``MLFRAME_PANDAS_VIEW_CACHE_MAX_MB``).
"""

from __future__ import annotations

from collections import OrderedDict

import pytest


def test_pandas_view_cache_slot_is_ordered_dict():
    """Pandas view cache slot is ordered dict."""
    from mlframe.training.core._training_context import TrainingContext

    ctx = TrainingContext()
    assert isinstance(ctx._pandas_view_cache, OrderedDict), "ctx._pandas_view_cache must be OrderedDict to support move_to_end-based LRU eviction"


def test_pandas_view_cache_bytes_helper_handles_empty_and_missing_attr():
    """Pandas view cache bytes helper handles empty and missing attr."""
    from mlframe.training.core._phase_train_one_target_polars_fastpath import _pandas_view_cache_bytes

    assert _pandas_view_cache_bytes(OrderedDict()) == 0

    class _NoMemUsage:
        """Groups tests covering no mem usage."""
        pass

    od = OrderedDict()
    od[1] = _NoMemUsage()
    assert _pandas_view_cache_bytes(od) == 0


def test_pandas_view_cache_bytes_helper_sums_pandas_frames():
    """Pandas view cache bytes helper sums pandas frames."""
    pd = pytest.importorskip("pandas")
    import numpy as np
    from mlframe.training.core._phase_train_one_target_polars_fastpath import _pandas_view_cache_bytes

    od = OrderedDict()
    od[1] = pd.DataFrame({"x": np.arange(1000, dtype=np.int64)})
    od[2] = pd.DataFrame({"y": np.arange(2000, dtype=np.int64)})
    n = _pandas_view_cache_bytes(od)
    assert n >= 8 * 3000, f"byte estimator should account for both frames' int64 buffers; got {n}"
