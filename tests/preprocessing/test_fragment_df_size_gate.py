"""Regression test for MEM3: fragment_df_on_ram_usage_increase byte-size gate."""

import numpy as np
import pandas as pd

from mlframe.preprocessing import cleaning
from mlframe.preprocessing.cleaning import fragment_df_on_ram_usage_increase


def _force_ram_rise(monkeypatch):
    # get_own_memory_usage is called once at function entry; return a value >50% above
    # prev_mem_usage (100.0) so the defrag branch is reached.
    """Test helper: monkeypatch.setattr(cleaning, 'get_own_memory_usage', lam...."""
    monkeypatch.setattr(cleaning, "get_own_memory_usage", lambda: 1000.0)


def test_above_threshold_returns_same_object_no_copy(monkeypatch):
    """Above threshold returns same object no copy."""
    _force_ram_rise(monkeypatch)
    df = pd.DataFrame({"a": [1, 2, 3]})
    # Pretend the frame is huge so the copy is skipped.
    monkeypatch.setattr(cleaning, "_DEFRAG_COPY_MAX_BYTES", 10)  # df is bigger than 10 bytes
    out, _ = fragment_df_on_ram_usage_increase(df, prev_mem_usage=100.0)
    assert out is df, "above the byte-size gate the original frame must be returned (no copy)"


def test_below_threshold_still_defragments(monkeypatch):
    """Below threshold still defragments."""
    _force_ram_rise(monkeypatch)
    df = pd.DataFrame({"a": np.arange(10)})
    # Large gate => small frame defrags as before.
    monkeypatch.setattr(cleaning, "_DEFRAG_COPY_MAX_BYTES", 2 * 1024**3)
    out, _ = fragment_df_on_ram_usage_increase(df, prev_mem_usage=100.0)
    assert out is not df, "below the gate a defragmenting copy is returned"
    pd.testing.assert_frame_equal(out, df)
