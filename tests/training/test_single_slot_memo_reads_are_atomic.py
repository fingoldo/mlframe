"""Single-slot memos must never pair one input's key with another input's value under threads (mrmr_audit_2026-09-14 PERIPHERY-5 / IMPL-7).

Three module-level memos short-circuit an expensive recompute when the same input recurs: the MRMR X-content hash, the pipeline
cache key, and the polars-to-pandas view. Each READ did two separate unlocked lookups -- compare the key, then fetch the value -- and
each WRITE published the value before the key, on the theory that a torn read would then degrade to a miss. It does not: a reader that
has already matched its key can have a concurrent writer publish a DIFFERENT input's value before it fetches, and returns that value
for its own input, whatever order the writer used. For the pandas view that is the wrong DataFrame, not just a wrong hash.

Reads and writes now share one lock per memo. These tests force the dangerous interleave deterministically with real threads: the
reader is held between its key compare and its value fetch while a writer publishes another input. Unlocked, the writer finishes and
the reader returns the wrong value; locked, the writer waits and the reader returns its own.
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest

_WAIT_S = 1.0


class _PausingDict(dict):
    """Pauses the READER thread right after it reads ``key_field``, until the writer finishes or the wait times out."""

    def __init__(self, base: dict, key_field: str):
        super().__init__(base)
        self.key_field = key_field
        self.reader: threading.Thread | None = None
        self.reader_in_window = threading.Event()
        self.writer_done = threading.Event()

    def _maybe_pause(self, name):
        """Open the race window only for the reader thread, only on the key read, only once."""
        if name == self.key_field and threading.current_thread() is self.reader and not self.reader_in_window.is_set():
            self.reader_in_window.set()
            self.writer_done.wait(_WAIT_S)

    def __getitem__(self, name):
        value = super().__getitem__(name)
        self._maybe_pause(name)
        return value

    def get(self, name, default=None):
        """Mirror ``__getitem__``'s pause for call sites that use ``.get``."""
        value = super().get(name, default)
        self._maybe_pause(name)
        return value


def _race(module, attr: str, key_field: str, read_a, write_b):
    """Prime the memo with A, then run reader(A) and writer(B) through the forced interleave; return what the reader got."""
    read_a()  # single-threaded: A becomes the cached entry
    original = getattr(module, attr)
    slot = _PausingDict(original, key_field)
    setattr(module, attr, slot)
    got: dict = {}
    try:
        def reader():
            """Read A through the memo; the slot pauses this thread inside the read."""
            got["value"] = read_a()

        def writer():
            """Once the reader is inside its read, publish B."""
            slot.reader_in_window.wait(_WAIT_S)
            write_b()
            slot.writer_done.set()

        rt = threading.Thread(target=reader)
        wt = threading.Thread(target=writer)
        slot.reader = rt
        rt.start()
        wt.start()
        rt.join(10)
        wt.join(10)
        assert slot.reader_in_window.is_set(), "the reader never reached the memo HIT path, so no race was exercised"
    finally:
        setattr(module, attr, original)
    return got["value"]


def test_x_content_hash_memo_never_returns_another_frames_digest():
    """MRMR's X-hash memo: the reader must get A's digest even while B's is being published."""
    import mlframe.feature_selection.filters._mrmr_fingerprints as fp

    A = np.arange(400, dtype=np.float64).reshape(20, 20)
    B = (np.arange(400, dtype=np.float64) + 1e6).reshape(20, 20)
    expected_a = fp._full_x_content_hash(A.copy())
    got = _race(fp, "_MRMR_LAST_X_HASH_CACHE", "id_shape", lambda: fp._full_x_content_hash(A), lambda: fp._full_x_content_hash(B))
    assert got == expected_a, "the X-hash memo returned another frame's digest"


def test_pipeline_cache_key_memo_never_returns_another_inputs_key():
    """The pre-pipeline cache-key memo: the reader must get the key for its own frames."""
    import mlframe.training.pipeline._pipeline_cache as pc

    tr_a, va_a = pd.DataFrame({"a": np.arange(8.0)}), pd.DataFrame({"a": np.arange(4.0)})
    tr_b, va_b = pd.DataFrame({"a": np.arange(8.0) + 100.0}), pd.DataFrame({"a": np.arange(4.0) + 100.0})
    tgt_a = pd.Series(np.arange(8.0), name="y")
    tgt_b = pd.Series(np.arange(8.0) + 100.0, name="y")

    def key_a():
        """Cache key for input A."""
        return pc._pre_pipeline_cache_key(tr_a, va_a, None, train_target=tgt_a, target_name="y")

    def key_b():
        """Cache key for input B."""
        return pc._pre_pipeline_cache_key(tr_b, va_b, None, train_target=tgt_b, target_name="y")

    expected_a = key_a()
    assert expected_a != key_b(), "fixture precondition: the two inputs must have different keys"
    got = _race(pc, "_LAST_KEY_CACHE", "id_tup", key_a, key_b)
    assert got == expected_a, "the pipeline cache-key memo returned another input's key"


def test_pandas_view_memo_never_returns_another_frames_data():
    """The polars-to-pandas view memo: a wrong hit here hands back a different DataFrame, not just a wrong hash."""
    pl = pytest.importorskip("polars")
    import mlframe.training.utils as ut

    frame_a = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    frame_b = pl.DataFrame({"a": [9.0, 9.0, 9.0], "b": [8.0, 8.0, 8.0]})
    expected_a = ut.get_pandas_view_of_polars_df(frame_a).copy()
    got = _race(ut, "_PD_VIEW_LAST_CACHE", "id_key", lambda: ut.get_pandas_view_of_polars_df(frame_a), lambda: ut.get_pandas_view_of_polars_df(frame_b))
    pd.testing.assert_frame_equal(got.reset_index(drop=True), expected_a.reset_index(drop=True), check_dtype=False)
