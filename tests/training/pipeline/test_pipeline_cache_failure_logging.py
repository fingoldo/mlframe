"""Baseline-debt wave 6: _pipeline_cache.py's fingerprint/hash fallback paths log a debug trace
when the fast-path sampling/hashing fails, instead of silently degrading to an uncachable
sentinel (or an empty-string skip for the full-content hash) with zero trace of why.
"""

from __future__ import annotations

import logging

import pandas as pd


def test_approx_entry_bytes_logs_on_failure(caplog):
    """`_approx_entry_bytes`'s per-object size probe must log when it fails, not just return 0."""
    from mlframe.training.pipeline._pipeline_cache import _approx_entry_bytes

    class _RaisingSizeObj:
        """No `.nbytes` (falls through to `.memory_usage`, which raises)."""

        def memory_usage(self, deep=False, index=True):
            """Always raises ``RuntimeError('boom')`` when called."""
            raise RuntimeError("boom")

    with caplog.at_level(logging.DEBUG, logger="mlframe.training.pipeline._pipeline_cache"):
        out = _approx_entry_bytes((_RaisingSizeObj(), None))
    assert out == 0
    assert any("byte-size estimation failed" in rec.message for rec in caplog.records)


def test_content_fingerprint_pandas_row_sampling_logs_on_failure(caplog):
    """`_content_fingerprint_for_cache`'s pandas-DataFrame row-sampling fallback must log --
    forced via a cell whose `.tolist()` marks it as needing `repr()`-coercion, and whose
    `__repr__` raises, exercising the same except-branch this fix touches."""
    from mlframe.training.pipeline._pipeline_cache import _content_fingerprint_for_cache

    class _RaisesOnRepr:
        """A cell that looks like an ndarray-ish object (has `.tolist()`) but explodes on repr()."""

        def tolist(self):
            """Return self so the row-to-hashable coercion takes the repr() branch."""
            return self

        def __repr__(self):
            """Always raises ``RuntimeError('boom')``."""
            raise RuntimeError("boom")

    df = pd.DataFrame({"a": [_RaisesOnRepr()] * 5})
    with caplog.at_level(logging.DEBUG, logger="mlframe.training.pipeline._pipeline_cache"):
        out = _content_fingerprint_for_cache(df)
    assert out[0] == "uncached"
    assert any("pandas DataFrame row-sampling failed" in rec.message for rec in caplog.records)


class _BrokenSequence:
    """Looks like a sequence (has `__len__`) but explodes on indexing -- makes
    `np.asarray(obj)` genuinely raise inside the array-from-sequence ducktype path, exercising
    the outer dispatch except in `_full_x_content_hash`/`_full_target_content_hash` (neither of
    which special-case a bare object; both fall through to the `else: np.asarray(arr)` branch)."""

    shape = (3,)

    def __len__(self):
        """Return 3 so numpy attempts sequence-style conversion."""
        return 3

    def __getitem__(self, idx):
        """Always raises ``RuntimeError('boom')`` on any index."""
        raise RuntimeError("boom")


def test_full_x_content_hash_logs_on_ndarray_conversion_failure(caplog):
    """`_full_x_content_hash`'s `np.asarray()` fallback except must log when hashing genuinely
    fails (reached before the outer dispatch except, since the inner except returns first)."""
    from mlframe.training.pipeline._pipeline_cache import _full_x_content_hash

    with caplog.at_level(logging.DEBUG, logger="mlframe.training.pipeline._pipeline_cache"):
        out = _full_x_content_hash(_BrokenSequence())
    assert out == ""
    assert any("np.asarray() fallback failed" in rec.message for rec in caplog.records)


def test_full_target_content_hash_logs_on_failure(caplog):
    """`_full_target_content_hash`'s outer-dispatch except must log (this function has no
    per-backend inner except around `np.asarray()`, unlike `_full_x_content_hash`, so the outer
    except is the one that actually catches this failure)."""
    from mlframe.training.pipeline._pipeline_cache import _full_target_content_hash

    with caplog.at_level(logging.DEBUG, logger="mlframe.training.pipeline._pipeline_cache"):
        out = _full_target_content_hash(_BrokenSequence())
    assert out == ""
    assert any("outer dispatch failed" in rec.message for rec in caplog.records)
