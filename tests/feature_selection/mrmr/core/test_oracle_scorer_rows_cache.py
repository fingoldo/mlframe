"""Wave 13 (6): _oracle_scorer_select.py's _learned_scorer re-read+re-parsed the entire on-disk oracle
parquet history on EVERY call. Adds a (store_path, mtime)-keyed cache (_cached_read_rows) so repeated
recommend_scorer calls between oracle writes reuse the parsed rows instead of re-reading the file.
"""

from __future__ import annotations

import os
import tempfile
from unittest import mock

import numpy as np
import pandas as pd


def _make_selector(store_dir):
    from mlframe.feature_selection.filters._oracle_scorer_select import OracleScorerSelector

    store_path = os.path.join(store_dir, "oracle_rows_cache_test.parquet")
    return OracleScorerSelector(store_path=store_path), store_path


def test_cached_read_rows_hits_cache_between_writes():
    from mlframe.feature_selection.filters._oracle_scorer_select import _cached_read_rows, _ROWS_CACHE

    with tempfile.TemporaryDirectory() as d:
        selector, _store_path = _make_selector(d)
        _ROWS_CACHE.clear()

        real_read_rows = selector.oracle.store.read_rows
        calls = {"n": 0}

        def _counting_read_rows():
            calls["n"] += 1
            return real_read_rows()

        with mock.patch.object(selector.oracle.store, "read_rows", _counting_read_rows):
            rows1 = _cached_read_rows(selector.oracle.store)
            rows2 = _cached_read_rows(selector.oracle.store)
            rows3 = _cached_read_rows(selector.oracle.store)

        assert calls["n"] == 1, f"expected exactly 1 underlying read_rows() call across 3 cache lookups, got {calls['n']}"
        assert rows1 == rows2 == rows3


def test_cached_read_rows_invalidates_on_write():
    """A write (mtime change) MUST invalidate the cache -- a stale read after a benchmark_all_scorers
    write would silently ignore the new observation."""
    from mlframe.feature_selection.filters._oracle_scorer_select import _cached_read_rows, _ROWS_CACHE

    with tempfile.TemporaryDirectory() as d:
        selector, _store_path = _make_selector(d)
        _ROWS_CACHE.clear()

        rows_before = _cached_read_rows(selector.oracle.store)
        assert rows_before == []  # no store file yet

        # Write one observation row via the real store API (append), which creates/updates the file
        # and therefore its mtime -- this must be visible on the NEXT _cached_read_rows call.
        selector.oracle.store.append(
            [
                {
                    "schema_version": 1,
                    "fn_name": "orth_scorer_select",
                    "host": "test-host",
                    "fp_bucket_json": "{}",
                    "param_combo_json": '{"scorer": "hsic"}',
                    "objective_json": '{"quality": 1.0}',
                    "n_obs": 1,
                    "ts": "2026-01-01T00:00:00+00:00",
                }
            ]
        )

        rows_after = _cached_read_rows(selector.oracle.store)
        assert len(rows_after) == 1, "cache must observe the new row after a write changed the store's mtime"


def test_recommend_scorer_unaffected_by_caching():
    """recommend_scorer's return value must be identical whether _learned_scorer's rows come from the
    cache or a fresh read -- the cache changes WHEN the file is read, never WHAT is returned for a
    stable mtime."""
    from mlframe.feature_selection.filters._oracle_scorer_select import _ROWS_CACHE

    with tempfile.TemporaryDirectory() as d:
        selector, _store_path = _make_selector(d)
        _ROWS_CACHE.clear()

        rng = np.random.default_rng(0)
        n = 200
        X = pd.DataFrame({"a": rng.random(n), "b": rng.random(n)})
        y = pd.Series(rng.random(n))

        r1 = selector.recommend_scorer(X, y)
        r2 = selector.recommend_scorer(X, y)
        assert r1 == r2
