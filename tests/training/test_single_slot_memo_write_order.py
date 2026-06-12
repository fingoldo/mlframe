"""Regression: single-slot id-memo caches must publish the VALUE field before the KEY field.

These unlocked module-level memos (``_MRMR_LAST_X_HASH_CACHE``, ``_PD_VIEW_LAST_CACHE``, ``_LAST_KEY_CACHE``) short-circuit an expensive
recompute when the same input recurs. Reads are unsynchronised and MRMR.fit / the pipeline-cache key build may run under joblib threads.
If the KEY field is published before the VALUE field, a concurrent reader can observe a NEW key paired with a STALE value (a value computed
for a DIFFERENT, prior input) and return it -- a wrong-data cache collision. Publishing the value first makes a torn read degrade to a
cache MISS (recompute), never a wrong hit. We pin the ordering deterministically via a dict that records assignment order; pre-fix code
(key-before-value) fails these.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class _OrderRecordingDict(dict):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.set_order = []

    def __setitem__(self, key, value):
        self.set_order.append(key)
        super().__setitem__(key, value)


def test_mrmr_x_hash_memo_publishes_value_before_key():
    import mlframe.feature_selection.filters._mrmr_fingerprints as fp

    rec = _OrderRecordingDict({"id_shape": None, "hash": None})
    orig = fp._MRMR_LAST_X_HASH_CACHE
    fp._MRMR_LAST_X_HASH_CACHE = rec
    try:
        fp._full_x_content_hash(pd.DataFrame({"a": np.arange(8.0), "b": np.arange(8.0)}))
    finally:
        fp._MRMR_LAST_X_HASH_CACHE = orig
    assert rec.set_order, "memo store path did not run"
    assert rec.set_order.index("hash") < rec.set_order.index("id_shape"), (
        "value (hash) must be published before key (id_shape) so a torn read cannot pair a new key with a stale hash"
    )


def test_pipeline_last_key_memo_publishes_value_before_key():
    import mlframe.training.pipeline._pipeline_cache as pc

    rec = _OrderRecordingDict({"id_tup": None, "key": None})
    orig = pc._LAST_KEY_CACHE
    pc._LAST_KEY_CACHE = rec
    try:
        tr = pd.DataFrame({"a": np.arange(8.0)})
        va = pd.DataFrame({"a": np.arange(4.0)})
        tgt = pd.Series(np.arange(8.0), name="y")
        pc._pre_pipeline_cache_key(tr, va, None, train_target=tgt, target_name="y")
    finally:
        pc._LAST_KEY_CACHE = orig
    assert rec.set_order, "memo store path did not run"
    assert rec.set_order.index("key") < rec.set_order.index("id_tup"), (
        "value (key) must be published before key (id_tup) so a torn read cannot pair a new id_tup with a stale key"
    )


def test_pd_view_memo_publishes_value_before_key():
    pl = pytest.importorskip("polars")
    import mlframe.training.utils as ut

    rec = _OrderRecordingDict({"id_key": None, "result": None})
    orig = ut._PD_VIEW_LAST_CACHE
    ut._PD_VIEW_LAST_CACHE = rec
    try:
        ut.get_pandas_view_of_polars_df(pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}))
    finally:
        ut._PD_VIEW_LAST_CACHE = orig
    if not rec.set_order:
        pytest.skip("pd-view memo store path not exercised on this build/config")
    assert rec.set_order.index("result") < rec.set_order.index("id_key"), (
        "value (result) must be published before key (id_key) so a torn read cannot pair a new id_key with a stale view"
    )
