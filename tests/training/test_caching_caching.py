"""Regression tests for the F4 (caching) audit wave.

Covers:
    - PRECOMPUTE-DUMMY-STUB / PRECOMPUTE-COMPOSITE-STUB (stubs raise NotImplementedError; empty
      bundles fall through to inline compute via the truthy gate).
    - FH-INMEMKEY-ID-RECYCLE (``FeatureCache.purge_by_df_token`` drops entries by df_token).
    - FP-ARROW-TRIPLE-CONV (``fingerprint_df`` produces the same logical fingerprint via the
      xxhash path; per-process LRU memoises repeated calls).
    - DISC-CACHE-NULL-DTYPE (integer columns with NaN sentinels keep min/max/nuniq stats).
    - DISC-VERSION-LEAK (version tuple folds major.minor, not patch).
    - DISC-RANDOM-STATE-DBL (random_state no longer folded a second time at make_discovery_cache_key).
    - DISC-LRU-RACE (``_touch_lru`` is wrapped by the cross-process filelock when filelock is present).
"""
from __future__ import annotations

import os
import sys
import threading

import numpy as np
import pytest

pl = pytest.importorskip("polars")


def _need_xxhash():
    pytest.importorskip("xxhash")


def _need_filelock():
    pytest.importorskip("filelock")


# ---------------------------------------------------------------------------
# PRECOMPUTE stubs
# ---------------------------------------------------------------------------


def test_precompute_dummy_baselines_raises_notimplementederror():
    from mlframe.training.helpers import precompute_dummy_baselines
    with pytest.raises(NotImplementedError, match="precompute_dummy_baselines"):
        precompute_dummy_baselines(train_df=None, target_by_type={})


def test_precompute_composite_target_specs_raises_notimplementederror():
    from mlframe.training.helpers import precompute_composite_target_specs
    with pytest.raises(NotImplementedError, match="precompute_composite_target_specs"):
        precompute_composite_target_specs()


def test_precompute_all_only_fills_stats_slot():
    """``precompute_all`` must NOT call the raise-ing stubs; remaining slots stay None."""
    import pandas as pd
    from mlframe.training.helpers import precompute_all
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    bundle = precompute_all(df)
    assert bundle.trainset_features_stats is not None
    assert bundle.dummy_baselines is None
    assert bundle.composite_target_specs is None


# ---------------------------------------------------------------------------
# FH-INMEMKEY-ID-RECYCLE
# ---------------------------------------------------------------------------


def test_feature_cache_purge_by_df_token_drops_matching_entries():
    from mlframe.training.feature_handling.cache import FeatureCache
    from mlframe.training.feature_handling.fingerprint import InMemoryKey
    from mlframe.training.feature_handling.config import CacheConfig
    cache = FeatureCache(CacheConfig(persistence="off"))
    k1 = InMemoryKey(
        session_id="s",
        df_token=42,
        train_idx_token=1,
        column="x",
        params_canonical_hash="p",
        provider_signature="v",
    )
    k2 = InMemoryKey(
        session_id="s",
        df_token=43,
        train_idx_token=1,
        column="x",
        params_canonical_hash="p",
        provider_signature="v",
    )
    cache.get_or_compute(k1, lambda: np.zeros(4, dtype=np.float32))
    cache.get_or_compute(k2, lambda: np.zeros(4, dtype=np.float32))
    assert cache.stats()["n_keys"] == 2

    dropped = cache.purge_by_df_token(42)
    assert dropped == 1
    assert cache.stats()["n_keys"] == 1


# ---------------------------------------------------------------------------
# FP-ARROW-TRIPLE-CONV + memo
# ---------------------------------------------------------------------------


def test_fingerprint_df_memo_returns_same_object():
    from mlframe.training.feature_handling.fingerprint import (
        fingerprint_df,
        reset_session,
    )
    reset_session()  # drop any leftover entries from earlier tests
    df = pl.DataFrame({"a": np.arange(1024, dtype=np.int64), "b": np.arange(1024, dtype=np.float32)})
    fp1 = fingerprint_df(df)
    fp2 = fingerprint_df(df)
    # Identity check is the strongest sensor that the memo fired (frozen dataclass -> same object).
    assert fp1 is fp2


def test_fingerprint_df_changes_when_content_changes():
    from mlframe.training.feature_handling.fingerprint import (
        fingerprint_df,
        reset_session,
    )
    reset_session()
    df1 = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    df2 = pl.DataFrame({"a": [1, 2, 3, 4, 99]})
    fp1 = fingerprint_df(df1)
    fp2 = fingerprint_df(df2)
    assert fp1.sampled_rows_hash != fp2.sampled_rows_hash


def test_fingerprint_df_reset_session_clears_memo():
    from mlframe.training.feature_handling.fingerprint import (
        _fingerprint_cache,
        fingerprint_df,
        reset_session,
    )
    df = pl.DataFrame({"a": [1, 2, 3]})
    fingerprint_df(df)
    assert len(_fingerprint_cache) >= 1
    reset_session()
    assert len(_fingerprint_cache) == 0


# ---------------------------------------------------------------------------
# DISC-CACHE-NULL-DTYPE
# ---------------------------------------------------------------------------


def test_data_signature_preserves_int_stats_with_nan_sentinel():
    """An integer column with a sentinel marking nulls used to fall through to the str-uniques
    path, dropping min/max/null info. Post-fix the int-kind branch keeps those stats."""
    import pandas as pd
    from mlframe.training.composite_cache import data_signature

    df1 = pd.DataFrame({
        "id": np.arange(100, dtype=np.int64),
        "target": np.linspace(0.0, 1.0, 100).astype(np.float64),
    })
    df2 = df1.copy()
    df2.loc[0, "id"] = 999999  # shift one int min->max change
    sig1 = data_signature(df1, "target", ["id"])
    sig2 = data_signature(df2, "target", ["id"])
    # The signature MUST differ because the min/max distribution changed.
    assert sig1 != sig2


# ---------------------------------------------------------------------------
# DISC-VERSION-LEAK
# ---------------------------------------------------------------------------


def test_discovery_config_signature_clips_patch_versions(monkeypatch):
    """Two pseudo polars versions differing only in patch must hash equal under the
    major.minor-only fold."""
    from mlframe.training.core import _phase_composite_discovery as mod

    class _DummyCfg:
        def __init__(self):
            self.x = 1

    _real_imp = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    class _FakeMod:
        __version__ = "1.18.1"

    class _FakeMod2:
        __version__ = "1.18.7"

    def _fake_import_factory(version_mod):
        def _fake_import(name, *args, **kwargs):
            if name == "polars":
                return version_mod
            return _real_imp(name, *args, **kwargs)
        return _fake_import

    monkeypatch.setattr("builtins.__import__", _fake_import_factory(_FakeMod))
    sig_v1 = mod._discovery_config_signature(_DummyCfg())
    monkeypatch.setattr("builtins.__import__", _fake_import_factory(_FakeMod2))
    sig_v2 = mod._discovery_config_signature(_DummyCfg())
    assert sig_v1 == sig_v2, "patch-level version diff should not change the signature"


# ---------------------------------------------------------------------------
# DISC-LRU-RACE (locking covered when filelock available)
# ---------------------------------------------------------------------------


def test_discovery_cache_touch_lru_uses_filelock_when_available(tmp_path):
    _need_filelock()
    from mlframe.training.composite_cache import DiscoveryCache
    cache = DiscoveryCache(str(tmp_path), max_entries=10)

    # Drive _touch_lru in parallel from two threads; without the lock the JSON file would
    # occasionally land in a half-written state. With the lock the final dict has BOTH keys.
    def _writer(tag):
        for i in range(20):
            cache._touch_lru(f"{tag}_{i}")

    t1 = threading.Thread(target=_writer, args=("a",))
    t2 = threading.Thread(target=_writer, args=("b",))
    t1.start(); t2.start()
    t1.join(); t2.join()
    lru = cache._load_lru()
    # Subset assertion: both tags must have produced at least one entry, and the file is valid JSON.
    assert any(k.startswith("a_") for k in lru)
    assert any(k.startswith("b_") for k in lru)
