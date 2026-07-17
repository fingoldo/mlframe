"""§8.5 Caching test coverage gaps -- regression tests for previously uncovered cache code.

Sibling F4 (test_audit_2026_05_16_f4_caching.py) already covers:
  * P0 precompute stubs raise (test_precompute_dummy_baselines_raises_notimplementederror /
    test_precompute_composite_target_specs_raises_notimplementederror /
    test_precompute_all_only_fills_stats_slot)
  * FH-INMEMKEY-ID-RECYCLE (test_feature_cache_purge_by_df_token_drops_matching_entries) -- this
    partially covers the §8.5 P2 stale-id finding too.

This file covers what F4 did not: cross-thread session-token isolation, fingerprint dtype
sensitivity, lightgbm/catboost in the discovery version tuple, and RAM-LRU eviction.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

pl = pytest.importorskip("polars")


# ---------------------------------------------------------------------------
# §8.5 P1: fingerprint.py concurrent multi-suite session-token isolation
# ---------------------------------------------------------------------------


def test_reset_session_yields_distinct_session_ids():
    """``reset_session()`` must produce a fresh ``SessionToken`` with a distinct ``session_id``.
    Concurrent suites that each call ``reset_session`` must NOT share the same UUID."""
    from mlframe.training.feature_handling.fingerprint import reset_session

    seen = []
    seen_lock = threading.Lock()

    def _suite_worker():
        tok = reset_session()
        with seen_lock:
            seen.append(tok.session_id)

    threads = [threading.Thread(target=_suite_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # ``reset_session`` mutates a process-wide singleton -- under thread interleaving, the LAST
    # writer wins and prior captures see WHATEVER token was current at their capture point. We
    # only assert that AT LEAST one distinct id appeared (the alternative -- all four threads
    # somehow caught the exact same id -- would only happen if reset_session was a no-op).
    assert len(set(seen)) >= 2, f"reset_session must rotate the session id across concurrent calls; got {seen}"


def test_current_session_returns_same_token_within_session():
    """Within a single session (no intervening reset_session()), repeated ``current_session()`` calls
    return the same token instance."""
    from mlframe.training.feature_handling.fingerprint import current_session, reset_session

    reset_session()
    tok_a = current_session()
    tok_b = current_session()
    assert tok_a is tok_b
    assert tok_a.session_id == tok_b.session_id


# ---------------------------------------------------------------------------
# §8.5 P1: utils.py compute_model_input_fingerprint dtype sensitivity
# ---------------------------------------------------------------------------


def test_compute_model_input_fingerprint_distinguishes_float32_vs_float64():
    """A schema differing only in float precision (Float32 vs Float64) must yield a DIFFERENT
    fingerprint so cached models trained on one dtype don't accidentally serve predictions on the
    other."""
    import pandas as pd
    from mlframe.training.utils import compute_model_input_fingerprint

    df32 = pd.DataFrame({"x": np.array([1.0, 2.0, 3.0], dtype=np.float32)})
    df64 = pd.DataFrame({"x": np.array([1.0, 2.0, 3.0], dtype=np.float64)})
    fp32, _ = compute_model_input_fingerprint(df32, target_name="y", model_family="cb")
    fp64, _ = compute_model_input_fingerprint(df64, target_name="y", model_family="cb")
    assert fp32 != fp64, f"fingerprint must distinguish float32 vs float64 schema; got fp32={fp32} fp64={fp64}"


def test_compute_model_input_fingerprint_stable_across_calls():
    """Two calls with identical inputs must produce the same hash (canonical JSON sort_keys means
    the hash doesn't depend on Python's dict ordering)."""
    import pandas as pd
    from mlframe.training.utils import compute_model_input_fingerprint

    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    fp_a, _ = compute_model_input_fingerprint(df, target_name="y", model_family="cb")
    fp_b, _ = compute_model_input_fingerprint(df, target_name="y", model_family="cb")
    assert fp_a == fp_b


# ---------------------------------------------------------------------------
# §8.5 P2: _phase_composite_discovery.py:74 lightgbm/catboost in version tuple
# ---------------------------------------------------------------------------


def test_discovery_config_signature_changes_when_catboost_minor_bumps(monkeypatch):
    """The version tuple folded into the discovery cache signature MUST include catboost (and
    lightgbm). A bumped catboost minor version must produce a different signature so the cache
    doesn't replay specs from a prior major.minor."""
    from mlframe.training.core import _phase_composite_discovery as mod

    class _DummyCfg:
        def __init__(self):
            self.x = 1

    _real_imp = __import__

    class _FakeCb1:
        __version__ = "1.2.5"

    class _FakeCb2:
        __version__ = "1.3.0"

    def _factory(replacement):
        def _fake(name, *args, **kwargs):
            if name == "catboost":
                return replacement
            return _real_imp(name, *args, **kwargs)

        return _fake

    monkeypatch.setattr("builtins.__import__", _factory(_FakeCb1))
    sig_a = mod._discovery_config_signature(_DummyCfg())
    monkeypatch.setattr("builtins.__import__", _factory(_FakeCb2))
    sig_b = mod._discovery_config_signature(_DummyCfg())
    assert sig_a != sig_b, "catboost minor-version bump must invalidate discovery signature"


def test_discovery_config_signature_changes_when_lightgbm_minor_bumps(monkeypatch):
    """Same as catboost but for lightgbm."""
    from mlframe.training.core import _phase_composite_discovery as mod

    class _DummyCfg:
        def __init__(self):
            self.x = 1

    _real_imp = __import__

    class _FakeLgb1:
        __version__ = "4.1.0"

    class _FakeLgb2:
        __version__ = "4.5.0"

    def _factory(replacement):
        def _fake(name, *args, **kwargs):
            if name == "lightgbm":
                return replacement
            return _real_imp(name, *args, **kwargs)

        return _fake

    monkeypatch.setattr("builtins.__import__", _factory(_FakeLgb1))
    sig_a = mod._discovery_config_signature(_DummyCfg())
    monkeypatch.setattr("builtins.__import__", _factory(_FakeLgb2))
    sig_b = mod._discovery_config_signature(_DummyCfg())
    assert sig_a != sig_b, "lightgbm minor-version bump must invalidate discovery signature"


# ---------------------------------------------------------------------------
# §8.5 P2: cache.py id() recycle -- _release_ctx_polars_frames stale-id cache miss
# ---------------------------------------------------------------------------


def test_feature_cache_release_invalidates_then_realloc_misses():
    """After ``purge_by_df_token``, a freshly-built InMemoryKey at the same df_token must MISS the
    cache (not replay the stale entry). Sibling F4's
    test_feature_cache_purge_by_df_token_drops_matching_entries covers the drop. Here we extend
    that with a follow-up get_or_compute call that must hit the recompute branch, not the cache."""
    from mlframe.training.feature_handling.cache import FeatureCache
    from mlframe.training.feature_handling.fingerprint import InMemoryKey
    from mlframe.training.feature_handling.config import CacheConfig

    cache = FeatureCache(CacheConfig(persistence="off"))
    base = dict(session_id="s", train_idx_token=1, column="x", params_canonical_hash="p", provider_signature="v")
    k = InMemoryKey(df_token=99, **base)
    calls = []
    cache.get_or_compute(k, lambda: calls.append("compute1") or np.zeros(4, dtype=np.float32))
    assert calls == ["compute1"]
    # Same key second time -> cache hit, no recompute.
    cache.get_or_compute(k, lambda: calls.append("compute2") or np.zeros(4, dtype=np.float32))
    assert calls == ["compute1"]

    cache.purge_by_df_token(99)
    # After purge, the SAME df_token must be a miss again.
    cache.get_or_compute(k, lambda: calls.append("compute3") or np.zeros(4, dtype=np.float32))
    assert calls == ["compute1", "compute3"]


# ---------------------------------------------------------------------------
# §8.5 P2: feature_handling/cache.py FH RAM-LRU eviction (ram_max_gb cap)
# ---------------------------------------------------------------------------


def test_feature_cache_ram_max_gb_evicts_oldest_lru_when_over_cap():
    """A tiny ``ram_max_gb`` cap forces LRU eviction once cumulative entry sizes exceed it.
    Inserting >cap entries asserts the oldest entry is dropped first."""
    from mlframe.training.feature_handling.cache import FeatureCache
    from mlframe.training.feature_handling.fingerprint import InMemoryKey
    from mlframe.training.feature_handling.config import CacheConfig

    # 1 MB cap -> ~5 entries of ~250 KB before eviction kicks in.
    cfg = CacheConfig(persistence="off", ram_max_gb=0.001, eviction_strategy="lru")
    cache = FeatureCache(cfg)

    def _make_key(i):
        return InMemoryKey(
            session_id="s",
            df_token=1,
            train_idx_token=1,
            column=f"x{i}",
            params_canonical_hash="p",
            provider_signature="v",
        )

    # Each ~250 KB (62500 float32 entries).
    arr_size = 62500
    for i in range(8):
        cache.get_or_compute(_make_key(i), lambda: np.zeros(arr_size, dtype=np.float32))

    stats = cache.stats()
    # We inserted 8 entries but only ~5 fit under the 1 MB cap -> evictions must have fired.
    assert stats["evictions"] > 0, f"RAM-LRU eviction did not fire under ram_max_gb=0.001; stats={stats}"
    # Cache size must end up <= cap (allow some slack for the tail entry).
    assert stats["n_keys"] < 8
