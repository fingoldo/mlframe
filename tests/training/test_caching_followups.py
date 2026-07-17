"""Regression sensors for Wave 10b A5 caching follow-ups.

Covers:
* SuiteArtefactCache `_evict_lru_locked` leaves orphan `.pkl.sha256` sidecars on disk after eviction; `_total_bytes_locked` ignores them so `total_bytes()` reports under cap while disk footprint includes the strays.
* `_PRE_PIPELINE_CACHE` X-content disambiguation: cache_key must distinguish two LRU slots when X frames have identical sampled cells but differ in unsampled rows.
* DiscoveryCache byte-size estimator helper symmetric to MRMR `_mrmr_instance_state_size_bytes`.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from mlframe.training.suite_artefact_cache import (
    SuiteArtefactCache,
    SuiteKeyBuilder,
)


# ---------------------------------------------------------------------------
# AP1 SuiteArtefactCache eviction bug: orphan sidecar leak
# ---------------------------------------------------------------------------


def _disk_files(cache_dir):
    """List every file under the cache directory; used to inspect physical residue."""
    out = []
    try:
        with os.scandir(cache_dir) as it:
            for de in it:
                if de.is_file():
                    out.append(de.name)
    except FileNotFoundError:
        return []
    return out


def test_evict_lru_removes_sidecars_under_max_entries(tmp_path):
    """After max-entries eviction, evicted .pkl files MUST also drop their .pkl.sha256 sidecars from disk.

    Pre-fix bug: `_evict_lru_locked` removed the .pkl successfully but silently swallowed any OSError on
    `os.remove(path + ".sha256")`, leaving sidecar orphans on disk. `_total_bytes_locked` only counted
    .pkl files, so `total_bytes()` returned under cap while the directory still held N stale sidecars
    (~64 bytes each) that nothing would ever clean up. Sensor proves: after writing N+2 entries to an
    N-entry cap, NO orphan .pkl.sha256 files remain whose .pkl peer is gone.
    """
    cache_dir = tmp_path / "ap1_evict"
    cache = SuiteArtefactCache(cache_dir=str(cache_dir), bytes_limit=10_000_000, max_entries=3)
    keys = []
    for i in range(7):
        k = SuiteKeyBuilder.build(df_fp=f"fp{i:03d}", config_canonical={"i": i})
        keys.append(k)
        cache.put(k, f"val_{i}")
    # Cap is 3, so 4 entries evicted.
    files = _disk_files(cache_dir)
    pkl_keys = {fn[:-4] for fn in files if fn.endswith(".pkl") and not fn.endswith(".pkl.sha256")}
    sidecar_keys = {fn[: -len(".pkl.sha256")] for fn in files if fn.endswith(".pkl.sha256")}
    orphan_sidecars = sidecar_keys - pkl_keys
    assert not orphan_sidecars, f"orphan .pkl.sha256 sidecars left on disk after eviction: {orphan_sidecars!r}"


def test_evict_lru_removes_sidecars_under_bytes_budget(tmp_path):
    """Same orphan-sidecar guard but triggered by the bytes-budget path."""
    cache_dir = tmp_path / "ap1_bytes"
    cache = SuiteArtefactCache(cache_dir=str(cache_dir), bytes_limit=4_096)
    payload = "x" * 1000
    for i in range(30):
        k = SuiteKeyBuilder.build(df_fp=f"fp{i:03d}", config_canonical={"i": i})
        cache.put(k, payload)
    files = _disk_files(cache_dir)
    pkl_keys = {fn[:-4] for fn in files if fn.endswith(".pkl") and not fn.endswith(".pkl.sha256")}
    sidecar_keys = {fn[: -len(".pkl.sha256")] for fn in files if fn.endswith(".pkl.sha256")}
    orphan_sidecars = sidecar_keys - pkl_keys
    assert not orphan_sidecars, f"orphan .pkl.sha256 sidecars left on disk after bytes eviction: {orphan_sidecars!r}"


def test_total_bytes_accounts_for_sidecars(tmp_path):
    """`total_bytes()` must reflect the FULL on-disk footprint -- pkl + pkl.sha256.

    Pre-fix `_total_bytes_locked` summed only .pkl, so the operator's budget check was always
    optimistic by ~64 bytes per cached entry. With N=1000 entries that's 64KB unaccounted
    (small in absolute terms but enough to make a sub-1MB budget completely misleading).
    """
    cache_dir = tmp_path / "ap1_total"
    cache = SuiteArtefactCache(cache_dir=str(cache_dir), bytes_limit=10_000_000)
    for i in range(5):
        k = SuiteKeyBuilder.build(df_fp=f"fp{i:03d}", config_canonical={"i": i})
        cache.put(k, {"data": list(range(50))})
    reported = cache.total_bytes()
    actual = 0
    with os.scandir(cache_dir) as it:
        for de in it:
            if de.is_file():
                actual += de.stat().st_size
    assert reported == actual, f"total_bytes() reported {reported} but on-disk footprint is {actual} (delta {actual - reported} -- sidecars not counted)"


# ---------------------------------------------------------------------------
# _PRE_PIPELINE_CACHE x-content disambiguation
# ---------------------------------------------------------------------------


def test_pre_pipeline_cache_key_distinguishes_x_content_with_shared_sample_cells():
    """`_pre_pipeline_cache_key` builds its X fingerprint from 4 sampled rows; two frames with identical
    sample positions but different unsampled rows should NOT collide (mirrors the MRMR full-x-hash fix).
    """
    from mlframe.training.pipeline._pipeline_cache import _pre_pipeline_cache_key

    rng = np.random.default_rng(0)
    n = 100
    base = rng.normal(size=(n, 3))
    # Frame A: original rng output.
    df_a = pd.DataFrame(base, columns=["x", "y", "z"])
    # Frame B: differs only at unsampled rows (1..n-2 except the midpoint).
    arr_b = base.copy()
    sampled_idx = {0, min(8, n - 1), n // 2, n - 1}
    flip = [i for i in range(n) if i not in sampled_idx]
    arr_b[flip[:10]] *= -1
    df_b = pd.DataFrame(arr_b, columns=["x", "y", "z"])
    y = pd.Series(np.arange(n))
    key_a = _pre_pipeline_cache_key(df_a, df_a, pipeline=None, train_target=y, target_name="t")
    key_b = _pre_pipeline_cache_key(df_b, df_b, pipeline=None, train_target=y, target_name="t")
    # Without a stronger X discriminator the two keys collide because the 4-row sample stays identical.
    # The fix folds a full-frame blake2b into the key so any unsampled-row drift busts the cache.
    assert key_a != key_b, (
        "pre-pipeline cache key collides when X frames differ only at unsampled rows -- would replay the wrong fit-transform output across targets"
    )


# ---------------------------------------------------------------------------
# DiscoveryCache byte-size estimator helper
# ---------------------------------------------------------------------------


def test_discovery_cache_bytes_total_helper_walks_pkl_files(tmp_path):
    """`_discovery_cache_bytes_total` should report the on-disk pkl+sidecar footprint so callers
    can compare against `max_size_mb` without inlining a directory walk.
    """
    from mlframe.training.composite.cache import (
        DiscoveryCache,
        _discovery_cache_bytes_total,
    )

    cache = DiscoveryCache(cache_dir=str(tmp_path), max_entries=10)
    for i in range(3):
        cache.set(f"k{i:02d}", {"i": i, "data": list(range(20))})
    reported = _discovery_cache_bytes_total(cache)
    actual = 0
    with os.scandir(tmp_path) as it:
        for de in it:
            if de.is_file() and (de.name.endswith(".pkl") or de.name.endswith(".pkl.sha256")):
                actual += de.stat().st_size
    assert reported == actual, f"_discovery_cache_bytes_total reported {reported}, on-disk pkl-set is {actual}"
