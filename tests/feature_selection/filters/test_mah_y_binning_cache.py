"""Wave 13 finding #4: mah_bin_edges/mah_mi re-derived y's quantile/label binning from scratch on EVERY column
call even though y is fit-constant across the whole per_feature_edges dispatch. _get_y_binning caches the binning
keyed on (id(y), K) with a weakref identity guard. These tests pin: (1) results are unchanged by the cache
(equivalence vs the pre-fix uncached computation), (2) the cache actually avoids recomputation on repeat calls
with the same y object, (3) different y objects (or y rebound to a new object after GC) never share a stale entry.
"""
import numpy as np

from mlframe.feature_selection.filters._mah import (
    _compute_y_binning,
    _get_y_binning,
    clear_mah_y_binning_cache,
    mah_bin_edges,
    mah_mi,
)


def test_get_y_binning_matches_uncached_compute():
    clear_mah_y_binning_cache()
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 500).astype(np.int64)
    K = 16
    expected = _compute_y_binning(y, K)
    yb, K_y = _get_y_binning(y, K)
    assert K_y == expected[1]
    assert np.array_equal(yb, expected[0])
    # second call with the SAME object hits the cache -- still identical result
    yb2, K_y2 = _get_y_binning(y, K)
    assert K_y2 == K_y
    assert np.array_equal(yb2, yb)


def test_get_y_binning_cache_hit_skips_recompute(monkeypatch):
    clear_mah_y_binning_cache()
    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, 500).astype(np.int64)
    calls = {"n": 0}
    orig = _compute_y_binning

    def counting_compute(y_arr, K):
        calls["n"] += 1
        return orig(y_arr, K)

    import mlframe.feature_selection.filters._mah as mah_mod
    monkeypatch.setattr(mah_mod, "_compute_y_binning", counting_compute)

    for _ in range(5):
        mah_mod._get_y_binning(y, 16)
    assert calls["n"] == 1, "cache hit should skip recompute for a repeated (id(y), K) key"


def test_mah_bin_edges_and_mah_mi_equivalent_across_repeated_calls_with_same_y():
    """Selection-equivalence: calling mah_bin_edges/mah_mi many times with different X columns but the SAME y
    object (the per_feature_edges dispatch shape) must give the same per-column result as calling each in
    isolation with a freshly-cleared cache -- the cache must never leak state across columns."""
    clear_mah_y_binning_cache()
    rng = np.random.default_rng(2)
    n = 2000
    y = rng.integers(0, 4, n).astype(np.int64)
    xs = [rng.normal(size=n) + i for i in range(6)]

    cached_edges = [mah_bin_edges(x, y, initial_k=16) for x in xs]
    cached_mi = [mah_mi(x, y, initial_k=16) for x in xs]

    # Recompute each column in total isolation (fresh cache, no cross-column reuse) as the ground truth.
    isolated_edges = []
    isolated_mi = []
    for x in xs:
        clear_mah_y_binning_cache()
        isolated_edges.append(mah_bin_edges(x, y, initial_k=16))
        clear_mah_y_binning_cache()
        isolated_mi.append(mah_mi(x, y, initial_k=16))

    for a, b in zip(cached_edges, isolated_edges):
        assert np.array_equal(a, b)
    for a, b in zip(cached_mi, isolated_mi):
        assert a == b


def test_get_y_binning_distinguishes_different_y_objects():
    clear_mah_y_binning_cache()
    rng = np.random.default_rng(3)
    y1 = rng.integers(0, 3, 300).astype(np.int64)
    y2 = rng.integers(0, 7, 300).astype(np.int64)
    yb1, Ky1 = _get_y_binning(y1, 16)
    yb2, Ky2 = _get_y_binning(y2, 16)
    assert Ky1 == 3
    assert Ky2 == 7
    assert not np.array_equal(yb1, yb2) or Ky1 != Ky2
