"""Prebin-matrix cache: bit-identity + behaviour + size-gate + biz_value.

``_prebin_feature_columns`` turns the small screen-sized float feature matrix into an int16/int32
bin-code matrix via per-column ``np.quantile`` + ``np.searchsorted`` (O(n*F*log n)). The codes are
a DETERMINISTIC function of (matrix bytes, nbins) only, so a second discovery on the same data +
sample + nbins but a different config recomputes the identical codes. ``PrebinCache`` (cache.py)
keys those codes by a content hash so the second run skips the binning and reuses bit-identical
codes.

These tests pin the full contract:

* cached codes are BIT-IDENTICAL to a fresh recompute (the optimization changes nothing numerically);
* the signature hits iff a recompute would reproduce byte-identical codes (and misses on any change);
* the per-entry byte ceiling refuses an oversized code matrix (the 100GB-frame guard);
* biz_value: a second prebin on the same matrix is materially faster via the cache than recomputing.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mlframe.training.composite.cache import (
    PrebinCache,
    get_prebin_cache,
    prebin_matrix_signature,
)
from mlframe.training.composite.discovery.screening import (
    _prebin_feature_columns,
    _prebin_feature_columns_cached,
)


def _make_matrix(n=4000, f=12, seed=11, nan_frac=0.0):
    """Make matrix."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, f)).astype(np.float32)
    if nan_frac > 0:
        mask = rng.random((n, f)) < nan_frac
        x[mask] = np.nan
    return x


@pytest.mark.parametrize("nbins", [10, 16, 200])
@pytest.mark.parametrize("nan_frac", [0.0, 0.05])
def test_cached_codes_bit_identical_to_recompute(nbins, nan_frac):
    """Cache HIT returns codes byte-identical to a fresh ``_prebin_feature_columns``."""
    cache = PrebinCache()
    x = _make_matrix(nan_frac=nan_frac)
    fresh = _prebin_feature_columns(x, nbins=nbins)

    sig = prebin_matrix_signature(x, nbins)
    assert cache.get(sig) is None  # cold miss
    cache.put(sig, fresh.copy())
    hit = cache.get(sig)
    assert hit is not None
    # Bit-identical: same dtype, same shape, same values element-for-element.
    assert hit.dtype == fresh.dtype
    assert hit.shape == fresh.shape
    assert np.array_equal(hit, fresh)


@pytest.mark.parametrize("nbins", [16, 200])
def test_wrapper_second_call_returns_identical_codes(nbins):
    """``_prebin_feature_columns_cached`` second call (cache hit) is bit-identical to first."""
    get_prebin_cache().clear()
    x = _make_matrix()
    first = _prebin_feature_columns_cached(x, nbins=nbins)
    second = _prebin_feature_columns_cached(x, nbins=nbins)
    direct = _prebin_feature_columns(x, nbins=nbins)
    assert np.array_equal(first, direct)
    assert np.array_equal(second, direct)
    # Second call must have hit the cache (the codes object is the same reference held by cache).
    assert second is first


def test_use_cache_false_bypasses_store_and_lookup():
    """``use_cache=False`` recomputes and never touches the cache (force-fresh path)."""
    cache = get_prebin_cache()
    cache.clear()
    x = _make_matrix()
    out = _prebin_feature_columns_cached(x, nbins=16, use_cache=False)
    assert np.array_equal(out, _prebin_feature_columns(x, nbins=16))
    assert len(cache) == 0  # nothing stored


def test_signature_changes_on_any_relevant_input_change():
    """Signature hits iff a recompute reproduces byte-identical codes; misses on any change."""
    x = _make_matrix(seed=1)
    base = prebin_matrix_signature(x, 16)
    # Same array, same nbins -> same signature (cache would hit).
    assert prebin_matrix_signature(x.copy(), 16) == base
    # Different nbins -> different codes -> different signature.
    assert prebin_matrix_signature(x, 32) != base
    # Different values -> different signature.
    x2 = x.copy()
    x2[0, 0] += 1.0
    assert prebin_matrix_signature(x2, 16) != base
    # Different shape (extra column) -> different signature.
    x3 = np.column_stack([x, x[:, :1]])
    assert prebin_matrix_signature(x3, 16) != base
    # Different dtype -> different signature (codes differ via quantile precision path).
    assert prebin_matrix_signature(x.astype(np.float64), 16) != base


def test_oversize_code_matrix_not_cached():
    """100GB-frame guard: a code matrix above the per-entry ceiling is refused (returns False)."""
    cache = PrebinCache(max_bytes=1024)  # tiny ceiling: 1 KiB
    x = _make_matrix(n=2000, f=20)
    codes = _prebin_feature_columns(x, nbins=16)
    assert codes.nbytes > 1024
    stored = cache.put(prebin_matrix_signature(x, 16), codes)
    assert stored is False
    assert len(cache) == 0
    assert cache.skipped_oversize == 1


def test_lru_eviction_caps_entry_count():
    """The cache evicts least-recently-used entries past ``max_entries``."""
    cache = PrebinCache(max_entries=2)
    for k in range(4):
        x = _make_matrix(seed=k, n=500, f=4)
        cache.put(prebin_matrix_signature(x, 8), _prebin_feature_columns(x, nbins=8))
    assert len(cache) == 2  # only the 2 most-recent survive


def test_biz_val_prebin_cache_second_call_faster():
    """biz_value: cache hit on a re-prebin is materially faster than recomputing the binning.

    Floor 3x (conservative). The cache returns the stored codes in O(hash) vs the recompute's
    O(n*F*log n) per-column quantile + searchsorted. Use a matrix large enough that the binning
    dominates the signature hash so the win is unambiguous.
    """
    get_prebin_cache().clear()
    x = _make_matrix(n=80_000, f=40, seed=3)
    nbins = 32

    # Cold: compute + store (warm-up the signature path once so we time steady-state).
    _prebin_feature_columns_cached(x, nbins=nbins)

    # Recompute cost (force-fresh path: full binning every call).
    t0 = time.perf_counter()
    for _ in range(5):
        _prebin_feature_columns_cached(x, nbins=nbins, use_cache=False)
    t_recompute = (time.perf_counter() - t0) / 5

    # Cache-hit cost (signature hash + dict lookup only).
    t0 = time.perf_counter()
    for _ in range(5):
        _prebin_feature_columns_cached(x, nbins=nbins, use_cache=True)
    t_cached = (time.perf_counter() - t0) / 5

    speedup = t_recompute / max(t_cached, 1e-9)
    assert speedup >= 3.0, (
        f"prebin cache hit should be >=3x faster than recompute; recompute={t_recompute * 1e3:.2f}ms cached={t_cached * 1e3:.2f}ms speedup={speedup:.1f}x"
    )
