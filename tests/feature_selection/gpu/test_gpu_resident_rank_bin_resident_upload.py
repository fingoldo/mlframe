"""RESIDENT UPLOAD (wave 10, 2026-07-13): ``rank_bin_codes_gpu_resident`` / ``rank_bin_codes_batch_gpu_resident``
previously rebuilt (host cumsum) AND re-uploaded (H2D) the ``_bin_boundaries(n, n_bins)`` vector on EVERY
call, even though it depends ONLY on ``(n, n_bins)`` -- both fit-constant across a whole FE scan (n = the
dataset row count, n_bins = a fixed hyperparameter). A module-level ``_BIN_BOUNDARIES_CACHE`` (mirroring
``_gpu_resident_fe._QLEVELS_CACHE`` / ``_quantile_levels_dev`` exactly) now builds it once per ``(n, n_bins)``
and shares the device array across calls.

Skips when cupy is unavailable (CI without a GPU)."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._gpu_resident_rank_bin import (
    _BIN_BOUNDARIES_CACHE,
    _bin_boundaries,
    _bin_boundaries_dev,
    rank_bin_codes_batch_gpu_resident,
    rank_bin_codes_gpu_resident,
)
from mlframe.feature_selection.filters.hermite_fe import _quantile_bin_njit


@pytest.fixture(autouse=True)
def _clear_cache():
    _BIN_BOUNDARIES_CACHE.clear()
    yield
    _BIN_BOUNDARIES_CACHE.clear()


def test_rank_bin_codes_bit_identical_to_cpu_reference_tie_free():
    """Tie-free column: the GPU rank binner must be BIT-IDENTICAL to _quantile_bin_njit (per the module's
    own bit-identity contract docstring) -- a real regression pin, not previously tested anywhere."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000).astype(np.float64)  # continuous -> tie-free w.p. 1
    for nb in (10, 20, 7):
        host_codes = _quantile_bin_njit(x, nb)
        xg = cp.asarray(x)
        dev_codes = cp.asnumpy(rank_bin_codes_gpu_resident(xg, nb))
        assert np.array_equal(host_codes, dev_codes), f"nb={nb}: maxdiff={np.max(np.abs(host_codes.astype(np.int64) - dev_codes.astype(np.int64)))}"


def test_boundaries_dedup_across_calls_same_n_nbins(monkeypatch):
    """Two rank-bin calls on DIFFERENT columns but the SAME (n, n_bins) -- the realistic per-column FE-scan
    loop -- must upload the _bin_boundaries vector only ONCE."""
    n, nb = 6000, 12
    bnd_host = _bin_boundaries(n, nb)

    rng = np.random.default_rng(1)
    x1 = cp.asarray(rng.standard_normal(n))
    x2 = cp.asarray(rng.standard_normal(n))

    upload_calls = {"n": 0}
    orig_asarray = cp.asarray

    def _counting_asarray(arr, *a, **kw):
        if isinstance(arr, np.ndarray) and arr.shape == bnd_host.shape and arr.dtype == bnd_host.dtype and np.array_equal(arr, bnd_host):
            upload_calls["n"] += 1
        return orig_asarray(arr, *a, **kw)

    monkeypatch.setattr(cp, "asarray", _counting_asarray)

    c1 = rank_bin_codes_gpu_resident(x1, nb)
    c2 = rank_bin_codes_gpu_resident(x2, nb)

    assert c1 is not None and c2 is not None
    assert upload_calls["n"] == 1, f"_bin_boundaries({n},{nb}) upload called {upload_calls['n']} times across 2 calls (expected 1)"


def test_boundaries_dedup_batch_variant_shares_cache_with_single_column(monkeypatch):
    """The batched variant (rank_bin_codes_batch_gpu_resident) must hit the SAME (n, n_bins) cache entry
    the single-column variant already warmed -- one shared _BIN_BOUNDARIES_CACHE, not two separate caches."""
    n, nb = 4000, 8
    bnd_host = _bin_boundaries(n, nb)

    rng = np.random.default_rng(2)
    x1 = cp.asarray(rng.standard_normal(n))
    Xbatch = cp.asarray(rng.standard_normal((n, 5)))

    upload_calls = {"n": 0}
    orig_asarray = cp.asarray

    def _counting_asarray(arr, *a, **kw):
        if isinstance(arr, np.ndarray) and arr.shape == bnd_host.shape and arr.dtype == bnd_host.dtype and np.array_equal(arr, bnd_host):
            upload_calls["n"] += 1
        return orig_asarray(arr, *a, **kw)

    monkeypatch.setattr(cp, "asarray", _counting_asarray)

    rank_bin_codes_gpu_resident(x1, nb)  # warms the (n, nb) cache entry
    codes_batch = rank_bin_codes_batch_gpu_resident(Xbatch, nb)  # must reuse it, not re-upload

    assert codes_batch is not None
    assert upload_calls["n"] == 1, f"_bin_boundaries({n},{nb}) upload called {upload_calls['n']} times across single+batch calls (expected 1)"


def test_cached_boundaries_bit_identical_to_fresh_upload():
    """A cache HIT must return the exact same bytes a fresh cp.asarray(_bin_boundaries(...)) would -- the
    caching change only skips the redundant rebuild+H2D, never alters the values."""
    n, nb = 3333, 9
    fresh = cp.asarray(_bin_boundaries(n, nb))
    cold = _bin_boundaries_dev(cp, n, nb)  # first call -> builds + caches
    warm = _bin_boundaries_dev(cp, n, nb)  # second call -> cache HIT, same object
    assert warm is cold, "second call did not hit the cache (returned a different device array)"
    np.testing.assert_array_equal(cp.asnumpy(fresh), cp.asnumpy(cold))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
