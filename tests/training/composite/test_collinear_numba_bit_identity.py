"""Bit-identity + correctness regression tests for the numba near-collinear dedup.

The numba-JIT pair walk in ``_collinear_numba.near_collinear_keep_mask_fast`` must
return a keep-mask BIT-IDENTICAL to the pure-numpy reference
``_eval_stats._near_collinear_keep_mask_numpy`` for every input class: continuous,
discrete / tied, NaN-holed, exact-duplicate (corr exactly 1.0), and degenerate
(constant) columns. These tests pin that across many seeds and the degenerate
case so a future kernel change cannot silently diverge the selection.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery import _collinear_numba as _cn
from mlframe.training.composite.discovery._collinear_numba import (
    _HAS_NUMBA,
    _MIN_COLS,
    _MIN_ROWS,
    near_collinear_keep_mask_fast,
)
from mlframe.training.composite.discovery._eval_stats import (
    _near_collinear_keep_mask_numpy,
    near_collinear_keep_mask,
)


def _fast(fm: np.ndarray, thr: float) -> np.ndarray:
    return near_collinear_keep_mask_fast(
        fm, corr_threshold=thr, reference_fn=_near_collinear_keep_mask_numpy,
    )


def _make_matrix(seed: int) -> tuple[np.ndarray, float]:
    """A mixed matrix big enough to hit the JIT path: some near-duplicate columns
    of a few latent bases, some independent columns, sometimes NaN-holed."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(_MIN_ROWS, 3000))
    n_cols = int(rng.integers(_MIN_COLS, 45))
    latent = rng.normal(size=(n, 4))
    cols = []
    for _ in range(n_cols):
        r = rng.random()
        if r < 0.4:
            noise = rng.choice([1e-5, 1e-3, 1e-2, 0.4])
            cols.append(latent[:, rng.integers(0, 4)] + noise * rng.normal(size=n))
        elif r < 0.5:
            # discrete / tied column (low cardinality) -- the ULP-flip danger zone.
            cols.append(rng.integers(0, 5, size=n).astype(np.float64))
        else:
            cols.append(rng.normal(size=n))
    fm = np.column_stack(cols)
    if rng.random() < 0.4:
        holes = rng.random((n, n_cols)) < 0.08
        fm[holes] = np.nan
    thr = float(rng.choice([0.9, 0.95, 0.99]))
    return fm, thr


@pytest.mark.parametrize("seed", range(40))
def test_fast_mask_bit_identical_to_numpy_reference(seed: int) -> None:
    """JIT keep-mask equals the numpy reference across continuous/discrete/NaN seeds."""
    fm, thr = _make_matrix(seed)
    ref = _near_collinear_keep_mask_numpy(fm, corr_threshold=thr)
    fast = _fast(fm, thr)
    assert np.array_equal(ref, fast), (
        f"seed={seed} kept ref={ref.sum()} fast={fast.sum()}"
    )


def test_exact_duplicate_columns_dropped_identically() -> None:
    """Exact-duplicate columns (corr exactly 1.0) drop identically to the reference."""
    rng = np.random.default_rng(7)
    n = 1000
    base = rng.normal(size=(n, 3))
    # cols 0..2 latent, 3=dup of 0, 4=dup of 1, plus independents to reach JIT size.
    cols = [base[:, 0], base[:, 1], base[:, 2], base[:, 0].copy(), base[:, 1].copy()]
    cols += [rng.normal(size=n) for _ in range(_MIN_COLS)]
    fm = np.column_stack(cols)
    ref = _near_collinear_keep_mask_numpy(fm, corr_threshold=0.99)
    fast = _fast(fm, 0.99)
    assert np.array_equal(ref, fast)
    # The two exact duplicates must be dropped.
    assert not fast[3] and not fast[4]


def test_degenerate_constant_column_kept_identically() -> None:
    """A constant (zero-variance) column never correlates -> always kept; identical."""
    rng = np.random.default_rng(11)
    n = 1500
    a = rng.normal(size=n)
    cols = [a, a + 1e-9 * rng.normal(size=n)]  # near-duplicate pair
    cols.append(np.full(n, 3.14))  # constant column
    cols.append(np.zeros(n))  # another constant
    cols += [rng.normal(size=n) for _ in range(_MIN_COLS)]
    fm = np.column_stack(cols)
    ref = _near_collinear_keep_mask_numpy(fm, corr_threshold=0.99)
    fast = _fast(fm, 0.99)
    assert np.array_equal(ref, fast)
    assert fast[2] and fast[3], "constant columns must be kept"


def test_public_dispatch_matches_reference_on_large_input() -> None:
    """The public ``near_collinear_keep_mask`` (dispatcher) equals the numpy reference."""
    fm, thr = _make_matrix(123)
    ref = _near_collinear_keep_mask_numpy(fm, corr_threshold=thr)
    pub = near_collinear_keep_mask(fm, corr_threshold=thr)
    assert np.array_equal(ref, pub)


def test_small_input_uses_reference_path() -> None:
    """Below the size gate the dispatcher returns the exact reference mask."""
    rng = np.random.default_rng(5)
    a = rng.normal(size=400)
    fm = np.column_stack([a, a + 1e-4 * rng.normal(size=400), rng.normal(size=400)])
    ref = _near_collinear_keep_mask_numpy(fm, corr_threshold=0.99)
    fast = _fast(fm, 0.99)
    assert np.array_equal(ref, fast)
    assert fast.tolist() == [True, False, True]


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba required for the JIT path")
def test_keep_mask_cache_hit_is_bit_identical_to_fresh() -> None:
    """A second call on a byte-identical matrix must hit the per-suite cache and
    return a mask BIT-IDENTICAL to the from-scratch (cache-cleared) recompute --
    the cache must never change the selection. Mirrors targets 2..N reusing a
    base's shared feature matrix in discovery."""
    _cn._KEEP_MASK_CACHE.clear()
    fm, thr = _make_matrix(321)
    # Fresh from-scratch (cache empty).
    fresh = _fast(fm, thr)
    assert len(_cn._KEEP_MASK_CACHE) >= 1, "cacheable matrix should populate the cache"
    # Byte-identical copy -> cache hit -> must equal fresh.
    hit = _fast(fm.copy(), thr)
    assert np.array_equal(fresh, hit)
    # And equal to a fully fresh recompute (cache cleared) -- pins cached == fresh.
    _cn._KEEP_MASK_CACHE.clear()
    refresh = _fast(fm, thr)
    assert np.array_equal(fresh, refresh)


def test_keep_mask_cache_isolates_distinct_matrices() -> None:
    """A different matrix content must not collide with a cached entry: each
    matrix gets its own mask (collision-safe content key)."""
    _cn._KEEP_MASK_CACHE.clear()
    fm_a, thr = _make_matrix(11)
    fm_b, _ = _make_matrix(404)
    mask_a = _fast(fm_a, thr)
    mask_b = _fast(fm_b, thr)
    # Recompute A fresh; must still equal mask_a (no stale-B contamination).
    _cn._KEEP_MASK_CACHE.clear()
    assert np.array_equal(mask_a, _fast(fm_a, thr))
    # A different threshold on the SAME matrix is a distinct key (no false hit).
    _cn._KEEP_MASK_CACHE.clear()
    m_lo = _fast(fm_a, 0.90)
    m_hi = _fast(fm_a, 0.999)
    ref_lo = _near_collinear_keep_mask_numpy(fm_a, corr_threshold=0.90)
    ref_hi = _near_collinear_keep_mask_numpy(fm_a, corr_threshold=0.999)
    assert np.array_equal(m_lo, ref_lo)
    assert np.array_equal(m_hi, ref_hi)


def test_keep_mask_cache_returned_mask_is_mutation_safe() -> None:
    """The cache returns a COPY, so a caller mutating the result must not corrupt
    the stored entry (discovery indexes/reassigns the mask downstream)."""
    _cn._KEEP_MASK_CACHE.clear()
    fm, thr = _make_matrix(77)
    first = _fast(fm, thr)
    first[:] = False  # caller mutates the returned mask in place.
    second = _fast(fm.copy(), thr)  # cache hit.
    assert second.any() or not first.any(), "cached entry must survive caller mutation"
    # second must equal a clean recompute, not the all-False mutated array.
    _cn._KEEP_MASK_CACHE.clear()
    clean = _fast(fm, thr)
    assert np.array_equal(second, clean)


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba required for the JIT path")
def test_threshold_disabled_and_degenerate_shapes() -> None:
    """thr>=1.0 disables dedup; 1-column / <3-row matrices keep everything."""
    rng = np.random.default_rng(9)
    a = rng.normal(size=_MIN_ROWS + 100)
    fm = np.column_stack([a] * _MIN_COLS)
    assert _fast(fm, 1.0).all()  # threshold disables -> all kept
    assert _fast(a.reshape(-1, 1), 0.5).all()
    assert _fast(fm[:2], 0.5).all()


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba required for the JIT path")
class TestAllFiniteFastPath:
    """The all-finite fast path precomputes per-column mean+ssq once, then costs ONE cross-term pass per kept pair (vs the two-pass per-pair mean/variance recompute of the general kernel). It must be selected on a finite matrix and produce a mask bit-identical to both the serial NaN-aware kernel and the numpy reference."""

    def test_allfinite_kernel_symbols_exist(self) -> None:
        assert hasattr(_cn, "_keep_mask_kernel_allfinite")
        assert hasattr(_cn, "_column_stats_allfinite")

    @pytest.mark.parametrize("seed", range(12))
    def test_allfinite_fast_path_bit_identical_to_serial_and_reference(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n = int(rng.integers(_MIN_ROWS, 4000))
        n_cols = int(rng.integers(_MIN_COLS, 40))
        latent = rng.normal(size=(n, 5))
        cols = []
        for _ in range(n_cols):
            r = rng.random()
            if r < 0.45:
                cols.append(latent[:, rng.integers(0, 5)] + rng.choice([1e-5, 1e-2, 0.3]) * rng.normal(size=n))
            elif r < 0.55:
                cols.append(rng.integers(0, 5, size=n).astype(np.float64))  # discrete / tied
            else:
                cols.append(rng.normal(size=n))
        fm = np.ascontiguousarray(np.column_stack(cols), dtype=np.float64)
        assert np.isfinite(fm).all()
        thr = float(rng.choice([0.9, 0.95, 0.99]))
        mean, var = _cn._column_stats_allfinite(fm)
        keep_af, _ = _cn._keep_mask_kernel_allfinite(fm, mean, var, thr, _cn._BORDERLINE_BAND)
        keep_serial, _ = _cn._keep_mask_kernel(fm, np.isfinite(fm), thr, _cn._BORDERLINE_BAND)
        assert np.array_equal(keep_af, keep_serial)
        ref = _near_collinear_keep_mask_numpy(fm, corr_threshold=thr)
        assert np.array_equal(keep_af.astype(bool), ref)

    def test_dispatcher_uses_allfinite_path_on_finite_input(self, monkeypatch) -> None:
        """On an all-finite matrix above the size gate, the dispatcher MUST route through the all-finite kernel (and NOT the general NaN-aware one)."""
        rng = np.random.default_rng(3)
        n, B = _MIN_ROWS + 500, _MIN_COLS + 4
        fm = np.ascontiguousarray(rng.normal(size=(n, B)))
        fm[:, 2] = fm[:, 1] + 1e-4 * rng.normal(size=n)
        _cn._KEEP_MASK_CACHE.clear()
        calls = {"af": 0, "general": 0}
        orig_af = _cn._keep_mask_kernel_allfinite
        orig_gen = _cn._keep_mask_kernel

        def spy_af(*a, **k):
            calls["af"] += 1
            return orig_af(*a, **k)

        def spy_gen(*a, **k):
            calls["general"] += 1
            return orig_gen(*a, **k)

        monkeypatch.setattr(_cn, "_keep_mask_kernel_allfinite", spy_af)
        monkeypatch.setattr(_cn, "_keep_mask_kernel", spy_gen)
        _fast(fm, 0.99)
        assert calls["af"] == 1 and calls["general"] == 0
        _cn._KEEP_MASK_CACHE.clear()

    def test_nan_input_still_uses_general_kernel(self, monkeypatch) -> None:
        """A matrix with any NaN must fall back to the general NaN-aware kernel (the all-finite cross-term formula is wrong on holed pairs)."""
        rng = np.random.default_rng(4)
        n, B = _MIN_ROWS + 500, _MIN_COLS + 4
        fm = np.ascontiguousarray(rng.normal(size=(n, B)))
        fm[3, 0] = np.nan
        _cn._KEEP_MASK_CACHE.clear()
        calls = {"af": 0, "general": 0}
        orig_af = _cn._keep_mask_kernel_allfinite
        orig_gen = _cn._keep_mask_kernel
        monkeypatch.setattr(_cn, "_keep_mask_kernel_allfinite", lambda *a, **k: (calls.__setitem__("af", calls["af"] + 1) or orig_af(*a, **k)))
        monkeypatch.setattr(_cn, "_keep_mask_kernel", lambda *a, **k: (calls.__setitem__("general", calls["general"] + 1) or orig_gen(*a, **k)))
        _fast(fm, 0.99)
        assert calls["general"] == 1 and calls["af"] == 0
        _cn._KEEP_MASK_CACHE.clear()
