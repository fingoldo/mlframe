"""Bit-identity + perf-sentinel regression tests for the njit ``_mi_from_binned_pair`` kernel.

``_mi_from_binned_pair`` is the pair-MI kernel called ~9.8k times per discovery run (per-feature
MI AND inside the per-permutation null loop in ``_auto_base``). It now dispatches to a
``numba.njit(cache=True)`` single-pass histogram+MI kernel that is bit-identical to the numpy
reference (``_mi_from_binned_pair_numpy``) within ~1e-12 -- the only divergence is the FP
reduction order of the final MI accumulation. These tests pin the bit-identity on random, tied,
and low-cardinality binned inputs (incl. the int16->int32 storage boundary at nbins>=182), plus a
warm-wall perf sentinel asserting the njit path is not slower than numpy at the production sample
size.
"""
from __future__ import annotations

from timeit import default_timer as timer

import numpy as np
import pytest

from mlframe.training.composite.discovery.screening import (
    _HAS_NUMBA,
    _mi_from_binned_pair,
    _mi_from_binned_pair_numpy,
)

_MAXDIFF = 1e-12


@pytest.fixture(scope="module", autouse=True)
def _warm_njit() -> None:
    """Warm the JIT once so per-test timings / first-call compile don't leak into asserts."""
    if _HAS_NUMBA:
        x = np.arange(1000, dtype=np.int16) % 16
        _mi_from_binned_pair(x, x, nbins=16)


def _codes(rng: np.random.Generator, n: int, nbins: int) -> np.ndarray:
    dtype = np.int16 if nbins < 182 else np.int32
    return rng.integers(0, nbins, n).astype(dtype)


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba unavailable; dispatcher falls back to numpy")
@pytest.mark.parametrize("nbins", [8, 16, 32, 64, 181, 182, 200])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_njit_bit_identical_random(nbins: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 20_000
    x = _codes(rng, n, nbins)
    y = _codes(rng, n, nbins)
    ref = _mi_from_binned_pair_numpy(x, y, nbins=nbins)
    got = _mi_from_binned_pair(x, y, nbins=nbins)
    assert abs(ref - got) < _MAXDIFF, f"nbins={nbins} seed={seed}: {ref} vs {got}"


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba unavailable")
def test_njit_bit_identical_tied_and_low_card() -> None:
    nbins = 16
    # All-constant codes -> MI exactly 0.0 on both paths.
    x0 = np.zeros(1000, dtype=np.int16)
    assert _mi_from_binned_pair(x0, x0, nbins=nbins) == _mi_from_binned_pair_numpy(
        x0, x0, nbins=nbins
    )
    # Perfectly dependent low-cardinality (2-level) -> MI = log(2), same on both.
    x2 = (np.arange(1000) % 2).astype(np.int16)
    assert abs(
        _mi_from_binned_pair(x2, x2, nbins=nbins) - _mi_from_binned_pair_numpy(x2, x2, nbins=nbins)
    ) < _MAXDIFF
    # Heavy ties: most mass in one cell, a few off-diagonal.
    rng = np.random.default_rng(3)
    x = np.zeros(5000, dtype=np.int16)
    y = np.zeros(5000, dtype=np.int16)
    idx = rng.choice(5000, 200, replace=False)
    x[idx] = rng.integers(0, nbins, 200).astype(np.int16)
    y[idx] = rng.integers(0, nbins, 200).astype(np.int16)
    assert abs(
        _mi_from_binned_pair(x, y, nbins=nbins) - _mi_from_binned_pair_numpy(x, y, nbins=nbins)
    ) < _MAXDIFF


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba unavailable")
def test_njit_handles_empty_and_single_row() -> None:
    nbins = 16
    empty = np.empty(0, dtype=np.int16)
    assert _mi_from_binned_pair(empty, empty, nbins=nbins) == 0.0
    one = np.array([3], dtype=np.int16)
    assert _mi_from_binned_pair(one, one, nbins=nbins) == _mi_from_binned_pair_numpy(
        one, one, nbins=nbins
    )


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba unavailable")
def test_njit_perf_sentinel_not_slower_than_numpy() -> None:
    """Warm multi-iter wall: njit must be at least as fast as numpy at the production size.

    Measured ~2.6-4.2x on dev hosts; the sentinel only asserts njit <= numpy * 1.05 so a
    future regression that silently routes the hot path back to the slower numpy kernel trips.
    """
    rng = np.random.default_rng(11)
    n, nbins = 20_000, 32
    x = _codes(rng, n, nbins)
    y = _codes(rng, n, nbins)

    def _wall(fn, repeats: int = 1500) -> float:
        fn()
        samples = []
        for _ in range(7):
            t0 = timer()
            for _ in range(repeats):
                fn()
            samples.append((timer() - t0) / repeats)
        samples.sort()
        return samples[len(samples) // 2]

    np_t = _wall(lambda: _mi_from_binned_pair_numpy(x, y, nbins=nbins))
    nj_t = _wall(lambda: _mi_from_binned_pair(x, y, nbins=nbins))
    assert nj_t <= np_t * 1.05, f"njit {nj_t * 1e6:.1f}us slower than numpy {np_t * 1e6:.1f}us"
