"""Bench + bit-identity check: ``_dyadic_haar_leg_njit`` (parallel-over-rows) vs its serial twin
``_dyadic_haar_leg_njit_serial``, and the ``_HAAR_LEG_PARALLEL_MIN_N`` threshold in ``_dyadic_haar_leg``.

Finding (2026-08-23): the njit-fused Haar-leg builder (``_dyadic_haar_leg_njit``, added 2026-08-03 as a
genuine win over the original 4-pass numpy form) was ``@njit(parallel=True)`` with ``prange`` over ROWS.
A 3-way compare-and-write is too trivial per-element to amortize numba's fixed per-call thread-pool
dispatch cost at any n this codebase's FE search realistically reaches. Measured here (same-process A/B,
warm, best-of-30-200):

    n=       600: parallel/serial ~12681x SLOWER
    n=     1_500: parallel/serial ~10693x SLOWER
    n=    15_000: parallel/serial   ~597x SLOWER
    n=   100_000: parallel/serial    ~38x SLOWER
    n= 1_000_000: parallel/serial    ~17x SLOWER
    n= 5_000_000: parallel/serial   ~4.6x SLOWER
    n=20_000_000: parallel/serial   ~1.2x SLOWER (near breakeven)
    n=50_000_000: parallel/serial   ~0.75x (1.33x FASTER)

Dispatch is via the per-host kernel_tuning_cache (``_HAAR_LEG_PARALLELISM_SPEC``, no hardcoded
threshold) -- ``_haar_leg_fallback_choice`` (used pre-sweep / on tuner failure) routes at
n=20_000_000, the near-breakeven point, biased toward the serial side: an unnecessary serial call at
n~20-50M costs at most tens of ms, an unnecessary parallel call below it costs up to 12681x per the
ratios above.

Run: ``python profiling/bench_dyadic_haar_leg_parallel_vs_serial.py``
"""

from __future__ import annotations

import time

import numpy as np

from mlframe.feature_selection.filters._wavelet_basis_fe import (
    _dyadic_haar_leg,
    _dyadic_haar_leg_njit,
    _dyadic_haar_leg_njit_serial,
    _haar_leg_fallback_choice,
)

_LEFT, _MID, _RIGHT = 0.0, 0.125, 0.25


def _make_z(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.random(n), dtype=np.float64)


def _bench(n: int, n_calls: int = 30) -> None:
    z = _make_z(n, seed=0)
    out_par = np.empty(n, dtype=np.float32)
    out_ser = np.empty(n, dtype=np.float32)
    _dyadic_haar_leg_njit(z, _LEFT, _MID, _RIGHT, out_par)
    _dyadic_haar_leg_njit_serial(z, _LEFT, _MID, _RIGHT, out_ser)

    t0 = time.perf_counter()
    for _ in range(n_calls):
        _dyadic_haar_leg_njit(z, _LEFT, _MID, _RIGHT, out_par)
    t_par = (time.perf_counter() - t0) / n_calls

    t0 = time.perf_counter()
    for _ in range(n_calls):
        _dyadic_haar_leg_njit_serial(z, _LEFT, _MID, _RIGHT, out_ser)
    t_ser = (time.perf_counter() - t0) / n_calls

    max_diff = float(np.max(np.abs(out_par.astype(np.float64) - out_ser.astype(np.float64))))
    print(f"n={n:>11}: parallel={t_par * 1000:9.4f}ms serial={t_ser * 1000:9.4f}ms ratio={t_par / t_ser:8.2f}x max_diff={max_diff:.3e}")


def _check_bit_identity() -> None:
    """Serial vs parallel must agree exactly -- same per-element test, independent of thread."""
    max_seen = 0.0
    for n in (0, 1, 2, 50, 500, 5000, 50_000):
        for j, k in ((0, 0), (1, 0), (1, 1), (3, 5), (5, 17)):
            for seed in range(3):
                z = _make_z(n, seed)
                out_par = np.empty(n, dtype=np.float32)
                out_ser = np.empty(n, dtype=np.float32)
                width = 1.0 / (2**j)
                left = k * width
                mid = left + width / 2.0
                right = left + width
                _dyadic_haar_leg_njit(z, left, mid, right, out_par)
                _dyadic_haar_leg_njit_serial(z, left, mid, right, out_ser)
                d = float(np.max(np.abs(out_par - out_ser))) if n else 0.0
                max_seen = max(max_seen, d)
                assert d == 0.0, f"n={n} j={j} k={k} seed={seed}: diff={d}"
    print(f"bit-identity sweep OK (7x5x3 = 105 scenarios), max diff = {max_seen}")

    # Dispatch threshold sanity + full _dyadic_haar_leg wrapper equivalence.
    z_small = _make_z(1000, seed=0)
    leg_wrapper = _dyadic_haar_leg(z_small, 2, 1)
    out_ref = np.empty(1000, dtype=np.float32)
    width = 1.0 / 4
    left = 1 * width
    mid = left + width / 2.0
    right = left + width
    _dyadic_haar_leg_njit_serial(z_small, left, mid, right, out_ref)
    assert np.array_equal(leg_wrapper, out_ref), "wrapper (below threshold) should route to the serial kernel"
    print(f"wrapper dispatch OK (fallback threshold={20_000_000:_})")
    assert _haar_leg_fallback_choice(19_999_999) == "serial"
    assert _haar_leg_fallback_choice(20_000_000) == "parallel"


if __name__ == "__main__":
    _check_bit_identity()
    for n in (600, 1_500, 15_000, 100_000, 1_000_000, 5_000_000, 20_000_000, 50_000_000):
        _bench(n)
