"""Bench + bit-identity check: ``parallel_mi_prange`` (parallel-over-permutations) vs its serial twin
``_parallel_mi_prange_serial``, and the ``_MI_PRANGE_PARALLEL_MIN_WORK`` threshold in ``_mi_prange_dispatch``.

Finding (2026-08-23): ``parallel_mi_prange``'s existing bench (``bench_permutation_njit_prange_vs_joblib.py``)
only validated parallel-vs-serial at npermutations=500 (large); the actual DEFAULT npermutations=10 was
never checked. Unlike this session's other three njit(parallel=True)-over-a-small-dimension fixes (ICE
kernel, Haar leg, minmax replay), EACH permutation iteration here does real O(n) work (a Fisher-Yates
shuffle + one MI evaluation), so the crossover genuinely depends on BOTH n and npermutations together
(their product, "total work"), not a single dimension in isolation. Measured here (same-process A/B,
warm, best-of-30):

    n=    600 nperm= 10 (work=    6,000): parallel/serial 8.60x SLOWER
    n=    600 nperm= 25 (work=   15,000): parallel/serial 1.40x SLOWER
    n=   1500 nperm= 10 (work=   15,000): parallel/serial 2.00x SLOWER
    n=    600 nperm=100 (work=   60,000): parallel/serial 0.74x (1.35x FASTER)
    n=  15000 nperm= 10 (work=  150,000): parallel/serial 0.43x (2.3x FASTER)
    n=  15000 nperm=100 (work=1,500,000): parallel/serial 0.33x (3.0x FASTER)
    n= 100000 nperm= 10 (work=1,000,000): parallel/serial 0.49x (2.0x FASTER)
    n= 100000 nperm=500 (work=50,000,000): parallel/serial 0.25x (4.0x FASTER)

Dispatch is via the per-host kernel_tuning_cache (``_MI_PRANGE_PARALLELISM_SPEC``, no hardcoded
threshold) -- ``_mi_prange_fallback_choice`` (used pre-sweep / on tuner failure) routes at
n*npermutations=30_000, in the dead zone between the confirmed-loss region (work<=15,000) and the
confirmed-win region (work>=60,000), biased toward the serial side per this session's established
asymmetric-risk reasoning.

Run: ``python profiling/bench_mi_prange_parallel_vs_serial.py``
"""

from __future__ import annotations

import time

import numpy as np

from mlframe.feature_selection.filters.permutation import (
    parallel_mi_prange,
    _parallel_mi_prange_serial,
    _mi_prange_dispatch,
    _mi_prange_fallback_choice,
)

_NBINS = 10


def _make_inputs(n: int, seed: int):
    rng = np.random.default_rng(seed)
    classes_x = rng.integers(0, _NBINS, n).astype(np.int32)
    freqs_x = np.bincount(classes_x, minlength=_NBINS).astype(np.int32)
    classes_y = rng.integers(0, 2, n).astype(np.int32)
    freqs_y = np.bincount(classes_y, minlength=2).astype(np.int32)
    return classes_x, freqs_x, classes_y, freqs_y


def _bench(n: int, npermutations: int, n_calls: int = 30) -> None:
    classes_x, freqs_x, classes_y, freqs_y = _make_inputs(n, seed=0)
    original_mi, base_seed = 0.01, np.uint64(42)
    parallel_mi_prange(classes_x, freqs_x, classes_y, freqs_y, npermutations, original_mi, base_seed, np.int32, False)
    _parallel_mi_prange_serial(classes_x, freqs_x, classes_y, freqs_y, npermutations, original_mi, base_seed, np.int32, False)

    t0 = time.perf_counter()
    for _ in range(n_calls):
        out_par = parallel_mi_prange(classes_x, freqs_x, classes_y, freqs_y, npermutations, original_mi, base_seed, np.int32, False)
    t_par = (time.perf_counter() - t0) / n_calls

    t0 = time.perf_counter()
    for _ in range(n_calls):
        out_ser = _parallel_mi_prange_serial(classes_x, freqs_x, classes_y, freqs_y, npermutations, original_mi, base_seed, np.int32, False)
    t_ser = (time.perf_counter() - t0) / n_calls

    work = n * npermutations
    match = "OK" if out_par == out_ser else "MISMATCH"
    print(f"n={n:>7} nperm={npermutations:>4} work={work:>10}: parallel={t_par * 1000:9.4f}ms serial={t_ser * 1000:9.4f}ms ratio={t_par / t_ser:6.2f}x {match}")


def _check_bit_identity() -> None:
    """Serial vs parallel must agree exactly -- each permutation iteration owns a private classes_y
    copy and an LCG state seeded purely from (base_seed, i), independent of thread/iteration order."""
    for n in (0, 1, 2, 50, 500, 2000):
        for npermutations in (0, 1, 5, 20):
            if n == 0 and npermutations > 0:
                continue  # pre-existing degenerate case in the original kernel (n=0 divides by zero
                # inside compute_relevance_score), unrelated to this fix -- both variants share it.
            for seed in range(3):
                classes_x, freqs_x, classes_y, freqs_y = _make_inputs(n, seed) if n else (
                    np.zeros(0, np.int32), np.zeros(_NBINS, np.int32), np.zeros(0, np.int32), np.zeros(2, np.int32)
                )
                base_seed = np.uint64(seed * 97 + 3)
                out_par = parallel_mi_prange(classes_x, freqs_x, classes_y, freqs_y, npermutations, 0.01, base_seed, np.int32, False)
                out_ser = _parallel_mi_prange_serial(classes_x, freqs_x, classes_y, freqs_y, npermutations, 0.01, base_seed, np.int32, False)
                assert out_par == out_ser, f"n={n} npermutations={npermutations} seed={seed}: parallel={out_par} serial={out_ser}"
    print("bit-identity sweep OK (6x4x3 = 72 scenarios)")

    # Dispatch threshold sanity via the real dispatcher.
    classes_x, freqs_x, classes_y, freqs_y = _make_inputs(100, seed=0)
    out_dispatch_below = _mi_prange_dispatch(classes_x, freqs_x, classes_y, freqs_y, 5, 0.01, np.uint64(1))
    out_serial_below = _parallel_mi_prange_serial(classes_x, freqs_x, classes_y, freqs_y, 5, 0.01, np.uint64(1), np.int32, False)
    assert out_dispatch_below == out_serial_below, "dispatch (below threshold) should route to the serial kernel"
    print(f"wrapper dispatch OK (fallback threshold={30_000:_})")
    assert _mi_prange_fallback_choice(1000, 29) == "serial"
    assert _mi_prange_fallback_choice(1000, 30) == "parallel"


if __name__ == "__main__":
    _check_bit_identity()
    for n, nperm in [(600, 10), (600, 25), (1500, 10), (15000, 10), (100000, 10), (600, 100), (15000, 100), (100000, 500)]:
        _bench(n, nperm)
