"""Bench + bit-identity check: ``_apply_minmax_njit`` (parallel-over-rows) vs its serial twin
``_apply_minmax_njit_serial``, and the ``_MINMAX_PARALLEL_MIN_N`` threshold in ``_apply_minmax``.

Finding (2026-08-23): the njit-fused minmax replay kernel (``_apply_minmax_njit``, added 2026-08-04 as
a genuine win over the original 3-4-pass numpy form, 6.58x measured then) was ``@njit(parallel=True)``
with ``prange`` over ROWS. A 2-flop-plus-optional-clip elementwise op is too trivial per-element to
amortize numba's fixed per-call thread-pool dispatch cost at any n this codebase's FE search
realistically reaches. Measured here (same-process A/B, warm, best-of-50):

    n=       600: parallel/serial  ~22x SLOWER
    n=     1_500: parallel/serial  ~28x SLOWER
    n=    15_000: parallel/serial 3.2x SLOWER
    n=   100_000: parallel/serial 2.9x SLOWER
    n= 1_000_000: parallel/serial 9.1x SLOWER
    n= 5_000_000: parallel/serial 2.4x SLOWER
    n=20_000_000: parallel/serial 0.64x (1.56x FASTER)
    n=50_000_000: parallel/serial 0.60x (1.67x FASTER)

``_MINMAX_PARALLEL_MIN_N = 10_000_000`` sits between the measured-loss (n=5M) and measured-win (n=20M)
points, biased toward the serial side: an unnecessary serial call at n~5-20M costs at most tens of ms
while an unnecessary parallel call below it costs up to 28x per the ratios above.

Run: ``python profiling/bench_apply_minmax_parallel_vs_serial.py``
"""

from __future__ import annotations

import time

import numpy as np

from mlframe.feature_selection.filters.hermite_fe import (
    _apply_minmax,
    _apply_minmax_njit,
    _apply_minmax_njit_serial,
    _MINMAX_PARALLEL_MIN_N,
)


def _make_x(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.random(n) * 10.0, dtype=np.float64)


def _bench(n: int, n_calls: int = 50) -> None:
    x = _make_x(n, seed=0)
    lo, span, has_clip, clip_val = 0.0, 10.0, True, 1.0
    _apply_minmax_njit(x, lo, span, has_clip, clip_val)
    _apply_minmax_njit_serial(x, lo, span, has_clip, clip_val)

    t0 = time.perf_counter()
    for _ in range(n_calls):
        out_par = _apply_minmax_njit(x, lo, span, has_clip, clip_val)
    t_par = (time.perf_counter() - t0) / n_calls

    t0 = time.perf_counter()
    for _ in range(n_calls):
        out_ser = _apply_minmax_njit_serial(x, lo, span, has_clip, clip_val)
    t_ser = (time.perf_counter() - t0) / n_calls

    max_diff = float(np.max(np.abs(out_par - out_ser)))
    print(f"n={n:>11}: parallel={t_par * 1000:9.4f}ms serial={t_ser * 1000:9.4f}ms ratio={t_par / t_ser:8.2f}x max_diff={max_diff:.3e}")


def _check_bit_identity() -> None:
    """Serial vs parallel must agree exactly -- same per-element arithmetic, independent of thread."""
    max_seen = 0.0
    for n in (0, 1, 2, 50, 500, 5000):
        for lo, hi, clip in ((0.0, 10.0, None), (-5.0, 5.0, 1.0), (0.0, 1.0, 0.5), (-100.0, -1.0, None)):
            for seed in range(3):
                x = _make_x(n, seed) if n else np.empty(0, dtype=np.float64)
                span = hi - lo + 1e-12
                has_clip = clip is not None
                clip_val = float(clip) if clip is not None else 0.0
                out_par = _apply_minmax_njit(x, lo, span, has_clip, clip_val)
                out_ser = _apply_minmax_njit_serial(x, lo, span, has_clip, clip_val)
                d = float(np.max(np.abs(out_par - out_ser))) if n else 0.0
                max_seen = max(max_seen, d)
                assert d == 0.0, f"n={n} lo={lo} hi={hi} clip={clip} seed={seed}: diff={d}"
    print(f"bit-identity sweep OK (6x4x3 = 72 scenarios), max diff = {max_seen}")

    # Dispatch threshold sanity via the real wrapper.
    x_small = _make_x(1000, seed=0)
    params = {"lo": 0.0, "hi": 10.0, "clip": 1.0}
    out_wrapper = _apply_minmax(x_small, params)
    span = params["hi"] - params["lo"] + 1e-12
    out_ref = _apply_minmax_njit_serial(np.ascontiguousarray(x_small, dtype=np.float64), params["lo"], span, True, 1.0)
    assert np.array_equal(out_wrapper, out_ref), "wrapper (below threshold) should route to the serial kernel"
    print(f"wrapper dispatch OK (threshold={_MINMAX_PARALLEL_MIN_N:_})")


if __name__ == "__main__":
    _check_bit_identity()
    for n in (600, 1_500, 15_000, 100_000, 1_000_000, 5_000_000, 20_000_000, 50_000_000):
        _bench(n)
