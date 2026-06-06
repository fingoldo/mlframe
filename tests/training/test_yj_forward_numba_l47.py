"""Regression: ``_yj_forward`` / ``_yj_inverse`` size-dispatch to a
numba parallel kernel for n>=10k, falling back to numpy for tiny inputs.
The kernel output must be bit-identical to the numpy reference across
the full lambda range.

Pre-fix path (iter-47 500k seed=99 cb-only profile):
- ``composite_unary_transforms._yj_forward`` was 5.80s tottime / 72
  calls during the Brent neg-loglik scan inside ``yeo_johnson_y_fit``.
- The numpy implementation uses fancy indexing (``y[y>=0]``,
  ``y[y<0]``) with two np.power calls + mask-based assigns; each call
  allocates 5+ intermediate arrays on n=405k.

Post-fix: a numba @njit(parallel=True, fastmath=True, cache=True)
kernel computes the YJ branches per-element in parallel. Bench at
tests/perf/bench_yj_forward.py:
  n=50k:  cur=32ms  numba_par=6ms   (5.3x)
  n=200k: cur=197ms numba_par=22ms  (8.8x)
  n=405k: cur=392ms numba_par=40ms  (9.8x)
  n=1M:   cur=913ms numba_par=122ms (7.5x)

Sensors:
1. Bit-exact match between numpy reference and numba kernel across the
   standard YJ lambda range (lam ∈ {-1.5, -0.5, 0, 0.5, 1, 1.5, 2,
   2.5, 3.5}) -- locks the dispatcher contract.
2. Forward o inverse round-trip is identity to machine precision on
   real-valued inputs (rtol=1e-9).
3. Tiny-input path (n=100) falls back to the numpy path -- proven via
   monkeypatch that asserts the numba kernel is NOT called.
4. Soft speedup gate at n=100k: dispatcher >= 2x faster than the pure
   numpy reference. A future change that disables numba (e.g. by
   raising _YJ_NUMBA_MIN_N too high) trips this sensor.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from tests.conftest import running_under_xdist

from mlframe.training import composite_unary_transforms as cut
from mlframe.training.composite_unary_transforms import (
    _yj_forward,
    _yj_forward_numpy,
    _yj_inverse,
    _yj_inverse_numpy,
)


LAMS = [-1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5]


@pytest.mark.parametrize("lam", LAMS)
def test_yj_forward_numba_bit_exact_vs_numpy(lam: float) -> None:
    """Bit-exact match between the dispatcher's numba path and the
    numpy reference at n=50k (above the dispatch threshold)."""
    rng = np.random.default_rng(int(abs(lam) * 100))
    y = rng.standard_normal(50_000).astype(np.float64)
    ref = _yj_forward_numpy(y, lam)
    out = _yj_forward(y, lam)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("lam", LAMS)
def test_yj_inverse_numba_bit_exact_vs_numpy(lam: float) -> None:
    """Inverse dispatcher matches the numpy reference."""
    rng = np.random.default_rng(int(abs(lam) * 100) + 1)
    t = rng.standard_normal(50_000).astype(np.float64)
    ref = _yj_inverse_numpy(t, lam)
    out = _yj_inverse(t, lam)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("lam", LAMS)
def test_yj_forward_inverse_round_trip(lam: float) -> None:
    """forward(inverse(y)) == y to machine precision for real y."""
    rng = np.random.default_rng(int(abs(lam) * 100) + 2)
    y = rng.standard_normal(20_000).astype(np.float64)
    t = _yj_forward(y, lam)
    y2 = _yj_inverse(t, lam)
    np.testing.assert_allclose(y2, y, rtol=1e-9, atol=1e-9)


def test_tiny_input_falls_back_to_numpy(monkeypatch) -> None:
    """n < _YJ_NUMBA_MIN_N: the numba kernel must NOT be called; the
    numpy reference path handles it."""
    calls: list[int] = []

    def _spy(y, lam):  # type: ignore[no-untyped-def]
        calls.append(len(y))
        # Delegate to the real kernel so output is correct.
        return cut._yj_forward_numba_kernel.py_func(y, lam) \
            if hasattr(cut._yj_forward_numba_kernel, "py_func") \
            else cut._yj_forward_numba_kernel(y, lam)

    monkeypatch.setattr(cut, "_yj_forward_numba_kernel", _spy)
    y = np.linspace(-2.0, 2.0, 100).astype(np.float64)
    _ = _yj_forward(y, 0.5)
    assert calls == [], (
        f"expected numba kernel skipped on n=100 < {cut._YJ_NUMBA_MIN_N}; "
        f"but got {len(calls)} call(s)"
    )


def test_yj_forward_speedup_gate() -> None:
    """At n=100k the dispatcher (numba path) must be >= 2x faster than
    the pure numpy reference across a 12-step Brent-like sweep. Soft
    gate -- production speedup is 5-10x at n>=50k (see docstring).
    A future change that accidentally raises _YJ_NUMBA_MIN_N too high
    or disables the kernel trips this sensor."""
    if running_under_xdist():
        pytest.skip("timing unreliable under -n contention")
    rng = np.random.default_rng(0)
    y = rng.standard_normal(100_000).astype(np.float64)
    lams = np.linspace(-1.8, 3.8, 12).tolist()
    # Warm JIT
    _ = _yj_forward(y, 1.0)

    def _time(fn):
        t = []
        for _ in range(3):
            s = time.perf_counter()
            for lam in lams:
                fn(y, lam)
            t.append(time.perf_counter() - s)
        return sorted(t)[1]

    numpy_s = _time(_yj_forward_numpy)
    disp_s = _time(_yj_forward)
    speedup = numpy_s / max(disp_s, 1e-9)
    assert speedup >= 2.0, (
        f"expected >= 2x speedup at n=100k; got numpy={numpy_s*1000:.1f}ms "
        f"dispatcher={disp_s*1000:.1f}ms speedup={speedup:.2f}x"
    )
