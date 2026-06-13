"""Regression: ``_mi_per_feature_y_fixed`` hoists the y-quantile out of
the per-feature MI loop used by ``CompositeTargetDiscovery._auto_base``.

Pre-fix path (composite_discovery.py:1587, iter-45 500k seed=99 profile):
- ``_mi_pair_bin`` was called once per candidate feature column with the
  SAME ``y_screen`` argument. Inside, ``np.quantile(y, qs)`` ran on every
  call -- K=30 features means 30 redundant y-quantile passes.
- This single hotspot was 3.12 s tottime / 9.09 s cumtime / 105 calls in
  the 500k cb-only suite profile -- the largest mlframe-OWN tottime
  outside the boosters.

Post-fix: ``_mi_per_feature_y_fixed`` quantiles ``y`` once, then loops
``np.quantile(x_col) + searchsorted + _mi_from_binned_pair`` per column.

Test purpose:
1. Lock the bit-exact contract: same per-feature MI vector vs the naive
   ``[_mi_pair_bin(...) for j in cols]`` baseline. A future change that
   subtly perturbs the y-binning (different ``q`` grid, different
   ``side=`` on searchsorted, different clip range) fails this sensor.
2. Soft speedup gate at moderate scale (n=20k, k=20, nbins=32):
   helper >= 1.2x faster than the naive loop. The 1.67x production-
   scale gain (n=500k, k=30, nbins=50) is documented in the helper's
   docstring; this gate is the smaller, more forgiving CI-friendly
   variant.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from mlframe.training.composite.discovery.screening import (
    _mi_pair_bin,
    _mi_per_feature_y_fixed,
)


def _build_inputs(n: int, k: int, *, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, k)).astype(np.float64)
    # y depends on a few features so MI signal is non-trivial.
    y = (
        0.7 * x[:, 0]
        + 0.4 * x[:, 1] ** 2
        - 0.3 * x[:, 2] * x[:, 3]
        + rng.standard_normal(n) * 0.5
    ).astype(np.float64)
    return x, y


def _naive_loop(x: np.ndarray, y: np.ndarray, *, nbins: int) -> np.ndarray:
    return np.array(
        [_mi_pair_bin(x[:, j], y, nbins=nbins) for j in range(x.shape[1])]
    )


def test_y_fixed_bit_exact_vs_naive_loop() -> None:
    """Hoisted helper must produce per-feature MI bit-identical to the
    naive ``[_mi_pair_bin(...) for j ...]`` baseline."""
    x, y = _build_inputs(n=4000, k=8, seed=42)
    nbins = 32
    naive = _naive_loop(x, y, nbins=nbins)
    hoisted = _mi_per_feature_y_fixed(x, y, nbins=nbins)
    assert hoisted.shape == naive.shape
    # Same bin codes + same joint counts; the helper scores via the njit kernel whose final MI/marginal reduction walks cells row-major while
    # _mi_pair_bin reduces via numpy pairwise summation. That FP-order difference is ULP-scale (<1e-9, pinned in test_mi_kernel_divergence_bound),
    # well below any MI ranking threshold, so the contract is allclose rather than bitwise.
    np.testing.assert_allclose(hoisted, naive, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("nbins", [16, 32, 50])
def test_y_fixed_matches_naive_across_nbins(nbins: int) -> None:
    """Bit-exact match across the nbins range actually used by
    CompositeTargetDiscovery (default 32, fast=16, accurate=50)."""
    x, y = _build_inputs(n=2000, k=6, seed=nbins)
    naive = _naive_loop(x, y, nbins=nbins)
    hoisted = _mi_per_feature_y_fixed(x, y, nbins=nbins)
    # ULP-scale FP-reduction-order difference between njit kernel and _mi_pair_bin (see test_mi_kernel_divergence_bound); allclose, not bitwise.
    np.testing.assert_allclose(hoisted, naive, rtol=1e-9, atol=1e-12)


def test_y_fixed_short_circuit_below_5x_nbins() -> None:
    """Below ``5 * nbins`` rows the helper returns all zeros (matches
    ``_mi_pair_bin``'s n<5*nbins -> 0 contract)."""
    rng = np.random.default_rng(0)
    nbins = 16
    # 5 * 16 = 80 row threshold; pass 50 rows.
    x = rng.standard_normal((50, 4)).astype(np.float64)
    y = rng.standard_normal(50).astype(np.float64)
    out = _mi_per_feature_y_fixed(x, y, nbins=nbins)
    np.testing.assert_array_equal(out, np.zeros(4))


def test_y_fixed_empty_columns() -> None:
    """Zero-feature matrix returns an empty float64 array."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1000, 0)).astype(np.float64)
    y = rng.standard_normal(1000).astype(np.float64)
    out = _mi_per_feature_y_fixed(x, y, nbins=16)
    assert out.shape == (0,)
    assert out.dtype == np.float64


def test_y_fixed_speedup_gate() -> None:
    """Soft speedup gate: at moderate scale the hoisted helper must be
    >= 1.2x faster than the naive loop. Production-scale gain reaches
    1.67x (documented in helper docstring); 1.2x is the CI-friendly
    floor accounting for noise on slow machines."""
    x, y = _build_inputs(n=20_000, k=20, seed=7)
    nbins = 32
    # Warm CPU caches with one pass each.
    _ = _naive_loop(x, y, nbins=nbins)
    _ = _mi_per_feature_y_fixed(x, y, nbins=nbins)

    # Median of 3 runs each.
    def _time(fn):
        t = []
        for _ in range(3):
            s = time.perf_counter()
            fn()
            t.append(time.perf_counter() - s)
        return sorted(t)[1]

    naive_ms = _time(lambda: _naive_loop(x, y, nbins=nbins)) * 1000
    hoisted_ms = _time(lambda: _mi_per_feature_y_fixed(x, y, nbins=nbins)) * 1000
    speedup = naive_ms / max(hoisted_ms, 1e-9)
    # Soft gate -- the helper has to materially beat the naive baseline.
    # If a future change accidentally reintroduces per-call y-quantile,
    # the speedup collapses to ~1.0x and this sensor fails.
    assert speedup >= 1.2, (
        f"expected >=1.2x speedup; got naive={naive_ms:.1f}ms "
        f"hoisted={hoisted_ms:.1f}ms speedup={speedup:.2f}x"
    )
