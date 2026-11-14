"""The single-kernel orthogonal-polynomial design is bit-identical to the cupy recurrence it replaces."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters.hermite_fe import _hermite_prewarp_gpu_resident as pw
from mlframe.feature_selection.filters.hermite_fe._basis_fused_gpu import MAX_COLUMNS, build_basis_fused


def _both(monkeypatch, basis, z, degree):
    """The design built with the loop and with the kernel."""
    monkeypatch.setenv("MLFRAME_FE_BASIS_FUSED", "0")
    loop = cp.asnumpy(pw._build_basis_matrix_gpu(cp, basis, z, degree))
    monkeypatch.setenv("MLFRAME_FE_BASIS_FUSED", "1")
    return loop, cp.asnumpy(pw._build_basis_matrix_gpu(cp, basis, z, degree))


@pytest.mark.parametrize("basis", ["hermite", "legendre", "chebyshev", "laguerre"])
@pytest.mark.parametrize("degree", [1, 2, 3, 6, 9])
def test_bit_identical_to_the_loop(monkeypatch, basis, degree):
    """Every entry equal, not merely close - the kernel rounds where cupy's separate elementwise kernels do."""
    rng = np.random.default_rng(degree)
    z = cp.asarray(rng.normal(size=20_001))
    loop, fused = _both(monkeypatch, basis, z, degree)
    assert loop.shape == fused.shape == (20_001, degree + 1)
    np.testing.assert_array_equal(fused, loop)


def test_float32_input_is_widened_like_the_loop(monkeypatch):
    """The recurrence runs in float64 whatever the input dtype."""
    z = cp.asarray(np.random.default_rng(1).normal(size=5_000).astype(np.float32))
    loop, fused = _both(monkeypatch, "hermite", z, 5)
    np.testing.assert_array_equal(fused, loop)


def test_requests_outside_the_kernel_fall_back():
    """An unknown basis or a design wider than the unrolled budget returns None so the caller keeps the loop."""
    x = cp.asarray(np.linspace(-1, 1, 100))
    assert build_basis_fused(cp, "wavelet", x, 4) is None
    assert build_basis_fused(cp, "hermite", x, MAX_COLUMNS + 1) is None
    assert build_basis_fused(cp, "hermite", x, 0) is None


def test_extreme_values_overflow_the_same_way(monkeypatch):
    """Large |z| overflows to inf/nan in both implementations at the same entries."""
    z = cp.asarray(np.array([0.0, 1e3, -1e3, 1e100, np.nan, np.inf]))
    loop, fused = _both(monkeypatch, "laguerre", z, 8)
    np.testing.assert_array_equal(np.isnan(fused), np.isnan(loop))
    np.testing.assert_array_equal(fused[np.isfinite(loop)], loop[np.isfinite(loop)])


@pytest.mark.parametrize("n", [1, 31, 33, 257, 4099])
@pytest.mark.parametrize("n_columns", [1, 2, MAX_COLUMNS])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ragged_row_counts_and_the_widest_design_match_the_loop(monkeypatch, n, n_columns, dtype):
    """The kernel writes each block's rows as one contiguous tile: row counts that leave a partial last tile, one column and the widest design (the tile halves the block) stay bit-identical."""
    z = cp.asarray(np.random.default_rng(n + n_columns).normal(size=n).astype(dtype))
    monkeypatch.setenv("MLFRAME_FE_BASIS_FUSED", "0")
    loop = cp.asnumpy(pw._build_basis_matrix_gpu(cp, "legendre", z, n_columns - 1))
    from mlframe.feature_selection.filters._fe_gpu_batch._devices import crit_float_dtype

    got = cp.asnumpy(build_basis_fused(cp, "legendre", cp.ascontiguousarray(z, dtype=crit_float_dtype()), n_columns))  # the loop widens to the critical dtype first
    assert got.shape == loop.shape == (n, n_columns)
    np.testing.assert_array_equal(got, loop)


def test_a_double_design_with_a_wide_tile_matches_the_numpy_recurrence():
    """Double input (8 bytes per tile value, so the widest design halves the block): equal to the Chebyshev recurrence run in numpy."""
    x = np.random.default_rng(5).normal(size=1_003)
    nc = MAX_COLUMNS
    want = np.empty((x.size, nc))
    want[:, 0], want[:, 1] = 1.0, x
    for k in range(2, nc):
        want[:, k] = 2.0 * x * want[:, k - 1] - want[:, k - 2]
    got = cp.asnumpy(build_basis_fused(cp, "chebyshev", cp.asarray(x), nc))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_a_tuner_entry_exists_and_the_fused_kernel_is_the_fallback():
    """The single kernel won at every measured size, so the pre-sweep fallback is the kernel; the entry exists for hosts where the sweep says otherwise."""
    from mlframe.feature_selection.filters.hermite_fe import _basis_fused_gpu as F

    assert F._SPEC is not None
    assert F._fallback_choice(5_000) == "fused" and F._fallback_choice(5_000_000) == "fused"


def test_the_builder_follows_the_tuner_verdict(monkeypatch):
    """A 'loop' verdict builds the design with the cupy loop (never touching the kernel); the env switch ``MLFRAME_FE_BASIS_FUSED=0`` still forces the loop."""
    from mlframe.feature_selection.filters.hermite_fe import _basis_fused_gpu as F

    z = cp.asarray(np.random.default_rng(1).normal(size=3_000))
    monkeypatch.setenv("MLFRAME_FE_BASIS_FUSED", "1")
    called = []
    real = F.build_basis_fused
    monkeypatch.setattr(F, "build_basis_fused", lambda *a, **k: called.append(1) or real(*a, **k))
    monkeypatch.setattr(F, "basis_fused_chosen", lambda n: False)
    loop = pw._build_basis_matrix_gpu(cp, "hermite", z, 4)
    assert not called
    monkeypatch.setattr(F, "basis_fused_chosen", lambda n: True)
    fused = pw._build_basis_matrix_gpu(cp, "hermite", z, 4)
    assert called
    np.testing.assert_array_equal(cp.asnumpy(fused), cp.asnumpy(loop))


def test_the_sweep_callables_agree_bit_for_bit():
    """The sweep's two builds (loop and kernel, both bases) are equal on a resident column."""
    from mlframe.feature_selection.filters.hermite_fe import _basis_fused_gpu as F

    (x,) = F._sweep_inputs({"n_samples": 9_001})
    np.testing.assert_array_equal(F._sweep_loop(x), F._sweep_fused(x))


def test_a_refused_vram_cushion_returns_none_without_allocating(monkeypatch):
    """The design is the minimal operand (nothing to stream), so a refused cushion declines before the allocation and the caller's loop decides."""
    import mlframe.feature_selection.filters._fe_gpu_vram as V

    monkeypatch.setattr(V, "fe_gpu_has_vram_cushion", lambda nbytes: False)
    assert build_basis_fused(cp, "hermite", cp.zeros(100, dtype=cp.float32), 4) is None
