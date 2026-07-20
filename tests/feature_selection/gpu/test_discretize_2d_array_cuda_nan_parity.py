"""Regression test for mrmr_audit_2026-07-20 B-12: ``discretize_2d_array_cuda`` did not filter NaN before
computing per-column min/max (``uniform``) or percentiles (``quantile``), unlike the CPU
``discretize_2d_array`` path. A single NaN anywhere in a column poisoned that column's derived bin
edges to NaN, and ``searchsorted``/the affine map against NaN edges silently collapsed the WHOLE column's
REAL values (not just the NaN rows) into a degenerate bucket -- a genuine algorithmic divergence from the
CPU path, not float-reduction-order noise.

Fixed via ``np.nanpercentile`` (host-side, no cupy nanpercentile exists) for the quantile branch and
``cp.nanmin``/``cp.nanmax`` for the uniform branch, plus routing individual NaN VALUES to the same
dedicated NaN bin code (``n_bins``) the CPU ``discretize_uniform`` kernel uses, instead of casting NaN to
an undefined garbage int code.

GPU-only; auto-skips when cupy/CUDA is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if not cp.cuda.is_available():
    pytest.skip("CUDA not available on this host", allow_module_level=True)


def _nan_scattered_fixture(seed: int = 0, n: int = 2000):
    """A clean column and a column with NaN scattered through it (real signal still present)."""
    rng = np.random.default_rng(seed)
    col_clean = rng.normal(size=n)
    col_nan = rng.normal(size=n)
    col_nan[::7] = np.nan
    return np.column_stack([col_clean, col_nan]).astype(np.float64)


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["quantile", "uniform"])
def test_gpu_matches_cpu_on_nan_bearing_column(method):
    """The GPU and CPU discretized codes must be bit-identical, including on the NaN-bearing column --
    pre-fix, the GPU column's REAL (non-NaN) values collapsed to a single degenerate bucket."""
    from mlframe.feature_selection.filters.discretization import discretize_2d_array, discretize_2d_array_cuda

    arr = _nan_scattered_fixture()
    cpu_out = discretize_2d_array(arr, n_bins=5, method=method, dtype=np.int8)
    gpu_out = discretize_2d_array_cuda(arr, n_bins=5, method=method, dtype=np.int8)

    assert np.array_equal(cpu_out[:, 0], gpu_out[:, 0]), f"{method}: clean column diverged between CPU and GPU"
    assert len(np.unique(gpu_out[:, 1])) > 1, (
        f"{method}: NaN-bearing column's GPU output collapsed to a single bucket " f"(pre-fix behaviour) -- got {np.unique(gpu_out[:, 1])}"
    )
    assert np.array_equal(cpu_out[:, 1], gpu_out[:, 1]), (
        f"{method}: NaN-bearing column diverged between CPU ({np.unique(cpu_out[:, 1])}) " f"and GPU ({np.unique(gpu_out[:, 1])})"
    )


@pytest.mark.gpu
def test_gpu_uniform_nan_rows_get_dedicated_code_not_garbage():
    """The 'uniform' GPU branch must route individual NaN VALUES to the dedicated NaN bin code
    (n_bins, one past the real range) matching the CPU discretize_uniform kernel's convention --
    pre-fix, an individual NaN value cast to an undefined garbage int8 code."""
    from mlframe.feature_selection.filters.discretization import discretize_2d_array_cuda

    arr = _nan_scattered_fixture()
    n_bins = 5
    gpu_out = discretize_2d_array_cuda(arr, n_bins=n_bins, method="uniform", dtype=np.int8)
    nan_rows = np.isnan(arr[:, 1])
    assert nan_rows.any(), "fixture invariant: column 1 must contain NaN"
    assert np.all(gpu_out[nan_rows, 1] == n_bins), f"NaN rows must all carry the dedicated code {n_bins}, got {np.unique(gpu_out[nan_rows, 1])}"
    assert np.all(gpu_out[~nan_rows, 1] < n_bins), "non-NaN rows must never carry the dedicated NaN code"
