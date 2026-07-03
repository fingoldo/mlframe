"""Numeric-bound pin (audit3 cross-backend #6 / FUTURE-GPU): the GPU-resident discretiser bins CANDIDATE
values born in float32 against float64-exact percentile edges, so a value within ~1e-6 of a bin edge can land
one bin off the CPU float64 codes. This was only SELECTION-tested (test_discretize_cuda_cpu_parity); here we
pin the NUMERIC bound directly -- the f32-vs-f64 divergence is at most +/-1 bin, on a small bounded fraction
of rows -- so a future kernel change that widened it (a selection-altering multi-bin shift) trips loudly.

The CPU test isolates the exact hazard (f64 edges, f32 vs f64 candidate values) deterministically; a
GPU-gated test asserts the same bound on the real device kernel when CUDA is present.
"""
import numpy as np
import pytest

from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _quantile_bin


@pytest.mark.parametrize("nbins", [10, 32, 64])
def test_f32_candidate_against_f64_edges_shifts_at_most_one_bin(nbins):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50_000).astype(np.float64)
    # float64-exact equi-frequency edges (what the GPU-resident path computes).
    edges = np.quantile(x, np.linspace(0.0, 1.0, nbins + 1)[1:-1])
    code_f64 = np.searchsorted(edges, x, side="right")
    code_f32 = np.searchsorted(edges, x.astype(np.float32).astype(np.float64), side="right")
    diff = np.abs(code_f64.astype(np.int64) - code_f32.astype(np.int64))
    assert diff.max() <= 1, f"f32 candidate shifted a bin code by {diff.max()} bins (nbins={nbins}); >1 can flip selection"
    frac_moved = float(np.mean(diff > 0))
    assert frac_moved < 0.02, f"{frac_moved:.4%} of rows moved a bin under f32 (nbins={nbins}); should be the ~edge sliver only"


@pytest.mark.parametrize("nbins", [16, 48])
def test_real_quantile_bin_f32_input_bounded(nbins):
    """The real CPU _quantile_bin on f32-cast vs f64 input: codes differ by at most 1 bin (full binner f32
    sensitivity, edges + values), matching the GPU binner's documented <~1e-5-of-rows, <=1-bin bound."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(40_000).astype(np.float64)
    c64 = np.asarray(_quantile_bin(x, nbins)).astype(np.int64)
    c32 = np.asarray(_quantile_bin(x.astype(np.float32).astype(np.float64), nbins)).astype(np.int64)
    diff = np.abs(c64 - c32)
    assert diff.max() <= 1, f"real _quantile_bin f32 shift {diff.max()} > 1 bin"
    assert float(np.mean(diff > 0)) < 0.02


def test_gpu_resident_discretize_matches_cpu_within_one_bin():
    """Same numeric bound on the REAL device kernel, when CUDA is available (skips otherwise)."""
    try:
        from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _quantile_bin_gpu, _cmi_gpu_enabled
    except Exception:
        pytest.skip("gpu quantile-bin path unavailable")
    if not _cmi_gpu_enabled():
        pytest.skip("CUDA / cupy not available")
    rng = np.random.default_rng(2)
    x = rng.standard_normal(60_000).astype(np.float64)
    for nbins in (10, 32):
        gpu = _quantile_bin_gpu(x, nbins)
        if gpu is None:
            pytest.skip("gpu binner returned None (fault / disabled)")
        cpu = np.asarray(_quantile_bin(x, nbins)).astype(np.int64)
        diff = np.abs(np.asarray(gpu).astype(np.int64) - cpu)
        assert diff.max() <= 1, f"GPU-resident f32 discretize differs from CPU f64 by {diff.max()} > 1 bin (nbins={nbins})"
        assert float(np.mean(diff > 0)) < 1e-3, "more than 0.1% of rows differ -- exceeds the documented boundary sliver"
