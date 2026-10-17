"""Symmetric-uncertainty noise gate runs from resident counts (device SU kernel) and matches the host reference."""

import numpy as np
import pytest

import mlframe.feature_selection.filters.batch_mi_noise_gate_gpu as bg
from mlframe.feature_selection.filters.info_theory import batch_mi_with_noise_gate, merge_vars

pytestmark = pytest.mark.skipif(not bg._CUDA_AVAIL, reason="numba.cuda not available on this host")


def _frame(n, K, nbins, ky, seed):
    """Columns that track the target to different degrees, plus pure noise."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, ky, size=n).astype(np.int32)
    cols = np.empty((n, K), dtype=np.int32)
    for k in range(K):
        c = (y + rng.integers(0, 3, size=n)) % nbins if k % 3 == 0 else rng.integers(0, nbins, size=n) if k % 3 == 1 else y.astype(np.int64)
        cols[:, k] = (c % nbins).astype(np.int32)
    cy, fy, _ = merge_vars(factors_data=y.reshape(-1, 1), vars_indices=np.array([0], dtype=np.int64), var_is_nominal=None, factors_nbins=np.array([ky], dtype=np.int64), dtype=np.int32)
    return cols, cy, fy


@pytest.mark.parametrize("fused", ["0", "1"])
@pytest.mark.parametrize("ky", [2, 5])
def test_su_gate_from_resident_counts_matches_the_host_gate(ky, fused, monkeypatch):
    """The gated SU vector of the resident path equals the host kernel's (fp round-off), and the global-atomic per-permutation kernel is not launched."""
    monkeypatch.setenv("MLFRAME_FE_GPU_FUSED_GATE", fused)
    n, K, nbins, nperm = 3000, 45, 8, 20
    disc, cy, fy = _frame(n, K, nbins, ky, seed=ky)
    nb = np.full(K, nbins, dtype=np.int64)
    kw = dict(disc_2d=disc, factors_nbins=nb, classes_y=cy, classes_y_safe=cy.copy(), freqs_y=fy, npermutations=nperm, base_seed=np.uint64(0), min_nonzero_confidence=0.99, use_su=True, dtype=np.int32)
    ref = batch_mi_with_noise_gate(**kw)
    bg._CUDA_HIST_KERNEL = None
    got = bg.batch_mi_with_noise_gate_cuda_resident(**kw)
    assert bg._CUDA_HIST_KERNEL is None  # the legacy per-permutation histogram was never built
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("use_su", [False, True])
def test_fused_and_two_kernel_gates_agree(use_su, monkeypatch):
    """The fused kernel (no counts matrix) returns the two-kernel gate's vector to fp round-off, for MI and for SU."""
    n, K, nbins, nperm = 2500, 60, 10, 15
    disc, cy, fy = _frame(n, K, nbins, 3, seed=9)
    nb = np.full(K, nbins, dtype=np.int64)
    kw = dict(disc_2d=disc, factors_nbins=nb, classes_y=cy, classes_y_safe=cy.copy(), freqs_y=fy, npermutations=nperm, base_seed=np.uint64(0), min_nonzero_confidence=0.99, use_su=use_su, dtype=np.int32)
    monkeypatch.setenv("MLFRAME_FE_GPU_FUSED_GATE", "0")
    two = bg.batch_mi_with_noise_gate_cuda_resident(**kw)
    monkeypatch.setenv("MLFRAME_FE_GPU_FUSED_GATE", "1")
    fused = bg.batch_mi_with_noise_gate_cuda_resident(**kw)
    np.testing.assert_allclose(fused, two, atol=1e-12, rtol=1e-12)
