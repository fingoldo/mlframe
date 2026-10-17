"""The cat-FE permutation sweep must feed inputs on which every backend returns the same exact count."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _cat_confirm_permutation_tuning as m


def _inputs(n: int, n_perms: int) -> tuple:
    """Sweep inputs for a small cell."""
    return m._make_perm_kernel_inputs({"n_samples": n, "n_perms": n_perms})


@pytest.mark.parametrize("n_perms", [10, 50])
def test_cpu_backends_count_every_permutation_whatever_the_seed(n_perms):
    """ii_obs sits below any attainable II, so serial and prange both return n_perms for any seed."""
    args = _inputs(20_000, n_perms)
    for seed in (1, 7, 12345):
        assert int(m._count_nfailed_joint_indep_prange(*args, seed, np.int32)) == n_perms
        assert int(m._count_nfailed_joint_indep_serial(*args, seed, np.int32)) == n_perms


def _has_cuda_device() -> bool:
    """True when cupy imports and a CUDA device is usable (evaluated at collection time, so the skip is declared rather than raised mid-test)."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() >= 1
    except Exception:
        return False


needs_cuda = pytest.mark.skipif(not _has_cuda_device(), reason="needs cupy and a CUDA device")


@needs_cuda
def test_cupy_backend_agrees_exactly_with_the_cpu_reference():
    """The GPU RNG stream differs from numba's, yet the count is the same exact n_perms (non-vacuous: the old ii_obs=0 gave 5 vs 7)."""
    args = _inputs(50_000, 50)
    assert int(m._count_nfailed_joint_indep_cupy(*args, base_seed=7)) == int(m._count_nfailed_joint_indep_prange(*args, 7, np.int32)) == 50


@needs_cuda
def test_old_threshold_would_have_made_backend_counts_diverge():
    """Negative control: with ii_obs=0.0 (the previous value) the backends count different permutations, so the exact check needs the new input."""
    args = list(_inputs(100_000, 50))
    args[8] = 0.0
    cpu = int(m._count_nfailed_joint_indep_prange(*args, 7, np.int32))
    gpu = int(m._count_nfailed_joint_indep_cupy(*args, base_seed=7))
    assert cpu != 50 or gpu != 50
