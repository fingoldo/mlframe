"""Parity: the per-call cupy CMI routes (bincount-based, post 2026-06-25) == per-call CPU path.

The per-call device CMI helpers (``_cmi_from_binned_cupy``, ``joint_cardinalities_cupy``,
``_cmi_from_binned_fixed_yz_cupy``) count partitions via ``cp.bincount`` (one kernel) instead of
``cp.unique`` (a sort = 5-10 cub MergeSort launches per call). This pins that the cheaper counting is
selection-equivalent to the CPU ``_cmi_from_binned`` so it can stay the STRICT/CMI_GPU per-call route.
cupy-gated."""

import numpy as np
import pytest


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
@pytest.mark.parametrize("conditional", [False, True])
def test_percall_cupy_cmi_matches_cpu(seed, conditional):
    """The bincount-based per-call cupy CMI helper matches the CPU _cmi_from_binned to 1e-9, both marginal and Z-conditional."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe as M

    rng = np.random.default_rng(seed)
    n = 5000
    x = rng.integers(0, 9, n).astype(np.int64)
    y = rng.integers(0, 6, n).astype(np.int64)
    z = rng.integers(0, 5, n).astype(np.int64) if conditional else None

    cpu = M._cmi_from_binned(x, y, z)
    gpu = M._cmi_from_binned_cupy(x, y, z)
    assert abs(cpu - gpu) < 1e-9, f"seed={seed} cond={conditional} cpu={cpu} gpu={gpu}"


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_joint_cardinalities_cupy_matches_host(seed):
    """joint_cardinalities_cupy's four partition cardinalities exactly match the host np.unique-based reference."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe as M

    rng = np.random.default_rng(seed)
    n = 5000
    x = rng.integers(0, 9, n).astype(np.int64)
    y = rng.integers(0, 6, n).astype(np.int64)
    z = rng.integers(0, 5, n).astype(np.int64)

    kz = int(z.max()) + 1
    yz = y * kz + z
    big = int(yz.max()) + 1
    host = (
        int(np.unique(z).size),
        int(np.unique(x * kz + z).size),
        int(np.unique(yz).size),
        int(np.unique(x * big + yz).size),
    )
    gpu = M.joint_cardinalities_cupy(x, y, z)
    assert tuple(gpu) == host, f"seed={seed} host={host} gpu={tuple(gpu)}"


@pytest.mark.parametrize("conditional", [False, True])
def test_cmi_from_binned_accepts_device_resident_x_when_gate_says_cpu(conditional):
    """Regression: _cmi_from_binned's shape-aware _cmi_gpu_enabled(n=x.size, p=1) gate can legitimately
    say "small n, use CPU" while x is ALREADY a cupy array (e.g. _step_score.py's DEVICE-BORN marginal-MI
    path bins a candidate via _quantile_bin_gpu_resident BEFORE calling here, independent of this call's
    own n/p heuristic). Pre-fix this crashed with 'TypeError: Implicit conversion to a NumPy array is not
    allowed' from np.ascontiguousarray(cupy_array) -- reproduced live during a production wellbore fit."""
    pytest.importorskip("cupy")
    import cupy as cp

    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe as M

    rng = np.random.default_rng(3)
    n = 8  # tiny -- the shape-aware gate should say "not worth GPU" for this n on its own
    x_host = rng.integers(0, 4, n).astype(np.int64)
    y_host = rng.integers(0, 3, n).astype(np.int64)
    z_host = rng.integers(0, 2, n).astype(np.int64) if conditional else None

    assert not M._cmi_gpu_enabled(n=n, p=1), "fixture assumption: gate must say False at this tiny n"

    x_dev = cp.asarray(x_host)
    result = M._cmi_from_binned(x_dev, y_host, z_host)
    expected = M._cmi_from_binned(x_host, y_host, z_host)
    assert abs(result - expected) < 1e-9


def test_cmi_from_binned_fixed_yz_accepts_device_resident_x_when_gate_says_cpu():
    """Same regression as test_cmi_from_binned_accepts_device_resident_x_when_gate_says_cpu, for the
    y/z-hoisted sibling cmi_from_binned_fixed_yz (identical bug pattern, separate function body)."""
    pytest.importorskip("cupy")
    import cupy as cp

    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe as M

    rng = np.random.default_rng(4)
    n = 8
    x_host = rng.integers(0, 4, n).astype(np.int64)
    y_host = rng.integers(0, 3, n).astype(np.int64)
    z_host = rng.integers(0, 2, n).astype(np.int64)

    assert not M._cmi_gpu_enabled(n=n, p=1), "fixture assumption: gate must say False at this tiny n"

    y_i, z_i, h_yz, h_z, k_yz, k_z, n_f = M.precompute_cmi_yz_terms(y_host, z_host)
    x_dev = cp.asarray(x_host)
    result = M.cmi_from_binned_fixed_yz(x_dev, y_i, z_i, h_yz, h_z, k_yz, k_z, n_f)
    expected = M.cmi_from_binned_fixed_yz(x_host, y_i, z_i, h_yz, h_z, k_yz, k_z, n_f)
    assert abs(result - expected) < 1e-9
