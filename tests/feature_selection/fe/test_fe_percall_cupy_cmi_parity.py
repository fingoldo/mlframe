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
