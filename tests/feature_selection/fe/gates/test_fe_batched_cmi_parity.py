"""Parity: batched born-on-device CMI == per-call CPU _cmi_from_binned (selection-equivalent).

Pins replatform-step-3 primitive (batched_cmi_gpu) so it can replace the launch-bound per-call cp.unique
CMI ports under STRICT knowing it reproduces the CPU plug-in CMI. cupy-gated."""

import numpy as np
import pytest


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
@pytest.mark.parametrize("dist", ["uniform", "normal", "lognormal"])
def test_batched_quantile_bin_gpu_selection_equivalent(seed, dist):
    """Born-on-device equi-freq binning == host _quantile_bin (selection-equivalent CMI; codes identical
    on continuous data). Pins that GPU-resident binning feeding batched_cmi_gpu reproduces the host path."""
    pytest.importorskip("cupy")
    import cupy as cp
    from mlframe.feature_selection.filters._fe_batched_mi import batched_quantile_bin_gpu, batched_cmi_gpu
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _quantile_bin, _cmi_from_binned

    rng = np.random.default_rng(seed)
    n, K = 8000, 12
    X = {"uniform": rng.uniform(0, 1, (n, K)), "normal": rng.normal(0, 1, (n, K)), "lognormal": rng.lognormal(0, 1, (n, K))}[dist]
    y = rng.integers(0, 6, n).astype(np.int64)
    z = rng.integers(0, 5, n).astype(np.int64)

    Hc = np.empty((n, K), np.int64)
    for k in range(K):
        Hc[:, k] = _quantile_bin(X[:, k], nbins=10)
    Gc = batched_quantile_bin_gpu(cp.asarray(X), 10)
    for z_ in (None, z):
        cpu = np.array([_cmi_from_binned(Hc[:, k], y, z_) for k in range(K)])
        gpu = batched_cmi_gpu(Gc, y, z_)
        assert np.allclose(cpu, gpu, atol=1e-9), f"{dist} seed={seed} max|d|={np.max(np.abs(cpu - gpu)):.2e}"


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
@pytest.mark.parametrize("conditional", [False, True])
def test_batched_cmi_matches_cpu(seed, conditional):
    """batched_cmi_gpu matches the per-column CPU _cmi_from_binned to atol=1e-9, both marginal and Z-conditional."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._fe_batched_mi import batched_cmi_gpu
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _cmi_from_binned

    rng = np.random.default_rng(seed)
    n, K = 5000, 16
    y = rng.integers(0, 6, n).astype(np.int64)
    kx = rng.integers(2, 10, K)
    X = np.empty((n, K), dtype=np.int64)
    for k in range(K):
        X[:, k] = rng.integers(0, kx[k], n)
    z = rng.integers(0, 5, n).astype(np.int64) if conditional else None

    batched = batched_cmi_gpu(X, y, z)
    cpu = np.array([_cmi_from_binned(X[:, k], y, z) for k in range(K)])
    assert batched.shape == (K,)
    assert np.allclose(batched, cpu, atol=1e-9), f"seed={seed} cond={conditional} max|d|={np.max(np.abs(batched - cpu)):.2e}"
