"""Parity: the batched born-on-device leg-MI primitive == the per-column CPU plug-in MI (selection-equiv).

Pins the replatform-step-1 primitive (``batched_binned_mi_gpu``) so it can be wired under STRICT knowing it
reproduces the per-leg ``_binned_mi`` ranking. cupy-gated (importorskip)."""
import numpy as np
import pytest


def test_wavelet_batched_mi_matches_per_column_cpu():
    pytest.importorskip("cupy")
    import mlframe.feature_selection.filters.hermite_fe  # noqa: F401  (resolve import cycle first)
    from mlframe.feature_selection.filters._wavelet_basis_fe import _binned_mi
    from mlframe.feature_selection.filters._wavelet_basis_fe_batched import batched_binned_mi_gpu

    rng = np.random.default_rng(0)
    n, K = 4000, 24
    Ky = 6
    y = rng.integers(0, Ky, n).astype(np.int64)
    # heterogeneous per-column cardinalities (legs have different joint-code spans)
    kx = rng.integers(2, 12, K)
    cols = np.empty((n, K), dtype=np.int64)
    for k in range(K):
        cols[:, k] = rng.integers(0, kx[k], n)

    mi_batched = batched_binned_mi_gpu(cols, y, kx_per_col=kx, ky=Ky)
    mi_cpu = np.array([
        _binned_mi(cols[:, k].astype(np.float64), y, nbins=int(kx[k]) + 1) for k in range(K)
    ])

    # numerically equal (plug-in MI is partition-based; value-order vs first-seen labels do not change it)
    assert mi_batched.shape == (K,)
    assert np.allclose(mi_batched, mi_cpu, atol=1e-9), (
        f"max|d|={np.max(np.abs(mi_batched - mi_cpu)):.2e}"
    )
    # ranking identical -> the leg SELECTION the batched path would produce matches the CPU path
    assert np.array_equal(np.argsort(mi_batched), np.argsort(mi_cpu)) or np.allclose(mi_batched, mi_cpu, atol=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_select_wavelet_legs_batched_same_admitted(seed):
    """The batched born-on-device leg selector returns the SAME admitted (j,k) legs as the CPU path."""
    pytest.importorskip("cupy")
    import mlframe.feature_selection.filters.hermite_fe  # noqa: F401
    from mlframe.feature_selection.filters._wavelet_basis_fe import (
        _WAVELET_MAX_LEGS,
        _WAVELET_MAX_SCALE,
        _WAVELET_SCALE_SIGMA,
        _select_wavelet_legs,
    )
    from mlframe.feature_selection.filters._wavelet_basis_fe_batched import select_wavelet_legs_batched

    rng = np.random.default_rng(seed)
    n = 6000
    x = rng.uniform(0.0, 1.0, n)
    # localized step in a sub-interval -> a genuine Haar leg carries signal, plus noise
    y = (((x > 0.30) & (x < 0.55)).astype(np.float64) + 0.4 * rng.standard_normal(n))
    yb = np.digitize(y, np.quantile(y, np.linspace(0, 1, 11)[1:-1])).astype(np.int64)
    kw = dict(max_scale=_WAVELET_MAX_SCALE, max_legs=_WAVELET_MAX_LEGS, scale_sigma=_WAVELET_SCALE_SIGMA)
    cpu = _select_wavelet_legs(x, yb, 0.0, 1.0, **kw)
    gpu = select_wavelet_legs_batched(x, yb, 0.0, 1.0, **kw)
    assert cpu == gpu, f"seed={seed}: CPU legs {cpu} != batched {gpu}"


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_fused_mi_from_codes_matches_cupy_batched(seed):
    """Fused one-launch MI-from-codes RawKernel == the cupy batched_binned_mi_gpu (plug-in MI parity)."""
    pytest.importorskip("cupy")
    import numpy as _np
    from mlframe.feature_selection.filters._fe_batched_mi import binned_mi_from_codes_gpu
    from mlframe.feature_selection.filters._wavelet_basis_fe_batched import batched_binned_mi_gpu
    rng = _np.random.default_rng(seed)
    n, K = 6000, 20
    kx = rng.integers(2, 4, K)
    C = _np.empty((n, K), _np.int64)
    for k in range(K):
        C[:, k] = rng.integers(0, kx[k], n)
    y = rng.integers(0, 10, n).astype(_np.int64)
    a = batched_binned_mi_gpu(C, y, kx_per_col=kx, ky=10)
    b = binned_mi_from_codes_gpu(C, y, kx_per_col=kx, ky=10)
    assert _np.allclose(a, b, atol=1e-9), f"seed={seed} max|d|={_np.max(_np.abs(a-b)):.2e}"
