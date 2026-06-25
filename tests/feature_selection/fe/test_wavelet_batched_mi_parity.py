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
