"""The window table of the second radix pass (v5 with ``MLFRAME_RADIX_F32_WINMAP``) gives the order statistics of the plain v5 bit for bit, for every row count and column shape."""

from __future__ import annotations

import numpy as np
import pytest

from tests._cuda_require import require_cuda_device


def _edges(cp, X, monkeypatch, winmap: bool, nbins: int):
    """The interior quantile edges of ``X`` through the radix select with the window-table switch set."""
    from mlframe.feature_selection.filters import _gpu_resident_select as sel

    monkeypatch.setenv("MLFRAME_RADIX_F32_WINMAP", "1" if winmap else "0")
    return cp.asnumpy(sel._radix_select_interior_edges(X.copy(), nbins))


@pytest.mark.parametrize("nbins", [4, 10, 20, 38])
@pytest.mark.parametrize("n", [97, 4097, 20011, 100003])
@pytest.mark.parametrize("kind", ["normal", "heavy", "uniform", "ties", "constant", "mixed_sign_zero", "wide_exponent"])
def test_window_table_equals_plain_v5(monkeypatch, n, kind, nbins):
    """Bell-shaped, heavy-tailed, tied, constant, signed-zero and many-exponent columns, at rank counts below and above 32."""
    require_cuda_device()
    import cupy as cp

    rng = np.random.default_rng(n + nbins)
    k = 5
    if kind == "normal":
        raw = rng.standard_normal((n, k))
    elif kind == "heavy":
        raw = rng.standard_normal((n, k)) ** 2 / np.maximum(rng.uniform(0.05, 1.1, (n, k)), 0.05)
    elif kind == "uniform":
        raw = rng.random((n, k))
    elif kind == "ties":
        raw = np.round(rng.standard_normal((n, k)) * 3)
    elif kind == "constant":
        raw = np.full((n, k), 2.5)
    elif kind == "wide_exponent":
        raw = rng.standard_normal((n, k)) * np.exp(rng.uniform(-30, 30, (n, k)))
    else:
        raw = np.where(rng.random((n, k)) < 0.3, 0.0, rng.standard_normal((n, k)))
        raw[::7] *= -1.0
    X = cp.asarray(raw.astype(np.float32))
    got = _edges(cp, X, monkeypatch, True, nbins)
    want = _edges(cp, X, monkeypatch, False, nbins)
    assert np.array_equal(got, want)


@pytest.mark.parametrize("threads", [48, 96, 256, 384, 768, 1024])
@pytest.mark.parametrize("nbins", [10, 38])
def test_warp_scan_equals_serial_scan_at_every_block_size(monkeypatch, threads, nbins):
    """The per-rank scan by a warp (block a whole number of warps) and the serial scan (48 threads is not) give the same edges, with fewer warps than ranks as well (96 threads, 38 ranks)."""
    require_cuda_device()
    import cupy as cp

    from mlframe.feature_selection.filters import _gpu_resident_select_kernels as kernels

    rng = np.random.default_rng(threads + nbins)
    X = cp.asarray((rng.standard_normal((6000, 4)) * np.exp(rng.uniform(-3, 3, (6000, 4)))).astype(np.float32))
    monkeypatch.setattr(kernels, "_RADIX_THREADS_OVERRIDE", threads)
    got = _edges(cp, X, monkeypatch, True, nbins)
    want = _edges(cp, X, monkeypatch, False, nbins)
    assert np.array_equal(got, want)
