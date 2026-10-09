"""The resident plug-in MI transposes its input once, and its result is unchanged."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _gpu_resident_select as sel
from mlframe.feature_selection.filters._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident


def _data(n, k, dtype, seed=0):
    """A resident candidate matrix and a class vector that depends on its first column."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k)).astype(dtype)
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(np.int64) + (X[:, -1] > 1).astype(np.int64)
    return cp.asarray(X), cp.asarray(y)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_one_transpose_per_call(monkeypatch, dtype):
    """Counting calls of the tiled transpose: a call used to transpose the matrix once for the edges and once for the MI kernel."""
    X, y = _data(20_000, 12, dtype)
    calls = []
    real = sel._transpose_to_cm
    monkeypatch.setattr(sel, "_transpose_to_cm", lambda a: calls.append(a.shape) or real(a))
    import mlframe.feature_selection.filters._fe_batched_mi as fbm
    monkeypatch.setattr(fbm, "_transpose_to_cm", lambda a: calls.append(a.shape) or real(a), raising=False)
    mi = _plugin_mi_classif_batch_cuda_resident(X, y, 10)
    assert mi.shape == (12,)
    assert len([c for c in calls if c == (20_000, 12)]) == 1, calls


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_result_matches_the_percentile_path(dtype):
    """Edges from the radix select equal cp.percentile's, so the MI equals what the percentile fallback gives."""
    X, y = _data(15_000, 8, dtype, seed=3)
    got = _plugin_mi_classif_batch_cuda_resident(X, y, 10)
    qs = cp.linspace(0.0, 100.0, 11)
    edges = cp.percentile(X.astype(cp.float64), qs, axis=0)[1:-1]
    codes = cp.stack([cp.searchsorted(edges[:, j], X[:, j].astype(cp.float64), side="right") for j in range(8)], axis=1)
    yy = cp.asnumpy(y)
    cc = cp.asnumpy(codes)
    want = []
    for j in range(8):
        joint = np.zeros((cc[:, j].max() + 1, yy.max() + 1))
        np.add.at(joint, (cc[:, j], yy), 1)
        p = joint / joint.sum()
        px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
        nz = p > 0
        want.append(float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum()))
    np.testing.assert_allclose(got, want, atol=1e-9)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("relax", [False, True])
def test_column_major_input_gives_identical_mi(dtype, relax):
    """A (k, n) block scores the same numbers, bit for bit, as the (n, k) matrix it is the transpose of."""
    X, y = _data(25_000, 16, dtype, seed=7)
    row = _plugin_mi_classif_batch_cuda_resident(X, y, 10, relax_binning=relax)
    col = _plugin_mi_classif_batch_cuda_resident(cp.ascontiguousarray(X.T), y, 10, relax_binning=relax, x_is_cm=True)
    np.testing.assert_array_equal(col, row)


def test_column_major_input_with_the_percentile_fallback(monkeypatch):
    """With the radix path disabled the block is transposed back for the percentile route and the answer is unchanged."""
    X, y = _data(8_000, 6, np.float64, seed=9)
    monkeypatch.setenv("MLFRAME_FE_GPU_RADIX_EDGES", "0")
    row = _plugin_mi_classif_batch_cuda_resident(X, y, 10)
    col = _plugin_mi_classif_batch_cuda_resident(cp.ascontiguousarray(X.T), y, 10, x_is_cm=True)
    np.testing.assert_array_equal(col, row)


def test_single_column_block():
    """k = 1: the (1, n) block."""
    X, y = _data(6_000, 1, np.float64, seed=11)
    row = _plugin_mi_classif_batch_cuda_resident(X, y, 10)
    col = _plugin_mi_classif_batch_cuda_resident(cp.ascontiguousarray(X.T), y, 10, x_is_cm=True)
    np.testing.assert_array_equal(col, row)
