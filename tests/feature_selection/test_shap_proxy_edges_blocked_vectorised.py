"""The blocked edge collector returns the same (i, j) pairs in the same order as the per-row scan it replaced, on the host and on the device, and refuses (None) exactly when the edge count passes the cap."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_cluster as sc


def _reference(Z, threshold, edge_cap, block):
    """The per-row scan (numpy)."""
    n, f = Z.shape
    ei, ej, total = [], [], 0
    for b0 in range(0, f, block):
        b1 = min(b0 + block, f)
        absCb = np.abs((Z[:, b0:b1].T @ Z) / np.float32(n))
        for r in range(b1 - b0):
            i = b0 + r
            js = np.where(absCb[r] > np.float32(threshold))[0]
            js = js[js > i]
            if js.size:
                ei.append(np.full(js.shape[0], i, dtype=np.int64))
                ej.append(js.astype(np.int64))
                total += int(js.shape[0])
                if total > edge_cap:
                    return None
    if not ei:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(ei), np.concatenate(ej)


def _data(n=400, f=70, seed=0):
    """Standardised columns with planted correlated groups."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, f)).astype(np.float32)
    for a, b in ((3, 40), (5, 6), (68, 1), (20, 21), (21, 22)):
        Z[:, b] = Z[:, a] * 0.95 + 0.05 * Z[:, b]
    return ((Z - Z.mean(0)) / Z.std(0)).astype(np.float32)


@pytest.mark.parametrize("use_gpu", [False, True])
@pytest.mark.parametrize("block", [1, 7, 32, 100])
@pytest.mark.parametrize("threshold", [0.3, 0.7, 0.99])
def test_pairs_and_order_equal_the_row_scan(use_gpu, block, threshold):
    """Equal arrays; with an edge cap one below the count the result is None, at the count it is the pairs."""
    if use_gpu:
        cp = pytest.importorskip("cupy")
        try:
            cp.cuda.runtime.getDeviceCount()
        except Exception:
            pytest.skip("no usable CUDA device")
    Z = _data()
    want = _reference(Z, threshold, 10**9, block)
    got = sc._edges_blocked(Z, threshold, 10**9, block, use_gpu)
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])
    count = want[0].size
    if count:
        assert sc._edges_blocked(Z, threshold, count - 1, block, use_gpu) is None
        assert sc._edges_blocked(Z, threshold, count, block, use_gpu) is not None
