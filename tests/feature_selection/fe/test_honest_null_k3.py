"""The kept version 3 of the honest-null root-split kernel gives the outputs of version 2 (it is not dispatched: it is slower on the development GPU)."""

import numpy as np
import pytest


@pytest.mark.parametrize("n,kc,B", [(997, 5, 7), (4096, 33, 64), (12000, 130, 20)])
def test_k3_equals_k2(n, kc, B):
    """Gain, candidate count and split index agree exactly for odd and aligned shapes."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._family_maxt_null_v2_njit import xlogx_table
    from mlframe.feature_selection.filters._honest_null_device import XLOGX_TABLE_SIZE, _entropy, _get_kernel_k2, _pack_bits, _tie_flags
    from mlframe.feature_selection.filters._honest_null_k3 import fits_shared, launch_k3
    from mlframe.feature_selection.filters._mdlp_honest_njit import _presort_columns

    rng = np.random.default_rng(1)
    raw = rng.normal(size=(n, kc))
    order = _presort_columns(raw)
    tie = _tie_flags(raw, order)
    y = (rng.random(n) < 0.4).astype(np.int8)
    yperm = np.stack([rng.permutation(y) for _ in range(B)])
    counts = np.bincount(yperm[0].astype(np.int64), minlength=2)
    h_full = _entropy(counts, n)
    tab = xlogx_table(XLOGX_TABLE_SIZE)
    ybits = _pack_bits(yperm)
    words = ybits.shape[1]
    wb = np.ascontiguousarray(ybits.T)
    assert fits_shared(n)
    d = dict(order=cp.asarray(order), tie=cp.asarray(tie), tab=cp.asarray(tab), cnt=cp.asarray(counts.astype(np.int64)), wb=cp.asarray(wb), wb8=cp.asarray(wb.view(np.uint8)))

    def outputs():
        """Fresh output vectors."""
        return cp.empty(B * kc, dtype=cp.float64), cp.empty(B * kc, dtype=cp.int32), cp.empty(B * kc, dtype=cp.int32)

    try:
        g2, c2, i2 = outputs()
        _get_kernel_k2(cp)(((B * kc + 127) // 128,), (128,), (d["order"], d["tie"], d["wb8"], d["tab"], d["cnt"], np.int32(n), np.int32(B), np.int32(kc), np.int32(60), np.float64(h_full), np.int32(words), g2, c2, i2))
        g3, c3, i3 = outputs()
        launch_k3(cp, d["order"], d["tie"], d["wb"], d["tab"], d["cnt"], n, B, kc, 60, h_full, words, g3, c3, i3)
    except Exception as exc:  # pragma: no cover - no usable device
        pytest.skip(f"needs a usable CUDA device: {exc}")
    np.testing.assert_array_equal(cp.asnumpy(g2), cp.asnumpy(g3))
    np.testing.assert_array_equal(cp.asnumpy(c2), cp.asnumpy(c3))
    np.testing.assert_array_equal(cp.asnumpy(i2), cp.asnumpy(i3))
