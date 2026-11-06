"""The one-pass honest-null root-split kernels (versions 4, 4f and 5) against version 2."""

import numpy as np
import pytest

_SHAPES = [(997, 5, 7, 0.4), (4096, 33, 64, 0.5), (12000, 130, 20, 0.05), (3000, 9, 30, 0.9)]


def _run(n, kc, B, p1):
    """Outputs of v2, v4 (full table), v4f and v5 on one random frame; skips without a usable device."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._family_maxt_null_v2_njit import xlogx_table
    from mlframe.feature_selection.filters._honest_null_device import XLOGX_TABLE_SIZE, _entropy, _get_kernel_k2, _pack_bits, _tie_flags
    from mlframe.feature_selection.filters._honest_null_k4 import get_kernel_k4, get_kernel_k4f, get_kernel_k5
    from mlframe.feature_selection.filters._mdlp_honest_njit import _presort_columns

    rng = np.random.default_rng(2)
    raw = np.round(rng.normal(size=(n, kc)), 1) if kc % 2 else rng.normal(size=(n, kc))  # ties on odd widths
    order = _presort_columns(raw)
    tie = _tie_flags(raw, order)
    y = (rng.random(n) < p1).astype(np.int8)
    yperm = np.stack([rng.permutation(y) for _ in range(B)])
    counts = np.bincount(yperm[0].astype(np.int64), minlength=2)
    h_full = _entropy(counts, n)
    ybits = _pack_bits(yperm)
    common = (cp.asarray(order), cp.asarray(tie), cp.asarray(np.ascontiguousarray(ybits.T).view(np.uint8)))
    tail = (cp.asarray(counts.astype(np.int64)), np.int32(n), np.int32(B), np.int32(kc), np.int32(60), np.float64(h_full), np.int32(ybits.shape[1]))
    small, full = cp.asarray(xlogx_table(XLOGX_TABLE_SIZE)), cp.asarray(xlogx_table(n))
    outs = {}
    for name, kernel, tab in (("k2", _get_kernel_k2(cp), small), ("k4", get_kernel_k4(cp), full), ("k4f", get_kernel_k4f(cp), small), ("k5", get_kernel_k5(cp), small)):
        g, c, i = cp.empty(B * kc, dtype=cp.float64), cp.empty(B * kc, dtype=cp.int32), cp.empty(B * kc, dtype=cp.int32)
        kernel(((B * kc + 127) // 128,), (128,), (*common, tab, *tail, g, c, i))
        outs[name] = (cp.asnumpy(g), cp.asnumpy(c), cp.asnumpy(i))
    return outs


@pytest.mark.parametrize("n,kc,B,p1", _SHAPES)
def test_k4_equals_k2(n, kc, B, p1):
    """Version 4 (exact, table over every count): same index and candidate count, gain within the device log's rounding."""
    o = _run(n, kc, B, p1)
    np.testing.assert_allclose(o["k4"][0], o["k2"][0], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(o["k4"][1], o["k2"][1])
    np.testing.assert_array_equal(o["k4"][2], o["k2"][2])


@pytest.mark.parametrize("n,kc,B,p1", _SHAPES)
def test_float_variants_are_close_to_k2(n, kc, B, p1):
    """The kept float variants count the same candidates and give gains within their documented error."""
    o = _run(n, kc, B, p1)
    for name, atol in (("k4f", 1e-5), ("k5", 1e-6)):
        np.testing.assert_array_equal(o[name][1], o["k2"][1])
        np.testing.assert_allclose(o[name][0], o["k2"][0], rtol=0, atol=atol)
