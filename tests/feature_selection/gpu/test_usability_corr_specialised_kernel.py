"""The form-specialised usability kernel and the float32 upload give the generic kernel's bits."""

import numpy as np
import pytest

pytest.importorskip("cupy")

from mlframe.feature_selection.filters import batch_pair_usability_corr_gpu as B

if not B._CUDA_AVAIL:  # pragma: no cover - host without a usable numba.cuda
    pytest.skip("numba.cuda is not usable on this host", allow_module_level=True)


def _data(n=4000, k=30, seed=0):
    """Operands with a NaN column, a constant column and an offset-heavy column, a target with a NaN, and all pairs."""
    rng = np.random.default_rng(seed)
    ops = (rng.standard_normal((k, n)) * 3 + rng.random((k, 1))).astype(np.float64)
    ops[3, ::40] = np.nan
    ops[5] = 2.0
    ops[8] = ops[8] + 1e7
    ops[11, ::7] = 0.0
    y = rng.standard_normal(n)
    y[::53] = np.nan
    ia, ib = np.triu_indices(k, 1)
    return y, ops, ia.astype(np.int64), ib.astype(np.int64)


@pytest.mark.parametrize("forms", [B.ALL_PAIR_FORM_IDS, B.ALL_FORM_IDS, np.array([B.FORM_X0_MUL_X1], dtype=np.int64)])
def test_specialised_equals_generic_bitwise(monkeypatch, forms):
    """Every form id, with unsafe (NaN / offset) columns that move some pairs to the double kernel."""
    y, ops, pa, pb = _data()
    monkeypatch.setenv("MLFRAME_USABILITY_CORR_KERNEL", "generic")
    want = B.batch_pair_usability_corr_cuda_warp(y, ops, pa, pb, forms)
    monkeypatch.setenv("MLFRAME_USABILITY_CORR_KERNEL", "specialised")
    got = B.batch_pair_usability_corr_cuda_warp(y, ops, pa, pb, forms)
    assert np.array_equal(got, want)


def test_float32_inputs_and_float64_inputs_give_the_same_bits():
    """On finite, well-scaled data the host float32 conversion before the upload is the kernel's own round-to-nearest conversion."""
    rng = np.random.default_rng(1)
    ops = (rng.standard_normal((30, 4000)) * 3 + rng.random((30, 1))).astype(np.float64)
    y = rng.standard_normal(4000)
    ia, ib = np.triu_indices(30, 1)
    pa, pb = ia.astype(np.int64), ib.astype(np.int64)
    a = B.batch_pair_usability_corr_cuda_warp(y, ops, pa, pb, B.ALL_PAIR_FORM_IDS)
    b = B.batch_pair_usability_corr_cuda_warp(y.astype(np.float32), ops.astype(np.float32), pa, pb, B.ALL_PAIR_FORM_IDS)
    assert np.array_equal(a, b)
