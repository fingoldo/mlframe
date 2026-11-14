"""The device twins of the residual rescue (bin codes of the pool, column scoring) equal the numba loops, and the dispatchers route to them and fall back from a device fault."""

from __future__ import annotations

import numpy as np
import pytest

from tests._cuda_require import require_cuda_device


def _matrix(dtype, n: int, p: int, seed: int) -> np.ndarray:
    """A pool matrix with NaN, infinities, a constant column, a nominal-like column and a mostly missing column among normal ones."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((p, n)).astype(dtype)
    M[0, ::7] = np.nan
    M[1, ::5] = np.inf
    M[2] = 3.0
    M[3] = np.round(M[3] * 2)
    M[4, : max(n - 600, 0)] = np.nan
    M[5] = rng.integers(0, 3, n)
    return M


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n,p,bins", [(900, 10, 8), (5000, 40, 16), (30011, 60, 24)])
def test_codes_equal_the_numba_loop(dtype, n, p, bins):
    """Integer codes and the usable mask are equal, for both float types."""
    require_cuda_device()
    from mlframe.feature_selection.filters import _residual_rescue_fe as rescue
    from mlframe.feature_selection.filters._residual_rescue_gpu import codes_matrix_gpu
    from mlframe.feature_selection.filters._y_encoding import FEW_CLASSES_MAX

    M = _matrix(dtype, n, p, n + p)
    c0, ok0 = rescue._codes_matrix(M, bins, rescue.CODE_SAMPLE, rescue.MIN_ROWS, FEW_CLASSES_MAX)
    c1, ok1 = codes_matrix_gpu(M, bins, rescue.CODE_SAMPLE, rescue.MIN_ROWS, FEW_CLASSES_MAX)
    assert np.array_equal(ok0, ok1)
    assert np.array_equal(c0[ok0], c1[ok1])
    assert int(c1[~ok1].sum()) == 0


@pytest.mark.parametrize("n,p,bins", [(2000, 12, 8), (25000, 80, 16), (30001, 7, 24)])
def test_scores_equal_the_numba_loop(n, p, bins):
    """t statistic and gain are equal to 1e-7 (the bin sums are 2^-36 fixed-point integers on the device), with an odd row count and few columns as well."""
    require_cuda_device()
    from mlframe.feature_selection.filters import _residual_rescue_fe as rescue
    from mlframe.feature_selection.filters._residual_rescue_gpu import score_columns_gpu

    rng = np.random.default_rng(n)
    Q = rng.integers(0, bins, (p, n)).astype(np.int64)
    resid = rng.standard_normal(n) + 0.3 * (Q[0] % 3)
    even = np.arange(n) % 2 == 0
    t0, g0 = rescue._score_columns(Q, resid, even, bins, rescue.SHRINK)
    t1, g1 = score_columns_gpu(Q, resid, even, bins, rescue.SHRINK)
    np.testing.assert_allclose(t1, t0, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(g1, g0, rtol=1e-7, atol=1e-10)


def test_column_blocks_give_the_same_scores(monkeypatch):
    """Scoring and coding in narrow column blocks join to the single-block result."""
    require_cuda_device()
    from mlframe.feature_selection.filters import _device_blocks
    from mlframe.feature_selection.filters import _residual_rescue_fe as rescue
    from mlframe.feature_selection.filters._residual_rescue_gpu import codes_matrix_gpu, score_columns_gpu
    from mlframe.feature_selection.filters._y_encoding import FEW_CLASSES_MAX

    M = _matrix(np.float32, 6000, 30, 1)
    whole_c = codes_matrix_gpu(M, 16, rescue.CODE_SAMPLE, rescue.MIN_ROWS, FEW_CLASSES_MAX)
    Q = np.random.default_rng(2).integers(0, 16, (30, 6000)).astype(np.int64)
    resid = np.random.default_rng(3).standard_normal(6000)
    even = np.arange(6000) % 2 == 0
    whole_s = score_columns_gpu(Q, resid, even, 16, rescue.SHRINK)
    monkeypatch.setattr(_device_blocks, "first_block", lambda cp_, n_rows, n_cols, itemsize, multiple: 7)
    blocked_c = codes_matrix_gpu(M, 16, rescue.CODE_SAMPLE, rescue.MIN_ROWS, FEW_CLASSES_MAX)
    blocked_s = score_columns_gpu(Q, resid, even, 16, rescue.SHRINK)
    assert np.array_equal(whole_c[0], blocked_c[0]) and np.array_equal(whole_c[1], blocked_c[1])
    assert np.array_equal(whole_s[0], blocked_s[0]) and np.array_equal(whole_s[1], blocked_s[1])


def test_dispatch_uses_the_device_when_chosen_and_falls_back_on_a_fault(monkeypatch):
    """With the tuned choice forced to the device the device result is returned; a device fault returns the host result."""
    require_cuda_device()
    from mlframe.feature_selection.filters import _residual_rescue_fe as rescue
    from mlframe.feature_selection.filters import _residual_rescue_gpu as gpu
    from mlframe.feature_selection.filters import _residual_rescue_ktc as ktc
    from mlframe.feature_selection.filters._y_encoding import FEW_CLASSES_MAX

    M = _matrix(np.float32, 5000, 20, 5)
    host = rescue._codes_matrix(M, 16, rescue.CODE_SAMPLE, rescue.MIN_ROWS, FEW_CLASSES_MAX)
    monkeypatch.setattr(ktc, "codes_device_chosen", lambda n, p: True)
    calls = []
    real = gpu.codes_matrix_gpu
    monkeypatch.setattr(gpu, "codes_matrix_gpu", lambda *a: (calls.append(1), real(*a))[1])
    dev = ktc.codes_matrix_dispatch(rescue._codes_matrix, M, 16, rescue.CODE_SAMPLE, rescue.MIN_ROWS, FEW_CLASSES_MAX)
    assert calls and np.array_equal(dev[1], host[1]) and np.array_equal(dev[0][host[1]], host[0][host[1]])
    monkeypatch.setattr(gpu, "codes_matrix_gpu", lambda *a: (_ for _ in ()).throw(RuntimeError("device fault")))
    fallback = ktc.codes_matrix_dispatch(rescue._codes_matrix, M, 16, rescue.CODE_SAMPLE, rescue.MIN_ROWS, FEW_CLASSES_MAX)
    assert np.array_equal(fallback[0], host[0]) and np.array_equal(fallback[1], host[1])


def test_the_rescue_admits_the_same_columns_on_either_backend(monkeypatch):
    """The whole rescue admits the same column with the host path and with both device paths forced."""
    require_cuda_device()
    from mlframe.feature_selection.filters import _residual_rescue_fe as rescue
    from mlframe.feature_selection.filters import _residual_rescue_ktc as ktc

    rng = np.random.default_rng(0)
    n = 8000
    design = {f"d{i}": rng.standard_normal(n) for i in range(4)}
    pool = {f"p{i}": rng.standard_normal(n) for i in range(40)}
    pool["hit"] = np.log(np.abs(design["d0"]) + 1.5) * np.sin(design["d1"] * 2)
    y = design["d0"] ** 2 + 2.0 * pool["hit"] + rng.standard_normal(n) * 0.5
    monkeypatch.setattr(ktc, "codes_device_chosen", lambda n_, p_: False)
    monkeypatch.setattr(ktc, "score_device_chosen", lambda n_, p_: False)
    want = rescue.residual_rescue(design, pool, y)
    monkeypatch.setattr(ktc, "codes_device_chosen", lambda n_, p_: True)
    monkeypatch.setattr(ktc, "score_device_chosen", lambda n_, p_: True)
    got = rescue.residual_rescue(design, pool, y)
    assert [g[0] for g in got] == [w[0] for w in want] and want
    np.testing.assert_allclose([g[1] for g in got], [w[1] for w in want], rtol=1e-6)


def test_scores_are_bit_reproducible_between_runs():
    """Two runs of the scoring give the same bits (the device adds the bin sums as integers, not as doubles in thread order)."""
    require_cuda_device()
    from mlframe.feature_selection.filters import _residual_rescue_fe as rescue
    from mlframe.feature_selection.filters._residual_rescue_gpu import score_columns_gpu

    rng = np.random.default_rng(9)
    n, p = 20000, 50
    Q = rng.integers(0, 16, (p, n)).astype(np.int64)
    resid = rng.standard_normal(n) + 0.2 * (Q[1] % 4)
    even = np.arange(n) % 2 == 0
    first = score_columns_gpu(Q, resid, even, 16, rescue.SHRINK)
    for _ in range(4):
        again = score_columns_gpu(Q, resid, even, 16, rescue.SHRINK)
        assert np.array_equal(first[0], again[0]) and np.array_equal(first[1], again[1])
