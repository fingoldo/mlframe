"""The row argmax: the numba host twin, the CUDA kernel and the per-host choice between them agree on the first-maximum index."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _row_argmax_gpu as ra


def _columns(n: int, m: int, dtype, ties: bool) -> list:
    """``m`` normal columns of ``n`` rows; with ``ties`` rounded to one decimal so many rows have equal maxima."""
    rng = np.random.default_rng(3)
    cols = [rng.normal(size=n).astype(dtype) for _ in range(m)]
    if ties:
        cols = [np.round(c, 1) for c in cols]  # many equal maxima: the first one must win
    return cols


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ties", [False, True])
def test_host_twin_matches_numpy_first_maximum(dtype, ties):
    """The numba kernel returns ``np.argmax`` of the stacked columns (first maximum on ties) as a float64 vector."""
    cols = _columns(5000, 4, dtype, ties)
    got = ra.row_argmax_host(cols)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, np.argmax(np.stack(cols, axis=1), axis=1))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ties", [False, True])
def test_device_kernel_matches_host_twin(dtype, ties):
    """The CUDA kernel and the host twin give the same index vector."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    cols = _columns(7001, 3, dtype, ties)
    got = cp.asnumpy(ra.row_argmax_cm(cp, [cp.asarray(c) for c in cols]))
    np.testing.assert_array_equal(got, ra.row_argmax_host(cols))


def test_choice_is_forced_by_the_environment(monkeypatch):
    """``MLFRAME_ROW_ARGMAX_DEVICE`` forces either side, whatever the tuning cache says."""
    monkeypatch.setenv("MLFRAME_ROW_ARGMAX_DEVICE", "0")
    assert ra.row_argmax_on_device(10_000_000) is False
    monkeypatch.setenv("MLFRAME_ROW_ARGMAX_DEVICE", "1")
    assert ra.row_argmax_on_device(10) is True


def test_gate_host_path_uses_the_twin_and_keeps_nan_semantics():
    """``_argmax_host`` takes the numba twin for finite columns and ``np.argmax`` (first NaN wins) otherwise."""
    from mlframe.feature_selection.filters._conditional_gate_fe import _argmax_host

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(_argmax_host([a, b]), [1.0, 0.0, 0.0])
    c = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    np.testing.assert_array_equal(_argmax_host([c, b]), np.argmax(np.stack([c, b], axis=1), axis=1))
