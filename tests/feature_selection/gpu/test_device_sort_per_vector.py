"""Long 2-D vectors are sorted one 1-D sort per vector (3.5 to 5 times faster than cupy's batched 2-D sort at 300000 rows); the result is the batched result, ties included."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _device_sort_safe as S


def _device_ok() -> bool:
    """Whether a CUDA device answers."""
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # no driver: skipped
        return False


@pytest.mark.skipif(not _device_ok(), reason="no CUDA device")
@pytest.mark.parametrize("axis", [0, 1, -1, -2])
@pytest.mark.parametrize("ties", [False, True])
def test_per_vector_sort_equals_the_batched_sort(monkeypatch, axis, ties):
    """argsort and sort along every axis give the batched result, with and without heavy ties."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((900, 7)).astype(np.float32)
    if ties:
        x = np.round(x, 1)
    a = cp.asarray(x if axis in (0, -2) else np.ascontiguousarray(x.T))
    want_arg, want_sort = cp.argsort(a, axis=axis), cp.sort(a, axis=axis)
    monkeypatch.setattr(S, "_PER_VECTOR_SORT_MIN_LEN", 100)
    assert S._vectors_of(a, axis) is not None
    assert cp.array_equal(S.device_argsort(a, axis=axis), want_arg)
    assert cp.array_equal(S.device_sort(a, axis=axis), want_sort)


@pytest.mark.skipif(not _device_ok(), reason="no CUDA device")
def test_short_vectors_and_kwargs_keep_the_batched_call():
    """Below the length floor, for a 1-D array and with extra arguments the batched cupy call runs."""
    a = cp.asarray(np.random.default_rng(3).standard_normal((50, 4)).astype(np.float32))
    assert S._vectors_of(a, 0) is None
    assert S._vectors_of(cp.arange(10), 0) is None
    assert cp.array_equal(S.device_argsort(a, axis=0, kind="stable"), cp.argsort(a, axis=0, kind="stable"))
