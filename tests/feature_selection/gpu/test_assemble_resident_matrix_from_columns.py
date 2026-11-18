"""``assemble_resident_matrix`` builds the same device matrix from a list of host columns as from the stacked host matrix, and shares the operands."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _fe_resident_operands as ro


def test_columns_route_equals_the_matrix_route_and_reuses_the_operands(monkeypatch):
    """Four float32 columns (one strided): both routes give the same ``(n, 4)`` float32 device matrix, and the second route uploads nothing (every column is already resident)."""
    monkeypatch.setenv("MLFRAME_FE_RESIDENT_OPERANDS", "1")
    rng = np.random.default_rng(0)
    base = rng.standard_normal((3000, 6)).astype(np.float32)
    cols = [np.ascontiguousarray(base[:, j]) for j in range(3)] + [base[:, 4]]
    names = ["a", "b", "c", "d"]
    from_matrix = ro.assemble_resident_matrix(np.column_stack(cols), names, ("t", "m"), dtype=np.float32)
    uploads = []
    real = cp.asarray

    def spy(a, *args, **kwargs):
        """Count host arrays handed to the device."""
        if isinstance(a, np.ndarray):
            uploads.append(a.shape)
        return real(a, *args, **kwargs)

    monkeypatch.setattr(cp, "asarray", spy)
    from_columns = ro.assemble_resident_matrix(None, names, ("t", "c"), dtype=np.float32, columns=cols)
    assert uploads == []
    assert from_columns.shape == (3000, 4) and from_columns.dtype == np.float32
    assert bool((from_columns == from_matrix).all())
    np.testing.assert_array_equal(cp.asnumpy(from_columns), np.column_stack(cols))


def test_columns_of_another_dtype_are_cast_to_the_requested_one():
    """Float64 columns asked for as float32 give the float32 matrix of the cast values."""
    rng = np.random.default_rng(1)
    cols = [rng.standard_normal(500) for _ in range(2)]
    got = ro.assemble_resident_matrix(None, ["p", "q"], ("t", "d"), dtype=np.float32, columns=cols)
    np.testing.assert_array_equal(cp.asnumpy(got), np.column_stack(cols).astype(np.float32))
