"""mrmr_audit_2026-07-20 edge_cases.md #147: a GPU-resident kernel handed a CPU (numpy, not cupy)
array by mistake must fail loudly, not silently produce a wrong numeric result via numpy's
duck-typed broadcasting. ``_renumber_joint_gpu`` documents its contract as 'ALREADY-RESIDENT
integer class arrays' -- calling it directly with plain numpy.ndarray inputs (bypassing the normal
dispatch that guarantees a cupy array) must raise immediately rather than silently misbehave."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _renumber_joint_gpu


def test_numpy_array_input_raises_typeerror_immediately():
    """A plain host ndarray passed where a resident cupy array is required must raise TypeError
    (cupy's own dtype/argument-type check), not silently compute a wrong joint numbering."""
    a = np.array([1, 2, 3, 1, 2, 3], dtype=np.int64)
    b = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    with pytest.raises(TypeError):
        _renumber_joint_gpu(a, b)


def test_mixed_numpy_and_cupy_input_raises_on_the_numpy_operand():
    """Even a single numpy operand mixed in with otherwise-genuine cupy operands must raise --
    the failure mode must not depend on ALL operands being wrong, catching a partial-conversion bug."""
    a_dev = cp.asarray([1, 2, 3, 1, 2, 3], dtype=cp.int64)
    b_host = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    with pytest.raises(TypeError):
        _renumber_joint_gpu(a_dev, b_host)


def test_genuine_cupy_input_does_not_raise():
    """Sanity check on the contract: the SAME call with genuinely resident cupy arrays succeeds,
    confirming the TypeError above is specific to the numpy-mistake, not a general kernel bug."""
    a_dev = cp.asarray([1, 2, 3, 1, 2, 3], dtype=cp.int64)
    b_dev = cp.asarray([0, 0, 0, 1, 1, 1], dtype=cp.int64)
    joint, nclasses = _renumber_joint_gpu(a_dev, b_dev)
    assert nclasses == 6
    assert joint.shape == (6,)
