"""The 2-variable joint kernels refuse codes outside the declared bin counts instead of writing out of bounds."""
import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory._class_encoding import joint_entropy_2var, joint_freqs_2var


@pytest.mark.parametrize("kernel", [joint_freqs_2var, joint_entropy_2var])
@pytest.mark.parametrize("bad", [-1, 4])
def test_joint_2var_out_of_range_code_raises(kernel, bad):
    """A code below 0 or at/above nb must raise ValueError."""
    data = np.zeros((10, 2), dtype=np.int32)
    data[3, 0] = bad
    with pytest.raises(ValueError):
        kernel(data, 0, 1, 4, 4)


def test_joint_2var_in_range_unchanged():
    """In-range codes keep returning a normalised histogram."""
    data = np.array([[0, 1], [1, 0], [1, 1], [0, 1]], dtype=np.int32)
    assert abs(joint_freqs_2var(data, 0, 1, 2, 2).sum() - 1.0) < 1e-12
    assert joint_entropy_2var(data, 0, 1, 2, 2) > 0.0
