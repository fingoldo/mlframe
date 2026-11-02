"""ERR refuses labels above max_grade instead of returning a value above 1 (tracker 2026-11-02 wave 17, W17MET-1)."""

import numpy as np
import pytest

from mlframe.metrics import expected_reciprocal_rank


def test_label_above_the_default_max_grade_raises():
    """y_true=[5, 5] with the default max_grade 4 gave R=1.94 and an ERR of 1.94."""
    with pytest.raises(ValueError, match="max_grade"):
        expected_reciprocal_rank(np.array([5.0, 5.0]), np.array([2.0, 1.0]), None, k=10)


def test_explicit_max_grade_makes_the_same_labels_valid_and_bounded():
    """With max_grade=5 the labels are on the declared scale and ERR stays in [0, 1]."""
    got = expected_reciprocal_rank(np.array([5.0, 5.0]), np.array([2.0, 1.0]), None, k=10, max_grade=5)
    assert 0.0 <= got <= 1.0
