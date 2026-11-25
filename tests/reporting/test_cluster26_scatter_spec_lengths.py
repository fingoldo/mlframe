"""Cluster 26 regression: ScatterPanelSpec refuses per-point arrays that do not match the points (audit 2026-11-20, finding 26-F27)."""

import numpy as np
import pytest

from mlframe.reporting.spec import ScatterPanelSpec


def test_mismatched_point_size_is_refused():
    """A point_size array of the wrong length fails at construction, not at render time (26-F27)."""
    with pytest.raises(ValueError, match="point_size"):
        ScatterPanelSpec(x=np.arange(5.0), y=np.arange(5.0), point_size=np.ones(3))


def test_mismatched_y_is_refused():
    """y and x must have the same number of points (26-F27)."""
    with pytest.raises(ValueError, match="y has"):
        ScatterPanelSpec(x=np.arange(5.0), y=np.arange(4.0))


def test_matching_arrays_are_accepted():
    """Control: matching per-point arrays construct (26-F27)."""
    spec = ScatterPanelSpec(x=np.arange(5.0), y=np.arange(5.0), point_size=np.ones(5), point_color=np.ones(5))
    assert len(spec.x) == 5
