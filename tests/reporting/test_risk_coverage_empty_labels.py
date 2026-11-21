"""Risk-coverage with no finite rows reaches the no-data path instead of raising (audit 2026-11-20, finding 21-F1)."""
import numpy as np

from mlframe.reporting.charts.risk_coverage import _label_column_index


def test_empty_label_array_gives_an_empty_index():
    """classes.min() on an empty label array used to raise; the empty index lets the no-data path report the chart."""
    idx = _label_column_index(np.array([], dtype=np.int64), 3)
    assert idx.shape == (0,)


def test_in_range_labels_are_their_own_column():
    """Labels already 0..K-1 map to themselves."""
    assert _label_column_index(np.array([0, 2, 1]), 3).tolist() == [0, 2, 1]
