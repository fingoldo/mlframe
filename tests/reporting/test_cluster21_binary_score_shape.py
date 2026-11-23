"""Cluster 21 regression: a binary split takes the positive column of a (n, 2) predict_proba and refuses other matrices (audit 2026-11-20, finding 21-F28).

Raveling an (n, K) matrix against an n-length label vector mis-paired labels and scores. The binary branch now uses column 1 of a 2-column
matrix and raises for any other multi-column shape.
"""

import numpy as np
import pytest

from mlframe.reporting.charts import split_comparison as sc


def test_two_column_proba_uses_the_positive_column():
    """A (n, 2) predict_proba gives the same metrics as its positive column (21-F28)."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    pos = np.clip(y * 0.6 + 0.2 + 0.1 * rng.random(200), 0.0, 1.0)
    proba = np.column_stack([1.0 - pos, pos])
    m_matrix, _ = sc._metrics_for_split("binary", {"y_true": y, "y_score": proba}, 0.5, 10_000, 0)
    m_vector, _ = sc._metrics_for_split("binary", {"y_true": y, "y_score": pos}, 0.5, 10_000, 0)
    assert m_matrix is not None and m_vector is not None
    for key in m_vector:
        assert np.isclose(m_matrix[key], m_vector[key])


def test_other_multi_column_shapes_are_refused():
    """A (n, 3) score matrix is refused with a clear error, not raveled (21-F28)."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=50)
    scores = rng.random((50, 3))
    with pytest.raises(ValueError, match="1-D or \\(n, 2\\)"):
        sc._metrics_for_split("binary", {"y_true": y, "y_score": scores}, 0.5, 10_000, 0)
