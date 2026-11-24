"""Cluster 08 regression: the pairwise-median helper has no unused chunk parameter (audit 2026-11-20, finding 08-F17).

``_median_pairwise_chunked`` accepted a ``chunk`` argument it never used. The parameter and its caller argument are removed; the median is unchanged.
"""

import inspect

import numpy as np

from mlframe.feature_engineering.transformer import _utils


def test_chunk_parameter_removed():
    """The helper takes only the sample (08-F17)."""
    assert list(inspect.signature(_utils._median_pairwise_chunked).parameters) == ["X_sub"]


def test_median_value_unchanged():
    """Control: the median of pairwise distances is still computed (08-F17)."""
    X = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    assert np.isclose(_utils._median_pairwise_chunked(X), 5.0)  # distances 5, 10, 5 -> median 5
