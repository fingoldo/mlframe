"""Cluster 22 regression: non-finite valuations are a ValueError naming the data problem (audit 2026-11-20, finding 22-F34)."""

import numpy as np
import pytest

from mlframe.data_valuation._weights import valuation_sample_weight


def test_nan_valuation_is_a_value_error_not_an_internal_bug():
    """A NaN valuation raises ValueError that blames the input, not the RuntimeError about internal bugs (22-F34)."""
    with pytest.raises(ValueError, match="non-finite"):
        valuation_sample_weight(np.array([0.1, np.nan, 0.3]))


def test_finite_valuations_still_weight():
    """Control: clean valuations return mean-one weights (22-F34)."""
    w = valuation_sample_weight(np.array([0.1, 0.5, 0.9]))
    assert np.isclose(w.mean(), 1.0)
