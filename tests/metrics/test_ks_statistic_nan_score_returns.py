"""ks_statistic returns NaN for a NaN score instead of spinning forever in the tie-group scan (tracker 2026-11-02 wave 8, NAN8-1)."""

import numpy as np
import pytest

from mlframe.metrics.classification._classification_extras import _ks_statistic_numpy, ks_statistic


@pytest.mark.timeout(60)
@pytest.mark.parametrize("fn", [ks_statistic, _ks_statistic_numpy])
def test_ks_with_a_nan_score_is_nan(fn):
    """One NaN among the scores gives NaN on every dispatch path."""
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.4, np.nan, 0.8])
    assert np.isnan(fn(y, s))


@pytest.mark.timeout(60)
def test_ks_clean_scores_unchanged():
    """Perfectly separated scores still give 1.0."""
    assert ks_statistic(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
