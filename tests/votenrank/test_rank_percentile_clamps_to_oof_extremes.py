"""Test values beyond the OOF range clamp to the OOF extreme percentiles, not to 0 and 1.

The OOF percentiles run from 0.5/n to (n-0.5)/n. A test value above every OOF value used to get 1.0 and one below every OOF value 0.0, so the
test scale had a gap that the meta-learner never saw during training.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.votenrank.rank_percentile_stacking import rank_percentile_transform


def test_values_beyond_the_oof_range_clamp_to_the_oof_extremes():
    """With OOF [1, 2] (n=2) the extremes are 0.25 and 0.75; out-of-range test values take them."""
    _, test_pct = rank_percentile_transform(np.array([1.0, 2.0]), np.array([0.5, 2.0001, 0.9999]))
    assert test_pct[0] == pytest.approx(0.25)
    assert test_pct[1] == pytest.approx(0.75)
    assert test_pct[2] == pytest.approx(0.25)


def test_in_range_values_keep_their_oof_percentile():
    """A test value equal to an OOF value gets that value's OOF percentile, as before."""
    oof = np.arange(1.0, 11.0)
    oof_pct, test_pct = rank_percentile_transform(oof, oof.copy())
    np.testing.assert_allclose(test_pct, oof_pct)
