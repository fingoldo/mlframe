"""Regression (wave6 Low): jmim/bur/relaxmrmr plug-in scorers indexed their njit joint histograms directly with
the input codes (no bounds check), so a -1 NaN-sentinel wrapped to the last bin and an over-range code wrote out
of bounds -- silent histogram corruption. PID already guards this class; the three siblings now do too.
"""

import numpy as np
import pytest

from mlframe.feature_selection.filters._jmim_scorer import jmim_score
from mlframe.feature_selection.filters._bur_term import bur_term
from mlframe.feature_selection.filters._relaxmrmr_3d import relax_mrmr_score


def _valid():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 3, 200).astype(np.int64)
    s = rng.integers(0, 3, 200).astype(np.int64)
    y = rng.integers(0, 2, 200).astype(np.int64)
    return x, [s], y


@pytest.mark.parametrize("scorer", [jmim_score, bur_term, relax_mrmr_score])
def test_scorers_reject_negative_sentinel(scorer):
    x, sel, y = _valid()
    x_bad = x.copy()
    x_bad[0] = -1  # NaN sentinel
    with pytest.raises(ValueError, match="negative"):
        scorer(x_bad, sel, y, 3, [3], 2)


@pytest.mark.parametrize("scorer", [jmim_score, bur_term, relax_mrmr_score])
def test_scorers_reject_out_of_range_code(scorer):
    x, sel, y = _valid()
    x_bad = x.copy()
    x_bad[0] = 5  # >= nbins_x=3
    with pytest.raises(ValueError, match="out of range"):
        scorer(x_bad, sel, y, 3, [3], 2)


@pytest.mark.parametrize("scorer", [jmim_score, bur_term, relax_mrmr_score])
def test_scorers_accept_valid_codes(scorer):
    x, sel, y = _valid()
    val = scorer(x, sel, y, 3, [3], 2)
    assert np.isfinite(val)
