"""Regression test for `04_test_coverage.md` gap #3: no test asserted ``get_support()``'s full
boolean-mask CONTENT (element-wise) against a hand-computed reference -- only a count check
existed (``get_support().sum() == len(support_)``), which cannot catch a mask with the right
COUNT of True values at the WRONG positions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR


def test_get_support_mask_matches_support_positions_exactly():
    """get_support() must be True at EXACTLY the positions in support_, False elsewhere --
    element-wise, not just a count match."""
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(
        {
            "signal": rng.standard_normal(n),
            "noise1": rng.standard_normal(n),
            "noise2": rng.standard_normal(n),
            "noise3": rng.standard_normal(n),
        }
    )
    y = (X["signal"] > 0).astype(int)
    m = MRMR(full_npermutations=2, baseline_npermutations=2, verbose=0, fe_max_steps=0)
    m.fit(X, y)

    mask = m.get_support()
    expected = np.zeros(m.n_features_in_, dtype=bool)
    expected[np.asarray(m.support_, dtype=np.intp)] = True

    assert mask.dtype == bool
    assert mask.shape == (m.n_features_in_,)
    np.testing.assert_array_equal(mask, expected)
    # The signal column itself must be True in the mask (sanity: this isn't a trivially all-False fit).
    signal_pos = list(X.columns).index("signal")
    assert mask[signal_pos], f"expected signal column at position {signal_pos} to be selected; mask={mask}"


def test_get_support_indices_mode_matches_boolean_mode():
    """get_support(indices=True) must return exactly np.where(get_support())[0]."""
    rng = np.random.default_rng(1)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"] > 0).astype(int)
    m = MRMR(full_npermutations=2, baseline_npermutations=2, verbose=0, fe_max_steps=0)
    m.fit(X, y)

    mask = m.get_support(indices=False)
    idx = m.get_support(indices=True)
    np.testing.assert_array_equal(idx, np.where(mask)[0])
