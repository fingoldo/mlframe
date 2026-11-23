"""Cluster 15 regression: prescreen_top_k=0 means no cap in the univariate prescreen, as it does in the mRMR prescreen (audit 2026-11-20, finding 15-F18).

Before the fix a zero cap kept nothing, so the search was empty. A None or 0 cap now keeps the candidates; a positive cap still limits them.
"""

import numpy as np
import pandas as pd

from mlframe.feature_selection.wrappers.rfecv import _prescreen as ps


class _Cfg:
    """Minimal stand-in carrying the one attribute the prescreen reads."""

    def __init__(self, top_k):
        """Store the cap."""
        self.prescreen_top_k = top_k


def _data():
    """A small binary-target frame with several candidate columns."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 5)), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"] + 0.1 * rng.normal(size=120) > 0).astype(int).to_numpy()
    return X, y


def test_zero_cap_keeps_candidates():
    """prescreen_top_k=0 is no cap, so the candidates are not emptied (15-F18)."""
    X, y = _data()
    kept = ps._prescreen_univariate_ht(_Cfg(0), X, y, list(X.columns))
    assert len(kept) > 0


def test_positive_cap_still_limits():
    """Control: a positive cap keeps at most that many features (15-F18)."""
    X, y = _data()
    kept = ps._prescreen_univariate_ht(_Cfg(2), X, y, list(X.columns))
    assert len(kept) <= 2
