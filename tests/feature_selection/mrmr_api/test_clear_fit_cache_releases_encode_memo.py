"""``MRMR.clear_fit_cache`` is the documented retraining-boundary release; it must also drop the memoised target encodings."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import _y_encoding
from mlframe.feature_selection.filters.mrmr import MRMR


def test_clear_fit_cache_drops_memoised_target_encodings():
    """A float target long enough to be memoised is gone from the memo after clear_fit_cache."""
    y = np.random.default_rng(0).normal(size=_y_encoding._ENCODE_MEMO_MIN_N + 10)
    _y_encoding.encode_y_for_classif_mi(y)
    assert len(_y_encoding._encode_memo) >= 1, "precondition: the float target was memoised"
    MRMR.clear_fit_cache()
    assert len(_y_encoding._encode_memo) == 0
