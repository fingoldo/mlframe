"""Regression test for ewma alpha validation (EDGE-P2).

Pre-fix: an out-of-range or NaN alpha was passed straight into the recurrence,
silently diverging / poisoning every output with no error.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.core.ewma import ewma


@pytest.mark.parametrize("bad_alpha", [-0.1, 1.5, float("nan")])
def test_ewma_rejects_out_of_range_alpha(bad_alpha):
    with pytest.raises(ValueError, match="alpha must be in"):
        ewma(np.arange(5, dtype=float), bad_alpha)


def test_ewma_accepts_valid_alpha():
    out = ewma(np.arange(5, dtype=float), 0.5)
    assert out.shape == (5,)
    assert np.isfinite(out).all()
