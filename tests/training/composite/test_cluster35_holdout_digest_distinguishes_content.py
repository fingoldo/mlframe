"""Audit 2026-11-20 finding 35-F9: equal-shape external holdouts must get different cache keys; the key must also track the target.

Before the fix, two holdouts with the same shape but different rows shared one OOF cache entry, so the second call was served the
first call's matrix. The holdout digest now covers the frame content.
"""

import numpy as np
import pandas as pd

from mlframe.training.composite.ensemble import _holdout_x_digest


def test_same_shape_different_content_gives_different_digest():
    """Two 10x2 frames with different values hash differently; an identical frame hashes the same."""
    a = pd.DataFrame({"x": np.arange(10.0), "z": np.arange(10.0)})
    b = pd.DataFrame({"x": np.arange(10.0) + 100.0, "z": np.arange(10.0)})
    assert _holdout_x_digest(a) != _holdout_x_digest(b)
    assert _holdout_x_digest(a) == _holdout_x_digest(a.copy())


def test_absent_holdout_digest_is_zero():
    """No external holdout keys as 0, distinct from any present frame's digest."""
    assert _holdout_x_digest(None) == 0
