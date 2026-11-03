"""PELT is shift invariant on far-from-zero levels, and the OOF category encoding is reproducible by default."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.preprocessing.category_support import smoothed_target_encode_column
from mlframe.signal._pelt_l2_njit import pelt_l2


def test_pelt_breakpoints_do_not_depend_on_the_signal_level():
    """A step at 1000 of n=2000 found at a level of 0 must be found identically at a level of 1e9 (raw prefix sums cancelled there)."""
    rng = np.random.default_rng(0)
    y = np.r_[rng.normal(0.0, 1.0, 1000), rng.normal(3.0, 1.0, 1000)]
    base = pelt_l2(y, 5, 3.0)
    shifted = pelt_l2(y + 1e9, 5, 3.0)
    assert list(base) == list(shifted)


def test_oof_encoding_is_reproducible_with_default_arguments():
    """Two identical calls give identical train encodings (the default seed used to be None, i.e. a fresh shuffle each call)."""
    rng = np.random.default_rng(1)
    tr = pd.Series(rng.choice(list("abcd"), 200))
    y = pd.Series(rng.random(200))
    te = pd.Series(list("abcd"))
    a, _ = smoothed_target_encode_column(tr, te, y)
    b, _ = smoothed_target_encode_column(tr, te, y)
    pd.testing.assert_series_equal(a, b)
