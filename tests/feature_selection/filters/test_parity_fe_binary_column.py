"""A binary column's raw MI is measured on its two levels, so a parity feature that is a copy of it adds no uplift and is not appended."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._parity_fe import _raw_mi, hybrid_parity_fe_with_recipes


def _binary_frame(n=6000, seed=0):
    """A binary signal column, a noise column and a binary target the signal drives."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, n).astype(float)
    y = (rng.random(n) < np.where(a == 1, 0.8, 0.2)).astype(np.int64)
    return pd.DataFrame({"A": a, "noise": rng.standard_normal(n)}), y


def test_raw_mi_of_a_binary_signal_column_is_positive():
    """The two levels are not merged into one code."""
    X, y = _binary_frame()
    assert _raw_mi(X["A"].to_numpy(), y) > 0.1


def test_no_parity_copy_of_a_binary_column_is_appended():
    """The parity grid contains periods whose square wave equals the 0/1 column; it brings no uplift over the column itself."""
    X, y = _binary_frame()
    _, appended, _, scores = hybrid_parity_fe_with_recipes(X, y)
    assert not [nm for nm in appended if nm.startswith("A__")], f"parity copy of the binary column appended: {appended}; scores:\n{scores}"
