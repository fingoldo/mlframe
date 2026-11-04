"""The operand-floor null permutes the weaker operand wherever it sits, and the DCD member matrix accepts a polars frame."""

from __future__ import annotations

import numpy as np
import polars as pl

from mlframe.feature_selection.filters._dynamic_cluster_discovery._dcd_swap_matrix import _resolve_member_matrix
from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_operand_floor import operand_uplift_null_bar_for_pair


def _bed():
    """A strong operand that equals y's code, a noise operand, and their marginal MIs."""
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 4, n)
    codes = {"strong": y.copy(), "noise": rng.integers(0, 4, n)}
    mi = {"strong": 1.0, "noise": 0.001}
    return codes, mi, y


def test_bar_does_not_depend_on_which_slot_the_strong_operand_is_in():
    """(noise, strong) and (strong, noise) give the same bar; before, the first collapsed to 1.0 because the strong operand was the one permuted."""
    codes, mi, y = _bed()
    args = dict(operand_marginal_mi=lambda k: mi[k], operand_discretized=lambda k: codes[k], codes_y=y, floor=1.0)
    first = operand_uplift_null_bar_for_pair(("strong", "noise"), **args)
    second = operand_uplift_null_bar_for_pair(("noise", "strong"), **args)
    assert first == second


def test_member_matrix_from_a_polars_frame():
    """The numeric columns of a polars frame are returned as the member matrix instead of raising AttributeError on dtype.kind."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 4.0, 5.0], "s": ["x", "y", "z"]})
    out = _resolve_member_matrix(df, [0, 1, 2], ["a", "b", "s"])
    assert out is not None
    M, _members, names = out
    assert names == ["a", "b"] and M.shape == (3, 2)
