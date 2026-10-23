"""The rescue's backward elimination removes the engineered exact copy of a raw column and keeps the raw column."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlframe.feature_selection.filters._mrmr_fit_impl._residual_rescue_stage import _exact_copies_of_raws, _unsupported_columns
from mlframe.feature_selection.filters._residual_rescue_fe import MIN_ROWS, residual_rescue


def _design(n: int, seed: int = 0):
    """A raw ``a`` that carries a dominant linear term, its exact copy ``a__mod_7`` (a modulo a period above its range), a compound of ``a`` with ``b``, and the target."""
    rng = np.random.default_rng(seed)
    a = rng.random(n).astype(np.float32)
    b = rng.random(n).astype(np.float32) + 0.1
    y = 10.0 * a + a**2 / b + 0.05 * rng.standard_normal(n)
    design = {"a": a, "a__mod_7": np.mod(a.astype(np.float64), 7.0).astype(np.float32), "div(sqr(a),b)": (a**2 / b).astype(np.float32)}
    return design, y, frozenset({"a", "b"})


def _estimator():
    """The attributes the prune reads."""
    return SimpleNamespace(redundancy_policy="drop", fe_residual_rescue_operand_min_linear_gain=0.001, fe_drop_redundant_raw_operands=True)


def test_exact_copies_are_found_by_value_not_by_name():
    """Only an engineered column equal to a raw column element for element is a copy; a transformed column is not."""
    design, _y, raw = _design(max(4000, MIN_ROWS * 2))
    assert _exact_copies_of_raws(design, raw) == ["a__mod_7"]


def test_the_copy_goes_and_the_raw_with_the_private_linear_term_stays():
    """The raw column is not judged against its own copy: the copy is pruned, the raw stays."""
    design, y, raw = _design(max(4000, MIN_ROWS * 2))
    pruned = _unsupported_columns(_estimator(), design, y, raw, residual_rescue)
    assert "a__mod_7" in pruned
    assert "a" not in pruned
