"""Raw columns kept ONLY because ``redundancy_policy='emit_both'`` re-attached them are named in ``emit_both_added_`` and flagged in ``column_stats_['added_by_emit_both']``."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR


def _frame(n: int = 3000, seed: int = 0):
    """y depends on the product x0*x1 (no marginal effect) plus x2; eight further noise columns."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, 11)), columns=[f"x{i}" for i in range(11)])
    y = (X["x0"] * X["x1"] + X["x2"] + 0.3 * rng.standard_normal(n) > 0).astype(np.int8)
    return X, y


def _fit(policy: str):
    """One MRMR fit with feature engineering and the given redundancy policy."""
    X, y = _frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MRMR(verbose=0, random_seed=0, fe_max_steps=1, redundancy_policy=policy).fit(X, y)


def test_drop_policy_adds_no_column():
    """Under ``drop`` nothing is re-attached, so no column is flagged."""
    m = _fit("drop")
    assert list(m.emit_both_added_) == []
    assert not m.column_stats_["added_by_emit_both"].any()


def test_emit_both_marks_exactly_the_reattached_raw_columns():
    """Every flagged column is a raw selected column; the names in ``emit_both_added_`` and the flags of ``column_stats_`` agree."""
    m = _fit("emit_both")
    added = list(m.emit_both_added_)
    flagged = set(m.column_stats_.loc[m.column_stats_["added_by_emit_both"], "column"])
    assert flagged == set(added)
    selected = {str(c) for c in m.get_feature_names_out()}
    assert set(added) <= selected, "a flagged column must be in the output"
    assert set(added) <= {f"x{i}" for i in range(11)}, "only raw columns are re-attached"
