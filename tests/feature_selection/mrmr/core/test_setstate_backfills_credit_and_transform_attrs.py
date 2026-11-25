"""An estimator pickled before the unary credit, the target-transform search and the emit_both record existed gets their fitted attributes back from ``__setstate__``."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR

_ATTRS = ("_unary_credit_done_key_", "_unary_credit_fired_", "_target_transform_cache_", "emit_both_added_")


def _fitted():
    """A small fitted selector."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(1500, 5)), columns=list("abcde"))
    y = pd.Series(2.0 * X["a"] + 0.2 * rng.normal(size=1500))
    return MRMR(verbose=0, random_state=0, n_jobs=1, fe_max_steps=0).fit(X, y), X, y


def test_the_legacy_state_without_the_four_attributes_gets_their_fit_defaults_and_refits():
    """The state of an old release has none of the four: they come back as a fit would have set them, and the estimator fits again."""
    sel, X, y = _fitted()
    state = dict(sel.__getstate__())
    for name in _ATTRS:
        state.pop(name, None)
    legacy = MRMR.__new__(MRMR)
    legacy.__setstate__(copy.deepcopy(state))
    assert legacy._unary_credit_done_key_ is None
    assert legacy._unary_credit_fired_ is False
    assert legacy._target_transform_cache_ is None
    assert legacy.emit_both_added_ == []
    legacy.emit_both_added_.append("x")
    other = MRMR.__new__(MRMR)
    other.__setstate__(copy.deepcopy(state))  # each unpickled estimator has a state of its own
    assert other.emit_both_added_ == [], "the default list must not be shared between unpickled estimators"
    MRMR.clear_fit_cache()
    assert list(legacy.fit(X, y).get_feature_names_out())
