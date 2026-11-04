"""The UAED elbow trim keeps the features of the first greedy rounds, whatever order ``support_`` is in, and retention does not re-grow them."""

from __future__ import annotations

import types

import numpy as np

from mlframe.feature_selection.filters._mrmr_fit_impl._finalise import _apply_uaed_elbow_trim
from mlframe.feature_selection.filters._mrmr_fit_impl._recipe_operand_retention import retain_recipe_operands_final


def _model(support):
    """MRMR-like state: greedy rounds picked columns 7, 5, 1, 3 with a sharp elbow after the second round; ``support_`` is in a different order."""
    gains = np.array([0.5, 0.4, 0.02, 0.01])
    log = tuple({"name": f"f{i}", "gain": float(g), "indices": (i,)} for i, g in zip((7, 5, 1, 3), gains))
    return types.SimpleNamespace(
        uaed_auto_size=True, mrmr_gains_=gains, support_=np.asarray(support, dtype=np.int64), n_features_=4, n_features_in_=8, _predictors_log_=log, _engineered_recipes_=[]
    )


def test_trim_keeps_the_strong_features_when_support_is_not_in_greedy_order():
    """A support ordered [1, 3, 5, 7] used to keep its first two entries (1 and 3, the weakest rounds); the strong rounds' 7 and 5 must stay."""
    m = _model([1, 3, 5, 7])
    _apply_uaed_elbow_trim(m)
    assert sorted(int(v) for v in m.support_) == [5, 7]
    assert m._uaed_trimmed_raw_ == frozenset({1, 3})


def test_trim_in_greedy_order_is_unchanged():
    """With support already in greedy order the identity trim equals the positional one."""
    m = _model([7, 5, 1, 3])
    _apply_uaed_elbow_trim(m)
    assert [int(v) for v in m.support_] == [7, 5]


def test_operand_retention_does_not_regrow_features_the_elbow_trimmed():
    """A recipe over columns 5 and 1 would normally pull column 1 back in; after a UAED trim of column 1 it stays out."""
    rec = types.SimpleNamespace(name="mul(f5,f1)", src_names=("f5", "f1"))
    m = types.SimpleNamespace(
        redundancy_policy="emit_both", _engineered_recipes_=[rec], support_=np.array([7, 5]), feature_names_in_=[f"f{i}" for i in range(8)],
        n_features_=3, _uaed_trimmed_raw_=frozenset({1}),
    )
    assert retain_recipe_operands_final(m) == []
    assert [int(v) for v in m.support_] == [7, 5]
