"""Freeze + replay fidelity for the cat x cat / cat x cat x cat synergy-cross families.

``cat_pair_cross`` / ``cat_triple_cross`` map each row's value-tuple to a dense integer cell
code frozen at fit (``encoding='raw'``) or to the per-cell smoothed mean-of-y
(``encoding='target'``). Replay reads only X. Contracts pinned:

* RAW replay reproduces the frozen cell codes for seen tuples; an UNSEEN tuple maps to the
  sentinel code ``len(mapping)`` (a fresh bin distinct from every seen cell), never a crash.
* TARGET replay emits the frozen per-cell mean-of-y; an unseen tuple (or a seen code absent
  from the te_lookup) falls back to the frozen ``global_mean`` -- and y is never read at
  replay, so the output is invariant to any y in scope.
* Pickle round-trip + frozen-extra immutability + column-order invariance.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.engineered_recipes import (
    apply_recipe,
    build_cat_pair_cross_recipe,
    build_cat_triple_cross_recipe,
)

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def pair_setup():
    # Three seen (i, j) cells -> dense codes 0,1,2.
    mapping = {("a", "x"): 0, ("a", "y"): 1, ("b", "x"): 2}
    te_lookup = {0: 0.10, 1: 0.90, 2: 0.50}
    return mapping, te_lookup


def test_cat_pair_raw_replay_codes(pair_setup):
    mapping, _ = pair_setup
    rec = build_cat_pair_cross_recipe(
        name="cross(i,j)",
        cat_i="i",
        cat_j="j",
        mapping=mapping,
        encoding="raw",
    )
    X = pd.DataFrame({"i": ["a", "a", "b"], "j": ["x", "y", "x"]})
    out = apply_recipe(rec, X)
    np.testing.assert_array_equal(out, [0.0, 1.0, 2.0])


def test_cat_pair_raw_unseen_tuple_maps_to_sentinel(pair_setup):
    mapping, _ = pair_setup
    rec = build_cat_pair_cross_recipe(
        name="cross(i,j)",
        cat_i="i",
        cat_j="j",
        mapping=mapping,
        encoding="raw",
    )
    # ("b","y") was never a fit cell -> sentinel = len(mapping) = 3.
    X = pd.DataFrame({"i": ["b", "a"], "j": ["y", "x"]})
    out = apply_recipe(rec, X)
    assert out[0] == 3.0  # sentinel
    assert out[1] == 0.0  # seen cell


def test_cat_pair_target_replay_and_unseen_global_mean(pair_setup):
    mapping, te_lookup = pair_setup
    gmean = 0.42
    rec = build_cat_pair_cross_recipe(
        name="cross(i,j)",
        cat_i="i",
        cat_j="j",
        mapping=mapping,
        encoding="target",
        te_lookup=te_lookup,
        global_mean=gmean,
    )
    X = pd.DataFrame({"i": ["a", "b", "z"], "j": ["y", "x", "z"]})
    out = apply_recipe(rec, X)
    assert out[0] == pytest.approx(0.90)  # ("a","y") -> code 1 -> 0.90
    assert out[1] == pytest.approx(0.50)  # ("b","x") -> code 2 -> 0.50
    assert out[2] == pytest.approx(gmean)  # unseen -> global mean
    assert np.isfinite(out).all()


def test_cat_pair_target_replay_ignores_y_in_scope(pair_setup):
    mapping, te_lookup = pair_setup
    rec = build_cat_pair_cross_recipe(
        name="cross(i,j)",
        cat_i="i",
        cat_j="j",
        mapping=mapping,
        encoding="target",
        te_lookup=te_lookup,
        global_mean=0.42,
    )
    X = pd.DataFrame({"i": ["a", "b"], "j": ["x", "x"]})
    a = apply_recipe(rec, X)
    _ = np.array([1.0, 0.0])  # y in scope
    b = apply_recipe(rec, X)
    np.testing.assert_array_equal(a, b)


def test_cat_triple_raw_replay_and_sentinel():
    mapping = {("a", "x", "p"): 0, ("b", "y", "q"): 1}
    rec = build_cat_triple_cross_recipe(
        name="cross3",
        cat_a="A",
        cat_b="B",
        cat_c="C",
        mapping=mapping,
        encoding="raw",
    )
    X = pd.DataFrame(
        {
            "A": ["a", "b", "a"],
            "B": ["x", "y", "x"],
            "C": ["p", "q", "ZZZ"],  # last row unseen triple
        }
    )
    out = apply_recipe(rec, X)
    np.testing.assert_array_equal(out[:2], [0.0, 1.0])
    assert out[2] == float(len(mapping))  # sentinel


def test_cat_cross_pickle_roundtrip_and_frozen_extra(pair_setup):
    mapping, te_lookup = pair_setup
    rec = build_cat_pair_cross_recipe(
        name="cross(i,j)",
        cat_i="i",
        cat_j="j",
        mapping=mapping,
        encoding="target",
        te_lookup=te_lookup,
        global_mean=0.42,
    )
    rec2 = pickle.loads(pickle.dumps(rec))
    assert rec2 == rec
    X = pd.DataFrame({"i": ["a", "b"], "j": ["x", "x"]})
    np.testing.assert_array_equal(apply_recipe(rec, X), apply_recipe(rec2, X))
    with pytest.raises(TypeError):
        rec.extra["mapping"] = []  # frozen MappingProxy


def test_cat_pair_column_order_invariant(pair_setup):
    mapping, _ = pair_setup
    rec = build_cat_pair_cross_recipe(
        name="cross(i,j)",
        cat_i="i",
        cat_j="j",
        mapping=mapping,
        encoding="raw",
    )
    X = pd.DataFrame({"i": ["a", "b"], "j": ["x", "x"], "extra": [1, 2]})
    Xrev = X[["extra", "j", "i"]]
    np.testing.assert_array_equal(apply_recipe(rec, X), apply_recipe(rec, Xrev))
