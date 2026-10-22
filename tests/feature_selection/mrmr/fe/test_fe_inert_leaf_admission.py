"""A pair candidate with an inert raw leaf is kept out of the pool; a candidate whose both leaves matter is kept."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlframe.feature_selection.filters._mrmr_fe_step._step_inert_operands import _raw_leaves, filter_inert_leaf_block


def _setup(candidate_uses_e: bool, n: int = 6000, seed: int = 0):
    """Binned pool of raw ``c`` and ``e`` plus one candidate column: ``rint(c)``-like (``e`` inert) or a function of both."""
    rng = np.random.default_rng(seed)
    c = rng.integers(0, 8, n)
    e = rng.integers(0, 8, n)
    y = rng.integers(0, 2, n)
    data = np.column_stack([c, e, y]).astype(np.int32)
    cand = (c + e) % 8 if candidate_uses_e else c.copy()
    self = SimpleNamespace(feature_names_in_=np.array(["c", "e"]), random_seed=0)
    cols = ["c", "e", "y"]
    features = [((("c", 0), ("e", 0)), 0)]
    return self, features, cand.astype(np.float64).reshape(-1, 1), ["max(rint(c),neg(e))"], np.array([8]), cols, data


def _identity(values):
    """The candidate values are already codes in this fixture."""
    return values.astype(np.int32)


def test_raw_leaves_reads_the_distinct_raw_names_of_an_engineered_name():
    """Tokens that are raw columns, once each, in order of appearance."""
    assert _raw_leaves("max(rint(c),neg(e))", {"c", "e", "d"}) == ["c", "e"]
    assert _raw_leaves("div(sqr(a),abs(a))", {"a"}) == ["a"]


def test_a_candidate_that_ignores_one_leaf_is_dropped():
    """The candidate equals ``c``: ``e`` adds nothing given ``c``, so the block is emptied."""
    self, features, vals, new_cols, new_nbins, cols, data = _setup(candidate_uses_e=False)
    kept, _vals, kept_cols, _nbins, original = filter_inert_leaf_block(self, features, vals, new_cols, new_nbins, cols, data, _identity, 0)
    assert kept == [] and kept_cols == [] and original == []


def test_a_candidate_that_uses_both_leaves_is_kept():
    """The candidate is ``(c + e) mod 8``: each leaf is informative about it given the other, so nothing is dropped."""
    self, features, vals, new_cols, new_nbins, cols, data = _setup(candidate_uses_e=True)
    kept, _vals, kept_cols, _nbins, _original = filter_inert_leaf_block(self, features, vals, new_cols, new_nbins, cols, data, _identity, 0)
    assert kept == features and kept_cols == new_cols


def test_without_pool_data_the_block_is_returned_unchanged():
    """No data to read the leaves from: the filter does nothing."""
    self, features, vals, new_cols, new_nbins, cols, _data = _setup(candidate_uses_e=False)
    kept, _vals, kept_cols, _nbins, _original = filter_inert_leaf_block(self, features, vals, new_cols, new_nbins, cols, None, _identity, 0)
    assert kept == features and kept_cols == new_cols
