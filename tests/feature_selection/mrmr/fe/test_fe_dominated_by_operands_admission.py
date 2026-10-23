"""A pair candidate that adds nothing about the target beyond its other raw leaves is kept out of the pool; a candidate that needs both leaves stays."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlframe.feature_selection.filters._mrmr_fe_step._step_inert_operands import filter_inert_leaf_block


def _identity(values):
    """The candidate values are already codes in this fixture."""
    return values.astype(np.int32)


def _setup(kind: str, n: int = 6000, seed: int = 0):
    """Raw leaves ``c`` (informative) and ``e`` (noise) plus a candidate column and the target. ``degraded``: the target depends on ``c`` only and the candidate is ``c`` made
    noisy by ``e`` on half of the rows (less information than ``c``, nothing beyond it). ``genuine``: the target is the parity of both leaves and the candidate is their sum modulo 8."""
    rng = np.random.default_rng(seed)
    c = rng.integers(0, 8, n)
    e = rng.integers(0, 8, n)
    if kind == "degraded":
        y = (c >= 4).astype(np.int64)
        cand = np.where(rng.random(n) < 0.5, c, e)
    else:
        y = ((c + e) % 2).astype(np.int64)
        cand = (c + e) % 8
    data = np.column_stack([c, e, y]).astype(np.int32)
    me = SimpleNamespace(feature_names_in_=np.array(["c", "e"]), random_seed=0)
    features = [((("c", 0), ("e", 0)), 0)]
    return me, features, cand.astype(np.float64).reshape(-1, 1), ["f(c,e)"], np.array([8]), ["c", "e", "y"], data, y


def test_a_degraded_copy_of_one_operand_is_dropped_and_recorded():
    """The candidate adds no information about the target beyond ``c``: the block is emptied and the drop is in the ledger with its numbers."""
    me, features, vals, new_cols, nbins, cols, data, y = _setup("degraded")
    kept, _v, kept_cols, _nb, _orig = filter_inert_leaf_block(me, features, vals, new_cols, nbins, cols, data, _identity, 0, y)
    assert kept == [] and kept_cols == []
    rows = [r for r in me._fe_rejection_records_ if r["gate"] == "dominated_by_operands"]
    assert len(rows) == 1 and rows[0]["candidate"] == "f(c,e)" and rows[0]["observed"] <= rows[0]["threshold"]


def test_a_candidate_that_needs_both_leaves_is_kept():
    """With a target that is the parity of both leaves the candidate adds information beyond each leaf alone, so nothing is dropped."""
    me, features, vals, new_cols, nbins, cols, data, y = _setup("genuine")
    kept, _v, kept_cols, _nb, _orig = filter_inert_leaf_block(me, features, vals, new_cols, nbins, cols, data, _identity, 0, y)
    assert kept == features and kept_cols == new_cols


def test_without_target_codes_the_test_is_not_run():
    """No target given: only the inert-leaf test applies, so the degraded copy is not dropped by this filter."""
    me, features, vals, new_cols, nbins, cols, data, _y = _setup("degraded")
    kept, _v, _kept_cols, _nb, _orig = filter_inert_leaf_block(me, features, vals, new_cols, nbins, cols, data, _identity, 0, None)
    assert kept == features
