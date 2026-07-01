"""Unit + biz_value tests for set-similarity coefficients (PZAD err_multirankcluster)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.core.set_similarity import (
    braun_blanquet,
    dice,
    jaccard,
    kulczynski,
    ochiai,
    overlap,
    tversky,
)


# ---------------------------------------------------------------- unit
def test_identical_sets_all_one():
    s = {1, 2, 3}
    for fn in (jaccard, dice, overlap, braun_blanquet, ochiai, kulczynski, tversky):
        assert abs(fn(s, s) - 1.0) < 1e-12


def test_disjoint_sets_all_zero():
    a, b = {1, 2}, {3, 4}
    for fn in (jaccard, dice, overlap, braun_blanquet, ochiai, kulczynski):
        assert fn(a, b) == 0.0


def test_known_values():
    a, b = {1, 2, 3, 4}, {3, 4, 5}  # inter=2, |A|=4, |B|=3
    assert abs(jaccard(a, b) - 2 / 5) < 1e-12
    assert abs(dice(a, b) - 4 / 7) < 1e-12
    assert abs(overlap(a, b) - 2 / 3) < 1e-12  # min=3
    assert abs(braun_blanquet(a, b) - 2 / 4) < 1e-12  # max=4
    assert abs(ochiai(a, b) - 2 / np.sqrt(12)) < 1e-12


def test_boolean_mask_input_matches_set_input():
    am = np.array([True, True, False, True])
    bm = np.array([False, True, True, True])
    # A={0,1,3}, B={1,2,3}: inter=2, union=4
    assert abs(jaccard(am, bm) - 2 / 4) < 1e-12
    assert abs(dice(am, bm) - 4 / 6) < 1e-12


def test_overlap_is_one_when_subset():
    assert overlap({1, 2}, {1, 2, 3, 4}) == 1.0  # A ⊂ B


def test_tversky_reduces_to_jaccard_and_dice():
    a, b = {1, 2, 3}, {2, 3, 4}
    assert abs(tversky(a, b, alpha=1.0, beta=1.0) - jaccard(a, b)) < 1e-12
    assert abs(tversky(a, b, alpha=0.5, beta=0.5) - dice(a, b)) < 1e-12


def test_tversky_asymmetry_and_guard():
    a, b = {1, 2, 3, 4, 5}, {1, 2}  # prediction big, reference small
    fp_heavy = tversky(a, b, alpha=0.9, beta=0.1)  # penalize A-only (false positives) hard
    fn_heavy = tversky(a, b, alpha=0.1, beta=0.9)
    assert fp_heavy < fn_heavy  # A has many extra items -> heavy alpha lowers score more
    with pytest.raises(ValueError):
        tversky(a, b, alpha=-1.0)


def test_both_empty_is_one():
    assert jaccard(set(), set()) == 1.0
    assert dice(set(), set()) == 1.0


def test_mask_shape_mismatch_raises():
    with pytest.raises(ValueError):
        jaccard(np.array([True, False]), np.array([True, False, True]))


# ---------------------------------------------------------------- biz_value
def test_biz_val_overlap_detects_containment_jaccard_misses():
    """When a small predicted span sits fully inside the true span, overlap=1 (perfect coverage of the smaller)
    while Jaccard is penalized by the size gap. The coefficient choice changes the verdict -- the lecture's point."""
    pred = set(range(10, 15))  # 5-wide, fully inside
    true = set(range(0, 100))
    assert overlap(pred, true) == 1.0
    assert jaccard(pred, true) < 0.1  # union dominated by the large true set


def test_biz_val_interval_iou_ranks_better_prediction_higher():
    """Interval-as-set: a predicted interval overlapping the truth more gets a higher Jaccard (IoU)."""
    true = set(range(20, 40))  # [20,40)
    good = set(range(22, 42))  # heavy overlap
    poor = set(range(35, 55))  # small overlap
    assert jaccard(good, true) > jaccard(poor, true) + 0.3
