"""The shared target encoder must keep non-integral float labels distinct (mrmr_audit_2026-09-14 RO-4 / IMPL-4).

Nineteen FE sites discretised a float target inline with ``if <distinct count> <= 32: y = y.astype(np.int64)``, treating
"few distinct floats" as "integral labels". That truncates ``{0.0, 0.5, 1.0, 1.5}`` to two classes and collapses a target
of levels inside ``[0, 1)`` to ONE class, so every MI score against it reads ~0 and the family silently emits nothing.
All nineteen now route through ``encode_y_for_classif_mi``; these pin the property they depend on, and each truncation
case asserts the old cast really did fail, so the test cannot pass for a reason unrelated to the fix.
"""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._y_encoding import encode_y_for_classif_mi


@pytest.mark.parametrize(
    "levels",
    [[0.0, 0.5, 1.0, 1.5], [0.1, 0.2, 0.3, 0.4, 0.5], [-2.5, -0.5, 0.5, 2.5]],
    ids=["half-step", "unit-interval", "signed-half-integers"],
)
def test_non_integral_float_levels_stay_distinct(levels):
    """Each distinct label must map to its own code, where the truncating cast merged some of them."""
    y = np.tile(np.asarray(levels, dtype=np.float64), 40)
    assert len(np.unique(y.astype(np.int64))) < len(levels), "fixture must be one the truncating cast actually breaks"
    assert len(np.unique(encode_y_for_classif_mi(y))) == len(levels)


def test_a_unit_interval_target_is_not_collapsed_to_a_single_class():
    """The worst case: every label in [0, 1) truncated to 0, leaving a constant target for the MI scorers."""
    y = np.tile(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 40)
    assert np.unique(y.astype(np.int64)).tolist() == [0], "the old cast produced a constant target here"
    codes = encode_y_for_classif_mi(y)
    assert len(np.unique(codes)) == 5


def test_codes_are_dense_int64_and_preserve_label_order():
    """Codes are 0..K-1 int64 and follow the sorted label order, so a monotone target keeps its ordering."""
    y = np.array([1.5, 0.0, 1.0, 0.5, 1.5, 0.0])
    codes = encode_y_for_classif_mi(y)
    assert codes.dtype == np.int64
    assert codes.tolist() == [3, 0, 2, 1, 3, 0]


@pytest.mark.parametrize("y", [np.tile([1.0, 2.0, 3.0], 30), np.tile(np.array([1, 2, 5], dtype=np.int64), 30)], ids=["integral-float", "sparse-int"])
def test_integral_targets_keep_their_class_count(y):
    """Integral labels were already handled correctly; the encoder only relabels them, never merges or splits."""
    assert len(np.unique(encode_y_for_classif_mi(y))) == len(np.unique(y))


def test_a_continuous_target_gets_the_same_decile_codes_as_before():
    """Above 32 distinct values the old path quantile-binned into deciles; the encoder must reproduce it bit for bit."""
    y = np.random.default_rng(0).normal(size=600)
    old = pd.qcut(y, q=10, labels=False, duplicates="drop").astype(np.int64)
    assert np.array_equal(encode_y_for_classif_mi(y), old)
