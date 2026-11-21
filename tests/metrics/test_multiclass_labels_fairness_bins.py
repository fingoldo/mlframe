"""Binary label remap, Series features in fairness subgroups, and excluded rare bins (audit 2026-11-20, findings 25-F5, 25-F6, 25-F7)."""

import numpy as np
import pandas as pd

from mlframe.metrics._fairness_metrics import create_fairness_subgroups
from mlframe.metrics._ice_metric import _infer_multilabel_from_labels


def test_binary_two_column_labels_are_remapped_to_their_columns():
    """Labels {1, 2} with two probability columns map column j to label sorted_unique[j], not to the wrong class."""
    probs = [np.array([0.9, 0.2]), np.array([0.1, 0.8])]
    y_true = np.array([1, 2, 1, 2])
    labels = _infer_multilabel_from_labels(None, False, probs, y_true)
    np.testing.assert_array_equal(labels, [1, 2])


def test_series_feature_does_not_crash_the_special_name_check():
    """A pd.Series feature used to make the ``**ORDER**`` tuple test raise on a Series comparison."""
    df = pd.DataFrame({"g": ["a", "a", "b", "b"] * 300})
    out = create_fairness_subgroups(df, [pd.Series(df["g"].to_numpy(), name="g")], min_pop_cat_thresh=10)
    assert "g" in out


def test_excluded_rare_bins_do_not_become_subgroup_values():
    """A rare bin excluded from the subgroups must be missing in ``bins``, not silently kept."""
    df = pd.DataFrame({"g": ["a"] * 10 + ["b"] * 10 + ["c"]})
    out = create_fairness_subgroups(df, ["g"], min_pop_cat_thresh=3, merge_lowpop_cats=False, exclude_terminal_lowpop_cats=True)
    bins = out["g"]["bins"]
    assert bins.isna().sum() == 1
    assert set(bins.dropna().unique()) == {"a", "b"}
