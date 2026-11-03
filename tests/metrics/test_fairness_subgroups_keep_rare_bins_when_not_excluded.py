"""create_fairness_subgroups with exclude_terminal_lowpop_cats=False keeps unmergeable rare bins instead of crashing or reusing the previous feature's bins."""

from __future__ import annotations

import pandas as pd

from mlframe.metrics._fairness_metrics import create_fairness_subgroups


def test_unmergeable_rare_categories_are_kept_and_do_not_raise():
    """Counts {a: 5000, b: 10, c: 5}: the rare pair cannot be merged (15 < 1000) and exclusion is off, so all three bins stay."""
    df = pd.DataFrame({"f": ["a"] * 5000 + ["b"] * 10 + ["c"] * 5})
    sub = create_fairness_subgroups(df, ["f"], min_pop_cat_thresh=1000, exclude_terminal_lowpop_cats=False)
    assert sorted(sub["f"]["bins_names"]) == ["a", "b", "c"]


def test_second_feature_does_not_inherit_the_first_features_bins():
    """A later feature in the same call gets its own categories."""
    df = pd.DataFrame({"f": ["a"] * 5000 + ["b"] * 10 + ["c"] * 5, "g": ["x"] * 2500 + ["y"] * 2515})
    sub = create_fairness_subgroups(df, ["g", "f"], min_pop_cat_thresh=1000, exclude_terminal_lowpop_cats=False)
    assert sorted(sub["g"]["bins_names"]) == ["x", "y"] and sorted(sub["f"]["bins_names"]) == ["a", "b", "c"]
