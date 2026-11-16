"""Merging rare fairness categories leaves the caller's column untouched and yields one bin label per row."""

import pandas as pd

from mlframe.metrics._fairness_metrics import create_fairness_subgroups


def test_merge_lowpop_cats_does_not_modify_the_input_column():
    """Rare categories 'b' and 'z' trigger the merge path; the input DataFrame keeps its values and the bins cover every row."""
    df = pd.DataFrame({"c": ["a"] * 30 + ["b"] * 3 + ["z"] * 2, "x": range(35)})
    before = df["c"].tolist()
    result = create_fairness_subgroups(df, ["c"], min_pop_cat_thresh=10, merge_lowpop_cats=True, exclude_terminal_lowpop_cats=False)
    assert len(result["c"]["bins"]) == len(df)
    assert df["c"].tolist() == before
