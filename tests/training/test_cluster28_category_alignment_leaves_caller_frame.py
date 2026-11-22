"""Audit 2026-11-20 finding 28-F46: category alignment never writes into the train frame the caller shares.

The helper used to trust a flag another model had set on the frame and assign in place. It now copies every frame before reassigning.
The test marks the train frame with the old flag, aligns categories with a new one in val, and checks the caller's frame keeps its
original categories.
"""

import pandas as pd

from mlframe.training import _eval_helpers as eh


def test_alignment_does_not_mutate_a_frame_marked_by_an_earlier_model():
    """The caller's train frame keeps its original categories after alignment with a val frame that has new ones."""
    train = pd.DataFrame({"c": pd.Categorical(["a", "b", "a"])})
    train._mlframe_filled = True
    val = pd.DataFrame({"c": pd.Categorical(["a", "z", "a"])})
    before = list(train["c"].cat.categories)
    eh._align_xgb_cat_categories("XGBClassifier", train, val, None)
    assert list(train["c"].cat.categories) == before
