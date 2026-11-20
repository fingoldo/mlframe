"""Audit 2026-11-20 finding 40-F16: the powerset composites are attached to every split or to none."""

import pandas as pd

from mlframe.training.pipeline import _categorical_composite_fe as mod


def test_split_missing_a_source_column_leaves_every_split_unchanged():
    """Train has both source columns, val lacks one: the step is skipped, so train gets no composite columns and val is unchanged."""
    cfg = type("C", (), {"categorical_powerset_concat_max_order": 2})()
    train = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"], "n": [1.0, 2.0]})
    val = pd.DataFrame({"a": ["x"], "n": [3.0]})
    before_train, before_val = list(train.columns), list(val.columns)
    out_train, out_val, _ = mod._attach_powerset_composites(train, val, None, train[["a", "b"]], ["a", "b"], cfg, {}, 0)
    assert list(out_train.columns) == before_train
    assert list(out_val.columns) == before_val
