"""Audit 2026-11-20 finding 40-F15: nearest-past-join refuses a later split that lacks the join columns the train split has."""

from types import SimpleNamespace

import pandas as pd
import pytest

from mlframe.training.pipeline._nearest_past_join_composite_fe import apply_nearest_past_join_composite_fe


def test_split_missing_a_join_column_that_train_has_is_refused():
    """Train has the entity and time columns; val lacks the entity column: the step names val instead of passing it through."""
    cfg = SimpleNamespace(nearest_past_join_on="t", nearest_past_join_by=["k"], nearest_past_join_value_cols=["v"], nearest_past_join_fallback_by_chain=None, nearest_past_join_min_group_size=None)
    train = pd.DataFrame({"k": [1, 1], "t": [1.0, 2.0]})
    val = pd.DataFrame({"t": [3.0]})
    aux = pd.DataFrame({"k": [1], "t": [0.5], "v": [9.0]})
    with pytest.raises(ValueError, match="val split lacks the join columns"):
        apply_nearest_past_join_composite_fe(train, val, None, cfg, aux)
