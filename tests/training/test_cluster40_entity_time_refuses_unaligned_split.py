"""Audit 2026-11-20 finding 40-F15: entity-time FE refuses a split with an unusable index instead of leaving that split without its columns."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlframe.training.pipeline._entity_time_composite_fe import apply_entity_time_composite_fe


def test_unaligned_split_is_refused_naming_the_split():
    """Group ids present and no train index: the step raises, naming the train split, rather than skipping it for one split only."""
    df = pd.DataFrame({"x": np.arange(6.0)})
    cfg = SimpleNamespace(state_duration_columns=[], recency_aggregation_columns=["x"])
    with pytest.raises(ValueError, match="train split has no row index aligned"):
        apply_entity_time_composite_fe(df, None, None, cfg, np.repeat(np.arange(2), 3), None, None, None, None)
