"""Audit 2026-11-20 finding 40-F18: with group ids present, a split whose row index does not align is refused, not averaged across entities."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlframe.training.pipeline._ma_crossover_composite_fe import apply_ma_crossover_composite_fe


def test_unaligned_split_with_group_ids_is_refused():
    """Group ids are given but the train index is missing: a ValueError names the split instead of a cross-entity mean."""
    df = pd.DataFrame({"a": np.arange(12.0)})
    cfg = SimpleNamespace(ma_crossover_columns=["a"], ma_crossover_windows=[2, 3], ma_crossover_long_window_weight_power=0.0)
    groups = np.repeat(np.arange(3), 4)
    with pytest.raises(ValueError, match="train split has no row index aligned"):
        apply_ma_crossover_composite_fe(df, None, None, cfg, groups, None, None, None, None)
