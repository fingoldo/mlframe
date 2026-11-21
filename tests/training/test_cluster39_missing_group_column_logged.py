"""Audit 2026-11-20 finding 39-F5: a configured group column that is missing from the frame is reported, not silently dropped."""

import logging

import numpy as np
import pandas as pd

from mlframe.training.extractors._extractors_base import FeaturesAndTargetsExtractor


def test_missing_group_column_is_logged_and_leaves_no_group_ids(caplog):
    """group_field names a column the frame does not have: the grouping is None and a WARNING names the missing column."""
    fte = FeaturesAndTargetsExtractor(group_field="well_id")
    df = pd.DataFrame({"x": np.arange(6.0)})
    with caplog.at_level(logging.WARNING):
        out = fte.transform(df)
    group_ids = out[3]
    assert group_ids is None
    assert any("well_id" in rec.getMessage() and "not a column" in rec.getMessage() for rec in caplog.records)
