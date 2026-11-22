"""Cluster 38 regressions: predict-side row-wise cast does not mutate the caller's frame; composite-FE target scan skips empty types (audit 2026-11-20, findings 38-F5, 38-F6)."""

import numpy as np
import pandas as pd

from mlframe.training.core._phase_fit_pipeline_composite import _composite_fe_supervised_target
from mlframe.training.core._predict_pre_pipeline import _apply_row_wise_extensions


def test_row_wise_extensions_do_not_mutate_callers_bool_columns():
    """Bool columns are cast on a copy; the caller's frame keeps its bool dtype (38-F5)."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "flag": [True, False, True]})
    original_dtype = df["flag"].dtype
    _apply_row_wise_extensions(df, {"summary_stats_enabled": False, "extreme_columns_enabled": False, "columns": ["a", "flag"]})
    assert df["flag"].dtype == original_dtype, "caller's bool column was cast in place"


def test_empty_per_type_dict_does_not_hide_later_target_types():
    """An empty dict for one target type must not stop the scan; the next type's target is recorded (38-F6)."""
    metadata = {}
    arr = np.array([0, 1, 1])
    out = _composite_fe_supervised_target({"regression": {}, "binary": {"y": pd.Series(arr)}}, metadata)
    assert out is not None
    assert metadata["composite_fe_supervised_target"] == {"target_type": "binary", "target_name": "y"}
