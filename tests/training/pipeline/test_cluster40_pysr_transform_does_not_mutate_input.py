"""Cluster 40 regression: PySRTransformer.transform leaves the caller's frame unchanged (audit 2026-11-20, finding 40-F28)."""

import numpy as np
import pandas as pd

import mlframe.training.pipeline as _pipeline
from mlframe.training.pipeline import PySRTransformer


def test_transform_adds_columns_to_a_copy_not_the_caller_frame(monkeypatch):
    """The PySR-derived column appears in the returned frame but not in the input (40-F28)."""
    monkeypatch.setattr(_pipeline, "pysr_predict_column", lambda model, df, idx: np.full(len(df), 7.0))
    t = PySRTransformer(model=object(), col_to_index={"pysr__f0": 0})
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    out = t.transform(df)
    assert "pysr__f0" in out.columns
    assert "pysr__f0" not in df.columns, "caller's frame was mutated"
