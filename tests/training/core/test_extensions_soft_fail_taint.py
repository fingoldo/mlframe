"""A predict served on RAW columns under MLFRAME_EXTENSIONS_SOFT_FAIL carries that taint in its result."""

import numpy as np
import pandas as pd
import pytest

from mlframe.training.core import _predict_pre_pipeline as pp


class _FailingPipeline:
    feature_names_in_ = np.array(["a"])

    def transform(self, X):
        raise RuntimeError("vocabulary drift")


def test_without_the_env_var_the_failure_is_an_outage(monkeypatch):
    monkeypatch.delenv("MLFRAME_EXTENSIONS_SOFT_FAIL", raising=False)
    pp.take_extensions_soft_fail_taint()
    with pytest.raises(RuntimeError, match="transform failed at predict time"):
        pp._apply_extensions_pipeline(pd.DataFrame({"a": [1.0]}), _FailingPipeline(), verbose=0)
    assert pp.take_extensions_soft_fail_taint() == []


def test_with_the_env_var_the_raw_frame_is_served_and_recorded(monkeypatch):
    monkeypatch.setenv("MLFRAME_EXTENSIONS_SOFT_FAIL", "1")
    pp.take_extensions_soft_fail_taint()
    df = pd.DataFrame({"a": [1.0]})
    out = pp._apply_extensions_pipeline(df, _FailingPipeline(), verbose=0)
    assert out is df
    taint = pp.take_extensions_soft_fail_taint()
    assert len(taint) == 1 and "vocabulary drift" in taint[0]
    assert pp.take_extensions_soft_fail_taint() == [], "the taint is handed over once, so the next predict starts clean"
