"""A2-10: predict-path extensions replay must HARD-FAIL on transform error, never silently serve the raw frame."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.core._predict_pre_pipeline import _apply_extensions_pipeline


class _BoomPipeline:
    """Stub sklearn-like pipeline whose .transform always raises."""

    feature_names_in_ = np.array(["a", "b"], dtype=object)

    def transform(self, df):
        raise ValueError("boom: shape mismatch inside extension transform")


def _frame():
    return pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})


def test_a2_10_transform_failure_raises_by_default() -> None:
    """Default (hard-fail): a failing extension transform raises, not silently returns the raw frame."""
    with pytest.raises(RuntimeError, match="transform failed at predict time"):
        _apply_extensions_pipeline(_frame(), _BoomPipeline())


def test_a2_10_soft_fail_escape_hatch_returns_raw(monkeypatch) -> None:
    """The explicit MLFRAME_EXTENSIONS_SOFT_FAIL=1 escape hatch restores the (unsafe) raw-frame fallback."""
    monkeypatch.setenv("MLFRAME_EXTENSIONS_SOFT_FAIL", "1")
    df = _frame()
    out = _apply_extensions_pipeline(df, _BoomPipeline())
    assert out is df, "soft-fail mode returns the raw frame unchanged"
