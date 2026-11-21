"""Audit 2026-11-20 finding 28-F45: decategorising text columns does not change the eval-set frame the caller keeps.

The helper wrote object columns into the caller's frame, so the next model reusing it saw demoted text columns and empty strings where
there had been NaN. The test checks the caller's categorical column is unchanged, and the returned frame has the object column.
"""

import pandas as pd

from mlframe.training._training_loop_fallback_helpers import _decategorize_text_cols


def test_caller_frame_keeps_its_categorical_text_column():
    """The returned frame is object with NaN filled by an empty string; the caller's frame is untouched."""
    df = pd.DataFrame({"t": pd.Categorical(["hello", None, "world"])})
    out = _decategorize_text_cols(df, ["t"])
    assert isinstance(df["t"].dtype, pd.CategoricalDtype)
    assert out["t"].dtype == object
    assert out["t"].tolist() == ["hello", "", "world"]
    assert df["t"].isna().sum() == 1
