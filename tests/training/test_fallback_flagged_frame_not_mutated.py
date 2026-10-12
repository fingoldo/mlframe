"""A frame already carrying ``_mlframe_filled`` must not be written in place by the CatBoost null-fill helpers.

The flag marks a frame that earlier code has already copied. The frame outlives one fit, and the next model in the same target reads
it as its training frame. Writing the ``__MISSING__`` sentinel into that frame leaks it to HGB, which then fails with
"could not convert string to float: '__MISSING__'".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training._training_loop_fallback_helpers import _train_model_with_f_pandas_model_type_name, _train_model_with_f_trainer_boundary_regardless_which
from mlframe.training.cb._cb_pool import _fill_cb_cat_nan_sentinel


def _flagged_frame_with_cat_nan() -> pd.DataFrame:
    """Frame whose categorical column has a NaN and that is already flagged as filled."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "c": pd.Categorical(["a", None, "b"])})
    df._mlframe_filled = True
    return df


def test_text_decategorise_does_not_write_sentinel_into_flagged_shared_frame():
    """The decategorise step for CatBoost text routing leaves a flagged caller frame's category values untouched."""
    df = _flagged_frame_with_cat_nan()
    fit_params = {"text_features": ["c"]}

    _train_model_with_f_trainer_boundary_regardless_which(True, "CatBoostClassifier", df, fit_params)

    assert df["c"].isna().tolist() == [False, True, False]


def test_pandas_nan_fill_does_not_write_sentinel_into_flagged_shared_frame():
    """The categorical NaN fill for CatBoost leaves a flagged caller frame untouched."""
    df = _flagged_frame_with_cat_nan()
    fit_params = {"cat_features": ["c"]}

    _train_model_with_f_pandas_model_type_name(True, "CatBoostClassifier", fit_params, df)

    assert df["c"].isna().tolist() == [False, True, False]


def test_predict_null_guard_does_not_write_sentinel_into_flagged_frame():
    """The predict-time NaN guard leaves a flagged input frame untouched while still filling its own copy."""
    df = _flagged_frame_with_cat_nan()

    class _Model:
        """Stand-in model exposing only the attributes the predict-time guard reads."""

        feature_names_ = ["x", "c"]

        def _get_cat_feature_indices(self):
            """Index of the categorical column in ``feature_names_``."""
            return [1]

    out = _fill_cb_cat_nan_sentinel(_Model(), df)

    assert df["c"].isna().tolist() == [False, True, False]
    assert out["c"].tolist() == ["a", "__MISSING__", "b"]
    assert np.all(out.index == df.index)
