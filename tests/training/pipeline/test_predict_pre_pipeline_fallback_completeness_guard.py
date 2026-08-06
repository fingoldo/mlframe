"""Regression: ``_try_predict_with_pp_fallback``'s pre-pipeline retry must not fire when the
fallback frame is itself missing fit-time features (e.g. FE-extensions-stage engineered columns
like ``row_summary_*``/``row_extreme_*`` that only exist post-pipeline).

Pre-fix, an encoder-dtype-mismatch TypeError on the primary frame always retried on the raw
pre-pipeline fallback frame -- if that fallback ALSO lacked required columns, the retry traded a
clear TypeError for a confusing downstream ``ValueError: columns are missing: {...}`` raised deep
inside sklearn's ColumnTransformer, discovered via ``tests/inference/test_predict_from_models_lgb_hgb_cat.py::test_iter80_lgb_hgb_polars_cat_both_predict_succeed``.
Post-fix: the retry is skipped and the original, informative TypeError propagates instead.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mlframe.training.core._predict_pre_pipeline import _try_predict_with_pp_fallback


def _isnan_typeerror(*_args, **_kwargs):
    """Raise the exact TypeError symptom sklearn's OrdinalEncoder produces on a dtype mismatch."""
    raise TypeError("ufunc 'isnan' not supported for the input types, and the inputs could not be safely coerced to any supported types")


def test_fallback_skipped_when_it_is_missing_fit_time_features(caplog):
    """The pre-pipeline fallback frame lacks an engineered column (row_summary_mean) that the
    model's feature_names_in_ requires -- the retry must be skipped and the original TypeError
    must propagate, not a confusing secondary error from calling fn on an incomplete frame."""
    primary = pd.DataFrame({"a": [1.0, 2.0], "row_summary_mean": [0.1, 0.2]})
    fallback = pd.DataFrame({"a": ["x", "y"]})  # raw pre-pipeline frame: missing row_summary_mean

    with pytest.raises(TypeError, match="isnan"):
        _try_predict_with_pp_fallback(
            _isnan_typeerror,
            primary,
            fallback,
            model=None,
            expected_list=["a", "row_summary_mean"],
            pandas_view_cache={},
            model_name="test_model",
        )
    assert any("missing" in r.message and "row_summary_mean" in r.message for r in caplog.records), "expected a warning naming the missing fallback column(s)"


def test_fallback_still_used_when_it_is_complete():
    """When the fallback frame DOES carry every expected feature, the retry still fires and
    succeeds -- the completeness guard must not regress the working case."""
    calls = []

    def _fn(df):
        """Record the frame it was called with and return a stand-in prediction array."""
        calls.append(df)
        if list(df.columns) != ["a", "b"] or df["a"].dtype == object:
            raise TypeError("ufunc 'isnan' not supported for the input types")
        return df["a"].to_numpy() + df["b"].to_numpy()

    primary = pd.DataFrame({"a": ["x", "y"], "b": [1.0, 2.0]})  # wrong dtype on 'a' -> triggers fallback
    fallback = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]})  # complete + right dtype

    out = _try_predict_with_pp_fallback(
        _fn,
        primary,
        fallback,
        model=None,
        expected_list=["a", "b"],
        pandas_view_cache={},
        model_name="test_model",
    )
    assert list(out) == [2.0, 4.0]
    assert len(calls) == 2, "expected one failed primary attempt + one successful fallback attempt"
