"""Sensor tests for audit B-P0-2 / B-P0-3 / Low-B11: apply_polars_categorical_fixes
and _precast_strings must use pl.Enum, never pl.Categorical, to avoid widening the
process-wide string cache across suite calls.

Memory rule: ``pl.Categorical`` participates in a global string cache that grows
monotonically across calls and cannot be reset (``pl.disable_string_cache()`` is a
no-op in polars 1.x). ``pl.Enum`` is per-Series.
"""
from __future__ import annotations

import polars as pl
import pytest


@pytest.mark.fast
def test_apply_polars_categorical_fixes_does_not_use_pl_categorical_for_utf8():
    """Step 4 of apply_polars_categorical_fixes must NOT emit pl.Categorical for raw
    Utf8 cat_features when an Enum domain is available from Step 2.
    """
    from mlframe.training.core._phase_polars_fixes import apply_polars_categorical_fixes

    train_df = pl.DataFrame({"cat": ["a", "b", "a", "c"]}, schema={"cat": pl.Utf8})
    val_df = pl.DataFrame({"cat": ["a", "b"]}, schema={"cat": pl.Utf8})
    test_df = pl.DataFrame({"cat": ["a"]}, schema={"cat": pl.Utf8})

    out = apply_polars_categorical_fixes(
        train_df_polars=train_df,
        val_df_polars=val_df,
        test_df_polars=test_df,
        train_df_pd=None, val_df_pd=None, test_df_pd=None,
        filtered_train_df=None, filtered_val_df=None,
        cat_features=["cat"],
        align_polars_categorical_dicts=True,
        defer_pandas_conv=False,
        was_polars_input=True,
        verbose=False,
    )
    dt = out.train_df_polars.schema["cat"]
    # The cast must end up as pl.Enum, not pl.Categorical.
    assert isinstance(dt, pl.Enum) or str(dt).startswith("Enum"), (
        f"Expected pl.Enum cast for raw Utf8 cat_features (no global string cache pollution), got {dt!r}"
    )
    # Concrete check: pl.Categorical is the regression dtype we are avoiding.
    assert dt != pl.Categorical, "Step 4 leaked back to pl.Categorical (cache-poisoning regression)"


@pytest.mark.fast
def test_cast_utf8_cats_helper_uses_enum_when_domain_supplied():
    """The standalone _cast_utf8_cats_to_categorical helper must respect an explicit
    enum_domains argument and cast to pl.Enum.
    """
    from mlframe.training.core._phase_polars_fixes import _cast_utf8_cats_to_categorical

    df = pl.DataFrame({"cat": ["a", "b"]}, schema={"cat": pl.Utf8})
    out = _cast_utf8_cats_to_categorical(df, ["cat"], enum_domains={"cat": ["a", "b"]})
    dt = out.schema["cat"]
    assert dt != pl.Categorical
    assert isinstance(dt, pl.Enum) or str(dt).startswith("Enum")
