"""Sensor test for audit B-P0-3 / Low-B11: the CB-native fastpath precast in
_phase_auto_detect_feature_types must cast Utf8/String cat columns to pl.Enum
(per-Series), not pl.Categorical (which widens the polars 1.x global string cache).

Memory rule: ``reference_polars_global_string_cache`` - pl.Categorical participates
in a process-wide string cache that grows monotonically and cannot be reset.
"""

from __future__ import annotations

import polars as pl
import pytest


@pytest.mark.fast
def test_phase_auto_detect_precast_strings_uses_enum_not_categorical():
    """When the CB-native fastpath kicks in (all_models_polars_native +
    skip_categorical_encoding), raw Utf8 columns must be precast to pl.Enum, NOT
    to pl.Categorical.
    """
    from mlframe.training.core._phase_helpers import _phase_auto_detect_feature_types

    train = pl.DataFrame(
        {"x_str": ["a", "b", "a", "c"], "y_num": [1.0, 2.0, 3.0, 4.0]},
        schema={"x_str": pl.Utf8, "y_num": pl.Float64},
    )
    val = pl.DataFrame(
        {"x_str": ["a", "b"], "y_num": [5.0, 6.0]},
        schema={"x_str": pl.Utf8, "y_num": pl.Float64},
    )
    test = pl.DataFrame(
        {"x_str": ["a"], "y_num": [7.0]},
        schema={"x_str": pl.Utf8, "y_num": pl.Float64},
    )

    # Minimal config shims matching the fastpath gating expectations.
    class _PipelineCfg:
        skip_categorical_encoding = True

    class _FeatureTypesCfg:
        text_features = []
        embedding_features = []
        cat_features = ["x_str"]
        auto_detect_feature_types = False  # disable autodetect to keep the test pinned
        honor_user_dtype = False

    metadata: dict = {}
    out = _phase_auto_detect_feature_types(
        train_df=train,
        val_df=val,
        test_df=test,
        train_df_polars_pre=train,
        val_df_polars_pre=val,
        test_df_polars_pre=test,
        cat_features=["x_str"],
        cat_features_polars=["x_str"],
        was_polars_input=True,
        all_models_polars_native=True,
        pipeline_config=_PipelineCfg(),
        feature_types_config=_FeatureTypesCfg(),
        metadata=metadata,
        verbose=False,
    )
    train_out = out[0]
    dt = train_out.schema["x_str"]
    assert isinstance(dt, pl.Enum) or str(dt).startswith("Enum"), f"Expected pl.Enum cast for raw Utf8 cat column in CB-fastpath; got {dt!r}"
    assert dt != pl.Categorical, "_precast_strings still uses pl.Categorical (cache pollution regression)"
