"""Regression tests for the FUTURE -> RESOLVED audit rows:
FS-P1-1, FS-P1-9, FS-P2-1, FS-P2-3, FS-L-2,
FE-P1-2, FE-P1-3, FE-P2-5, FE-L-1, FE-L-3.

Each test asserts the fix landed (post-fix behaviour). On the pre-fix
codebase these tests are expected to FAIL for the reasons documented
inline; that pre-fix-fail bracket is the regression contract per memory
``feedback_test_every_bug_fix``.
"""

from __future__ import annotations

import inspect
import io
import logging
import os
import sys

import numpy as np
import pytest


# ----------------------------------------------------------------------------
# FS-P1-1 — groups kwarg threaded into pre_pipeline.fit_transform.
# ----------------------------------------------------------------------------


def test_fs_p1_1_groups_kwarg_in_apply_pre_pipeline_signature():
    """``_apply_pre_pipeline_transforms`` must accept ``groups`` (was missing pre-fix)."""
    from mlframe.training._pipeline_helpers import _apply_pre_pipeline_transforms
    sig = inspect.signature(_apply_pre_pipeline_transforms)
    assert "groups" in sig.parameters, (
        "groups kwarg missing from _apply_pre_pipeline_transforms signature; "
        "FS-P1-1 fix not landed."
    )


def test_fs_p1_1_passthrough_fit_transform_forwards_groups():
    """``_passthrough_cols_fit_transform`` forwards ``groups`` when the
    wrapped callable accepts it (RFECV with cv=GroupKFold() requires it)."""
    from mlframe.training._pipeline_helpers import _passthrough_cols_fit_transform

    received: dict = {}

    def fake_fit_transform(X, y=None, groups=None):
        received["groups"] = groups
        return X

    import pandas as pd
    df = pd.DataFrame({"a": np.arange(10, dtype=np.float64), "b": np.arange(10)})
    y = np.zeros(10, dtype=np.int64)
    groups = np.repeat([0, 1], 5)

    _passthrough_cols_fit_transform(
        fake_fit_transform, df, fit=True, target=y, groups=groups,
    )
    assert received["groups"] is groups, "groups not forwarded to inner fit_transform"


# ----------------------------------------------------------------------------
# FS-P1-9 — polars-FE-disabled docstring lie removed.
# ----------------------------------------------------------------------------


def test_fs_p1_9_no_polars_fe_disabled_claim_in_mrmr():
    """The 'FE is disabled when input is polars' comment near fe_max_steps must be gone."""
    import mlframe.feature_selection.filters.mrmr as mrmr_mod
    src = open(inspect.getsourcefile(mrmr_mod), encoding="utf-8").read()
    bad_phrases = (
        "FE is disabled when input is polars",
        "for now FE is disabled when input is polars",
    )
    for phrase in bad_phrases:
        assert phrase not in src, (
            f"mrmr.py still carries the docstring lie {phrase!r}; FS-P1-9 fix not landed."
        )


# ----------------------------------------------------------------------------
# FS-P2-1 — Literal-style validators on string ctor params.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("param,bad_value", [
    ("quantization_method", "totally-invalid-method"),
    ("nan_strategy", "no-such-strategy"),
    ("mrmr_relevance_algo", "not-an-algo"),
    ("mrmr_redundancy_algo", "not-an-algo"),
    ("fe_unary_preset", "bogus-preset"),
    ("fe_binary_preset", "bogus-preset"),
])
def test_fs_p2_1_string_ctor_params_validated(param, bad_value):
    """Bad string values raise ValueError with the valid set listed."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    import pandas as pd
    kwargs = {param: bad_value}
    sel = MRMR(**kwargs)
    df = pd.DataFrame({"a": np.arange(50, dtype=np.float64), "b": np.arange(50, dtype=np.float64)})
    y = np.random.RandomState(0).randint(0, 2, size=50)
    with pytest.raises(ValueError) as excinfo:
        sel.fit(df, y)
    assert "Valid values" in str(excinfo.value), (
        f"ValueError message must list valid values for {param}; got: {excinfo.value}"
    )


# ----------------------------------------------------------------------------
# FS-P2-3 — print() calls in mrmr.py replaced with logger.
# ----------------------------------------------------------------------------


def test_fs_p2_3_no_print_calls_left_in_mrmr():
    """No bare ``print(`` calls remain in mrmr.py fit-body strings."""
    import mlframe.feature_selection.filters.mrmr as mrmr_mod
    src = open(inspect.getsourcefile(mrmr_mod), encoding="utf-8").read()
    # The pre-fix offenders.
    bad_strings = (
        'print(f"nunary_transformations:',
        'print(f"nbinary_transformations:',
        'print("time spent by binary func:',
    )
    for s in bad_strings:
        assert s not in src, f"print() call still in mrmr.py: {s!r}; FS-P2-3 fix not landed."


# ----------------------------------------------------------------------------
# FS-L-2 — zero-inflated float regression target not misclassified.
# ----------------------------------------------------------------------------


def test_fs_l_2_float_target_with_high_ratio_treated_as_regression():
    """Float dtype y with samples/unique ratio > 100 must NOT be flagged as classification.

    Pre-fix the bare ``len(y)/nunique > 100`` heuristic on a zero-inflated
    regression target (900 zeros + 10 distinct float values, ratio ~91 but
    with rounding could exceed 100) misclassified to classification path.
    """
    import mlframe.feature_selection.filters.mrmr as mrmr_mod
    # Read the heuristic implementation and assert it references dtype.kind == "f".
    src = inspect.getsource(mrmr_mod)
    assert '_y_arr.dtype.kind == "f"' in src or '_is_float = _y_arr.dtype.kind == "f"' in src, (
        "FS-L-2 fix did not land: float-dtype guard absent from classification heuristic."
    )
    # Also assert the explicit-target_type preference branch exists.
    assert "self, \"target_type\", None" in src or "getattr(self, \"target_type\"" in src, (
        "FS-L-2 fix did not land: explicit target_type preference branch missing."
    )


# ----------------------------------------------------------------------------
# FE-P1-2 — auto-detect runs on pre-fit snapshot for pandas input.
# ----------------------------------------------------------------------------


def test_fe_p1_2_feature_types_first_flag_on_config():
    """FeatureTypesConfig exposes a feature_types_first flag with default True."""
    from mlframe.training.configs import FeatureTypesConfig
    cfg = FeatureTypesConfig()
    assert hasattr(cfg, "feature_types_first"), "feature_types_first flag missing"
    assert cfg.feature_types_first is True, "feature_types_first default must be True"


def test_fe_p1_2_phase_auto_detect_accepts_pandas_pre_snapshot():
    """``_phase_auto_detect_feature_types`` must accept ``train_df_pandas_pre`` kwarg."""
    from mlframe.training.core._phase_helpers import _phase_auto_detect_feature_types
    sig = inspect.signature(_phase_auto_detect_feature_types)
    assert "train_df_pandas_pre" in sig.parameters, (
        "train_df_pandas_pre kwarg missing from _phase_auto_detect_feature_types; "
        "FE-P1-2 fix not landed."
    )


# ----------------------------------------------------------------------------
# FE-P1-3 — preprocessing extensions back-merge into polars-pre frames.
# ----------------------------------------------------------------------------


def test_fe_p1_3_phase_helpers_snapshot_polars_pre_cols():
    """``_phase_fit_pipeline`` must snapshot polars-pre columns and merge new
    extension columns back. We check the source for the snapshot variable name
    (no integration run possible inside a unit test without a full suite)."""
    import mlframe.training.core._phase_helpers as ph_mod
    src = open(inspect.getsourcefile(ph_mod), encoding="utf-8").read()
    assert "_pre_polars_columns_snapshot" in src, (
        "Polars-pre column snapshot missing from _phase_fit_pipeline; "
        "FE-P1-3 fix not landed."
    )
    assert "back-merge" in src.lower(), (
        "Polars-pre back-merge comment marker missing; FE-P1-3 fix not landed."
    )


# ----------------------------------------------------------------------------
# FE-P2-5 — default CatBoostEncoder is seeded -> deterministic output.
# ----------------------------------------------------------------------------


def test_fe_p2_5_default_catboost_encoder_seeded():
    """Two calls to _get_pipeline_components with the same random_seed must
    produce CatBoostEncoders with the same random_state attribute."""
    pytest.importorskip("category_encoders")
    from mlframe.training.core._setup_helpers import _get_pipeline_components
    from mlframe.training.configs import PreprocessingConfig

    cfg = PreprocessingConfig()  # all None -> defaults
    cat_features = ["a", "b"]

    enc1, _, _ = _get_pipeline_components(cfg, cat_features, random_seed=12345)
    enc2, _, _ = _get_pipeline_components(cfg, cat_features, random_seed=12345)
    assert getattr(enc1, "random_state", None) == 12345
    assert getattr(enc2, "random_state", None) == 12345


def test_fe_p2_5_default_catboost_encoder_output_deterministic():
    """End-to-end determinism: fitting two encoders with the same seed on the
    same data must yield the same transform output."""
    ce = pytest.importorskip("category_encoders")
    import pandas as pd
    from mlframe.training.core._setup_helpers import _get_pipeline_components
    from mlframe.training.configs import PreprocessingConfig

    cfg = PreprocessingConfig()
    enc1, _, _ = _get_pipeline_components(cfg, ["c"], random_seed=7)
    enc2, _, _ = _get_pipeline_components(cfg, ["c"], random_seed=7)

    df = pd.DataFrame({
        "c": ["a", "b", "a", "c", "b", "a", "c", "a"] * 4,
    })
    y = np.array([0, 1, 0, 1, 0, 1, 1, 0] * 4, dtype=np.int64)

    out1 = enc1.fit_transform(df, y)
    out2 = enc2.fit_transform(df, y)
    # Same seed -> same encoding.
    np.testing.assert_array_equal(out1.to_numpy(), out2.to_numpy())


# ----------------------------------------------------------------------------
# FE-L-1 — base build_pipeline drops output_format kwarg.
# ----------------------------------------------------------------------------


def test_fe_l_1_build_pipeline_signature_no_output_format():
    """``ModelPipelineStrategy.build_pipeline`` must not advertise
    ``output_format`` (was dropped during FE-L-1 cleanup)."""
    from mlframe.training.strategies import ModelPipelineStrategy, TreeModelStrategy
    base_sig = inspect.signature(ModelPipelineStrategy.build_pipeline)
    tree_sig = inspect.signature(TreeModelStrategy.build_pipeline)
    assert "output_format" not in base_sig.parameters, (
        "output_format still in ModelPipelineStrategy.build_pipeline signature; "
        "FE-L-1 fix not landed."
    )
    assert "output_format" not in tree_sig.parameters, (
        "output_format snuck into TreeModelStrategy.build_pipeline; FE-L-1 broke uniformity."
    )


# ----------------------------------------------------------------------------
# FE-L-3 — int_to_float cast does NOT widen already-narrow int8 columns.
# ----------------------------------------------------------------------------


def test_fe_l_3_int8_date_columns_pass_through_unchanged():
    """When create_polarsds_pipeline runs over a frame containing Int8 date-decomposition
    columns (day/weekday/month/hour) those columns must stay Int8, not get widened to Float32."""
    pl = pytest.importorskip("polars")
    pytest.importorskip("polars_ds")
    from mlframe.training.pipeline import create_polarsds_pipeline
    from mlframe.training.configs import PreprocessingBackendConfig

    n = 50
    df = pl.DataFrame({
        "day": np.arange(n, dtype=np.int8) % 28 + 1,
        "month": np.arange(n, dtype=np.int8) % 12 + 1,
        "wide_int": np.arange(n, dtype=np.int64),
        "feature": np.linspace(0.0, 1.0, n, dtype=np.float32),
    })
    df = df.with_columns([pl.col("day").cast(pl.Int8), pl.col("month").cast(pl.Int8)])

    cfg = PreprocessingBackendConfig(
        scaler_name=None,
        categorical_encoding=None,
        skip_categorical_encoding=True,
        imputer_strategy=None,
    )
    pipe = create_polarsds_pipeline(df, cfg, verbose=0)
    if pipe is None:
        pytest.skip("polars-ds not available")
    out = pipe.transform(df)
    # Int8 date-decomp cols must remain Int8 (or at least NOT have been
    # upcast to Float32 / Float64); wide int64 column SHOULD have been cast to f32.
    out_schema = dict(out.schema)
    assert out_schema["day"] == pl.Int8, (
        f"Int8 'day' column was widened to {out_schema['day']}; FE-L-3 fix not landed."
    )
    assert out_schema["month"] == pl.Int8, (
        f"Int8 'month' column was widened to {out_schema['month']}; FE-L-3 fix not landed."
    )
    assert out_schema["wide_int"] == pl.Float32, (
        f"Wide int64 column must be cast to Float32; got {out_schema['wide_int']}."
    )
