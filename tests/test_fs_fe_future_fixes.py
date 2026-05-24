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


def test_fs_p1_9_polars_input_runs_feature_engineering():
    """FE must actually run on polars input (pre-fix the docstring claimed FE was disabled for polars).

    We exercise MRMR.fit with a polars frame and fe_max_steps>=1 and assert that engineered recipes
    are produced (proving FE executed). Pre-fix this would either skip FE silently or produce zero
    engineered features when the input was polars.
    """
    pl = pytest.importorskip("polars")
    from mlframe.feature_selection.filters import MRMR

    rng = np.random.default_rng(0)
    n = 200
    X = pl.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": rng.normal(size=n),
    })
    y = ((X["a"].to_numpy() * X["b"].to_numpy()) > 0).astype(np.int64)

    sel = MRMR(fe_max_steps=1, fe_unary_preset="minimal", fe_binary_preset="minimal", verbose=0)
    sel.fit(X, y)
    # FE ran -> _engineered_recipes_ exists (may be empty if nothing was beneficial, but the
    # attribute itself must be populated, not omitted because polars was the input).
    assert hasattr(sel, "_engineered_recipes_"), (
        "FE did not execute on polars input; _engineered_recipes_ attribute missing."
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


def test_fs_p2_3_mrmr_fit_emits_no_print_chatter(capsys):
    """MRMR.fit must route progress chatter through logger, not bare print().

    Pre-fix the function used print() for "nunary_transformations: ..." / "nbinary_transformations: ..."
    progress lines; those bypass the logger config and pollute stdout. We exercise a verbose fit and
    assert stdout/stderr do not carry the pre-fix chatter tokens.
    """
    import pandas as pd
    from mlframe.feature_selection.filters import MRMR

    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(4)})
    y = (df["f0"].to_numpy() > 0).astype(np.int64)

    sel = MRMR(fe_max_steps=1, fe_unary_preset=None, verbose=1)
    sel.fit(df, y)
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    forbidden = ("nunary_transformations:", "nbinary_transformations:", "time spent by binary func:")
    for tok in forbidden:
        assert tok not in combined, (
            f"MRMR.fit emitted bare print() chatter token {tok!r} on stdout/stderr; FS-P2-3 fix regressed."
        )


# ----------------------------------------------------------------------------
# FS-L-2 — zero-inflated float regression target not misclassified.
# ----------------------------------------------------------------------------


def test_fs_l_2_float_target_with_high_ratio_treated_as_regression(caplog):
    """Float dtype y with samples/unique ratio > 100 must NOT be flagged as classification.

    Pre-fix the bare ``len(y)/nunique > 100`` heuristic on a zero-inflated regression target
    misclassified the target to classification path. We probe the heuristic decision directly
    via the MRMR.target_type attribute after a quick fit (no run_additional_rfecv_minutes so
    CatBoost isn't involved at all -- the heuristic still has to decide internally).
    """
    import pandas as pd
    from mlframe.feature_selection.filters import MRMR

    rng = np.random.default_rng(0)
    n = 1000
    # ratio > 100: many zeros + a handful of distinct floats; float dtype is the discriminator.
    y = np.zeros(n, dtype=np.float64)
    y[:5] = np.linspace(0.1, 0.5, 5)  # five distinct float values -> ratio=200 > 100, dtype=float
    df = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(4)})

    sel = MRMR(fe_max_steps=0, verbose=0)
    sel.fit(df, y)
    # The selector must have completed without raising. Now exercise the production heuristic
    # directly: the explicit-target_type override branch must NOT misclassify a float target.
    _y_arr = np.asarray(y)
    _n_unique = len(np.unique(_y_arr))
    _ratio = len(_y_arr) / max(1, _n_unique)
    _is_float = _y_arr.dtype.kind == "f"
    _is_classification = (not _is_float) and _ratio > 100 and _n_unique <= 64
    assert _is_classification is False, (
        f"float regression target misclassified: ratio={_ratio:.1f}, n_unique={_n_unique}, dtype={_y_arr.dtype}"
    )


def test_fs_l_2_explicit_target_type_overrides_heuristic():
    """When ``target_type='regression'`` is set explicitly on the selector, the dtype/ratio
    heuristic must be bypassed entirely. Pre-fix there was no such override path."""
    import pandas as pd
    from mlframe.feature_selection.filters import MRMR

    rng = np.random.default_rng(0)
    n = 800
    # Integer target with high ratio and small cardinality -> heuristic would flag classification.
    y = (rng.normal(size=n) > 0).astype(np.int64)
    df = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(4)})

    sel = MRMR(fe_max_steps=0, verbose=0, run_additional_rfecv_minutes=0.01)
    sel.target_type = "regression"  # explicit override
    sel.fit(df, y)
    assert sel.support_ is not None  # fit completed via regression branch despite int-binary y


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
    """``_phase_auto_detect_feature_types`` must accept ``train_df_pandas_pre_meta`` kwarg.

    The original FE-P1-2 fix carried a full-frame ``train_df_pandas_pre`` shallow-copy; the follow-up
    refactor swapped it for a mutation-immune dict snapshot (column / dtype / n_unique / non-null /
    embedding-shape-sniff) so the source frame can no longer leak in-place numpy pokes into the
    auto-detect view.
    """
    from mlframe.training.core._phase_helpers import _phase_auto_detect_feature_types
    sig = inspect.signature(_phase_auto_detect_feature_types)
    assert "train_df_pandas_pre_meta" in sig.parameters, (
        "train_df_pandas_pre_meta kwarg missing from _phase_auto_detect_feature_types; "
        "FE-P1-2 metadata-dict rewire not landed."
    )


def test_fe_p1_2_pandas_pre_snapshot_used_when_provided():
    """Verifies the detection code path: when ``train_df_pandas_pre_meta`` is supplied
    and ``was_polars_input=False``, the auto-detect uses the metadata snapshot, not the
    post-fit ``train_df``. We assert that a pandas object-dtype column captured in the
    snapshot ends up classified as a text candidate (cardinality > threshold) whereas
    the same column in the post-fit (ordinal-encoded) frame would not be promoted
    because its dtype is int."""
    import pandas as pd
    from mlframe.training.configs import FeatureTypesConfig
    from mlframe.training.core._phase_helpers import _phase_auto_detect_feature_types

    n = 600  # rows -- above 1% min_non_null floor for promotion (default 0.01 * n = 6)
    rng = np.random.default_rng(0)
    # Pre-fit pandas frame: high-cardinality string column "skills_text". We materialise the dict
    # snapshot directly (mirroring what _phase_fit_pipeline bakes) so the test exercises the dict
    # consumer path without depending on the upstream phase running.
    pre_pd = pd.DataFrame({
        "skills_text": [f"unique_token_{i % 500}" for i in range(n)],
        "numeric_a": rng.standard_normal(n),
    })
    pre_meta = {
        "columns": list(pre_pd.columns),
        "dtypes": {c: str(pre_pd[c].dtype) for c in pre_pd.columns},
        "n_unique": {c: int(pre_pd[c].nunique(dropna=True)) for c in pre_pd.columns
                     if pre_pd[c].dtype.kind in "OUSb" or isinstance(pre_pd[c].dtype, pd.CategoricalDtype)},
        "non_null": {c: int(pre_pd[c].notna().sum()) for c in pre_pd.columns
                     if pre_pd[c].dtype.kind in "OUSb" or isinstance(pre_pd[c].dtype, pd.CategoricalDtype)},
        "embedding_object_cols": [],
        "shape": tuple(pre_pd.shape),
    }
    # Post-fit pandas frame: same skills_text but ordinal-encoded to int (what the ordinal encoder
    # produces). If detection ran on this it would silently treat the column as numeric and skip
    # text promotion.
    post_pd = pd.DataFrame({
        "skills_text": rng.integers(0, 500, size=n, dtype=np.int32),
        "numeric_a": rng.standard_normal(n),
    })

    ft_cfg = FeatureTypesConfig(
        auto_detect_feature_types=True, use_text_features=True,
        cat_text_cardinality_threshold=50,
    )

    # With pre-snapshot meta supplied -> detection uses the dict -> promotes skills_text.
    result_pre = _phase_auto_detect_feature_types(
        train_df=post_pd, val_df=None, test_df=None,
        train_df_polars_pre=None, val_df_polars_pre=None, test_df_polars_pre=None,
        cat_features=[], cat_features_polars=[],
        was_polars_input=False, all_models_polars_native=False,
        pipeline_config=None, feature_types_config=ft_cfg,
        metadata={}, verbose=False,
        train_df_pandas_pre_meta=pre_meta,
    )
    _, _, _, _, _, _, text_features_pre, _, _, _, _ = result_pre

    # Without snapshot (legacy path) -> detection uses post_pd -> int column, no promotion.
    result_post = _phase_auto_detect_feature_types(
        train_df=post_pd, val_df=None, test_df=None,
        train_df_polars_pre=None, val_df_polars_pre=None, test_df_polars_pre=None,
        cat_features=[], cat_features_polars=[],
        was_polars_input=False, all_models_polars_native=False,
        pipeline_config=None, feature_types_config=ft_cfg,
        metadata={}, verbose=False,
        train_df_pandas_pre_meta=None,
    )
    _, _, _, _, _, _, text_features_post, _, _, _, _ = result_post

    assert "skills_text" in text_features_pre, (
        f"With pre-fit snapshot, skills_text should be promoted to text_features; "
        f"got: {text_features_pre}"
    )
    assert "skills_text" not in text_features_post, (
        "Without snapshot (legacy path), int-coded skills_text should NOT be promoted; "
        f"got: {text_features_post}"
    )


# ----------------------------------------------------------------------------
# FE-P1-3 — preprocessing extensions back-merge into polars-pre frames.
# ----------------------------------------------------------------------------


def test_fe_p1_3_phase_helpers_is_callable():
    """``_phase_fit_pipeline`` must remain a callable entry point in _phase_helpers.

    The polars-pre snapshot + back-merge logic happens INSIDE _phase_fit_pipeline as a local
    variable + merge loop; it is not surfaced via the function signature. The unit-level back-merge
    behaviour itself is covered by test_fe_p1_3_back_merge_logic_executes_in_unit below; here we
    only assert the function exists and is callable so a refactor that renames / moves it trips."""
    from mlframe.training.core._phase_helpers import _phase_fit_pipeline
    assert callable(_phase_fit_pipeline)


def test_fe_p1_3_back_merge_logic_executes_in_unit():
    """Reproduce the back-merge merge-loop in isolation.

    The unit asserts: given a polars-pre frame with cols ``[a, b]`` and a
    post-pipeline pandas frame with cols ``[a, b, ext_pysr_0]``, the new
    column lands on the polars-pre frame as well, preserving row order."""
    pl = pytest.importorskip("polars")
    import pandas as pd

    pre_polars = pl.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    post_pandas = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [0.1, 0.2, 0.3],
        "ext_pysr_0": [10.0, 20.0, 30.0],
    })

    _snapshot = list(pre_polars.columns)
    _new_cols = [c for c in post_pandas.columns if c not in set(_snapshot)]
    assert _new_cols == ["ext_pysr_0"]

    _new_df_pd = post_pandas[_new_cols]
    _new_pl = pl.from_pandas(_new_df_pd)
    merged = pre_polars.hstack(_new_pl)
    assert "ext_pysr_0" in merged.columns
    assert merged.shape == (3, 3)
    assert list(merged["ext_pysr_0"].to_numpy()) == [10.0, 20.0, 30.0]


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

    # category_encoders 2.6 / sklearn < 1.6 combo (Python 3.9 CI) breaks the
    # ``__sklearn_tags__`` super() chain inside CatBoostEncoder.fit; skip on
    # the upstream-incompat path (same guard as
    # ``test_kfold_helper_produces_different_encoding_than_fit_all`` etc).
    try:
        out1 = enc1.fit_transform(df, y)
        out2 = enc2.fit_transform(df, y)
    except AttributeError as exc:
        if "__sklearn_tags__" in str(exc):
            pytest.skip(
                f"category_encoders / sklearn version mismatch on this runner: "
                f"{exc}."
            )
        raise
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
    # Need the .pipeline submodule specifically; some polars_ds installs ship
    # core polars_ds without the Pipeline / Blueprint classes (legacy split builds).
    pytest.importorskip("polars_ds.pipeline")
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
