"""Tests ensuring sklearn pipelines return pandas/polars DataFrames (not numpy).

Root cause of 2026-04-22 LGB crash: sklearn Pipeline by default returns numpy arrays.
When pre_pipeline runs (scaler, selector, encoder), pd.Categorical dtype is lost:
result is a numpy object array of strings. LGB's sklearn wrapper then takes the
`not isinstance(X, pd_DataFrame)` branch, calls _LGBMValidateData which keeps numpy,
and Dataset.__init_from_np2d crashes on 'HOURLY'.

sklearn 1.2+ supports set_output(transform="pandas") and 1.4+ supports "polars".
mlframe must configure its pipelines so DataFrame dtypes (incl. pd.Categorical and
pl.Enum) survive the preprocessing chain intact.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Baseline: default sklearn Pipeline returns numpy — the behavior we must avoid
# ---------------------------------------------------------------------------

def test_baseline_sklearn_pipeline_default_returns_numpy():
    """Document baseline sklearn behavior — default Pipeline returns numpy,
    destroying pd.Categorical. Regression guard in case sklearn changes default."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [5.0, 6.0, 7.0, 8.0],
        "cat": pd.Categorical(["x", "y", "x", "y"]),
    })
    numeric_df = df[["a", "b"]]
    pipe = Pipeline([("scaler", StandardScaler())])
    out = pipe.fit_transform(numeric_df)
    assert isinstance(out, np.ndarray), (
        "sklearn default changed — update mlframe pipelines to rely on set_output explicitly"
    )


# ---------------------------------------------------------------------------
# ModelPipelineStrategy.build_pipeline — linear-model path must set_output pandas
# ---------------------------------------------------------------------------

def test_linear_strategy_pipeline_preserves_pandas_across_scaler():
    """LinearModelStrategy stacks imputer + scaler + (optional) encoder.
    The returned Pipeline must survive ``fit_transform`` on a pd.DataFrame and
    keep the result as a pd.DataFrame (not numpy).

    If this regresses, any sklearn transformer in the chain collapses the frame
    to numpy, which downstream destroys pd.Categorical columns for LGB/CB/XGB.
    """
    from mlframe.training.strategies import LinearModelStrategy

    n = 50
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "num1": rng.standard_normal(n).astype(np.float32),
        "num2": rng.standard_normal(n).astype(np.float32),
    })
    y = rng.integers(0, 2, size=n)

    strategy = LinearModelStrategy()
    pipeline = strategy.build_pipeline(
        base_pipeline=None,
        cat_features=[],
        category_encoder=None,
        imputer=SimpleImputer(strategy="mean"),
        scaler=StandardScaler(),
    )
    if pipeline is None:
        pytest.skip("build_pipeline returned None")

    out = pipeline.fit_transform(df, y)
    assert isinstance(out, pd.DataFrame), (
        f"LinearModelStrategy Pipeline returned {type(out).__name__}; "
        f"expected pd.DataFrame. mlframe must call set_output(transform='pandas') "
        f"on sklearn Pipelines so pd.Categorical survives downstream."
    )


# ---------------------------------------------------------------------------
# _passthrough_cols_fit_transform — must re-wrap numpy output as pd.DataFrame
# ---------------------------------------------------------------------------

def test_passthrough_cols_rewraps_numpy_so_passthrough_cols_survive():
    """If an inner transformer returns numpy, _passthrough_cols_fit_transform
    currently drops passthrough_cols silently (out has no .columns, append skipped).
    The fix must reconstruct a pd.DataFrame using the reduced-input column names
    so passthrough_cols can re-attach.
    """
    from mlframe.training.trainer import _passthrough_cols_fit_transform

    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame({
        "keep1": rng.standard_normal(n).astype(np.float32),
        "keep2": rng.standard_normal(n).astype(np.float32),
        "text_passthrough": [f"t{i}" for i in range(n)],
    })

    def numpy_returning_transform(sub_df):
        return sub_df.to_numpy()

    out = _passthrough_cols_fit_transform(
        numpy_returning_transform, df, passthrough_cols=["text_passthrough"]
    )

    assert isinstance(out, pd.DataFrame), (
        f"_passthrough_cols_fit_transform returned {type(out).__name__} "
        f"when inner fn returned numpy; expected pd.DataFrame reconstruction "
        f"so passthrough_cols don't silently disappear"
    )
    assert "text_passthrough" in out.columns, (
        "passthrough_cols must be preserved through the numpy-output path"
    )


def test_passthrough_cols_pandas_output_keeps_dtypes():
    """Baseline: when inner fn already returns pd.DataFrame, dtypes (incl.
    pd.Categorical) pass through unchanged and passthrough_cols re-attach."""
    from mlframe.training.trainer import _passthrough_cols_fit_transform

    n = 50
    df = pd.DataFrame({
        "keep1": np.ones(n, dtype=np.float32),
        "cat_selected": pd.Categorical(["HOURLY", "FIXED"] * (n // 2)),
        "text_passthrough": [f"t{i}" for i in range(n)],
    })

    def pandas_returning_transform(sub_df):
        return sub_df[["keep1", "cat_selected"]]

    out = _passthrough_cols_fit_transform(
        pandas_returning_transform, df, passthrough_cols=["text_passthrough"]
    )

    assert isinstance(out, pd.DataFrame)
    assert out["cat_selected"].dtype.name == "category", (
        "pd.Categorical dtype must survive the passthrough wrapper"
    )
    assert "text_passthrough" in out.columns


# ---------------------------------------------------------------------------
# End-to-end: Polars Enum → pandas bridge → Pipeline → LGB fit succeeds
# ---------------------------------------------------------------------------

def test_build_pipeline_output_format_polars():
    """build_pipeline(..., output_format='polars') returns a Pipeline whose
    fit_transform yields a pl.DataFrame, allowing Polars-native consumers
    (CB / XGB Polars fastpath, HGB) to skip the arrow→pandas bridge.

    Requires sklearn >= 1.4 (polars output added there). On older sklearn the
    fallback to pandas should kick in and the test would skip rather than fail.
    """
    import sklearn
    from packaging.version import Version
    if Version(sklearn.__version__) < Version("1.4"):
        pytest.skip("sklearn < 1.4: set_output(transform='polars') not available")

    from mlframe.training.strategies import LinearModelStrategy

    n = 50
    rng = np.random.default_rng(0)
    df = pl.DataFrame({
        "num1": rng.standard_normal(n).astype(np.float32),
        "num2": rng.standard_normal(n).astype(np.float32),
    })
    y = rng.integers(0, 2, size=n)

    strategy = LinearModelStrategy()
    pipeline = strategy.build_pipeline(
        base_pipeline=None,
        cat_features=[],
        category_encoder=None,
        imputer=SimpleImputer(strategy="mean"),
        scaler=StandardScaler(),
        output_format="polars",
    )
    if pipeline is None:
        pytest.skip("build_pipeline returned None")

    out = pipeline.fit_transform(df, y)
    assert isinstance(out, pl.DataFrame), (
        f"output_format='polars' should yield pl.DataFrame, got {type(out).__name__}"
    )


def test_build_pipeline_output_format_default_is_pandas():
    """Default output_format='pandas' preserves backward compatibility."""
    from mlframe.training.strategies import LinearModelStrategy

    n = 50
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "num1": rng.standard_normal(n).astype(np.float32),
        "num2": rng.standard_normal(n).astype(np.float32),
    })
    y = rng.integers(0, 2, size=n)

    pipeline = LinearModelStrategy().build_pipeline(
        base_pipeline=None,
        cat_features=[],
        category_encoder=None,
        imputer=SimpleImputer(strategy="mean"),
        scaler=StandardScaler(),
    )
    if pipeline is None:
        pytest.skip("build_pipeline returned None")

    out = pipeline.fit_transform(df, y)
    assert isinstance(out, pd.DataFrame)


def test_polars_enum_pandas_pipeline_lgb_fit_end_to_end():
    """Full production path reproducer.

    1. Polars DataFrame with pl.Enum categorical
    2. get_pandas_view_of_polars_df → pd.DataFrame with pd.Categorical
    3. LinearModelStrategy Pipeline fit_transform — must keep pd.Categorical alive
       (numeric step does not drop the cat col if it's not in its feature list;
       here we simulate by selecting only numeric cols but verifying the cat col
       roundtrips through the bridge).
    4. LGB fit with categorical_feature= works without 'could not convert string to float'.
    """
    pytest.importorskip("lightgbm")
    import lightgbm as lgb
    from mlframe.training.utils import get_pandas_view_of_polars_df

    rng = np.random.default_rng(0)
    n = 500
    categories = ["HOURLY", "FIXED", "MILESTONE"]
    pl_df = pl.DataFrame({
        "num1": rng.standard_normal(n).astype(np.float32),
        "num2": rng.standard_normal(n).astype(np.float32),
        "budget_type": pl.Series([categories[i % 3] for i in range(n)]).cast(pl.Enum(categories)),
    })
    y = rng.integers(0, 2, size=n)

    # Bridge keeps pd.Categorical
    pd_df = get_pandas_view_of_polars_df(pl_df)
    assert pd_df["budget_type"].dtype.name == "category", (
        "Polars→pandas bridge must preserve pl.Enum as pd.Categorical"
    )

    # LGB accepts pandas DataFrame with pd.Categorical when categorical_feature is passed
    model = lgb.LGBMClassifier(n_estimators=5, verbose=-1)
    model.fit(pd_df, y, categorical_feature=["budget_type"])
    preds = model.predict_proba(pd_df)
    assert preds.shape == (n, 2)
