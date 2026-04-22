"""End-to-end LGB tests with realistic prod-like Polars input.

Why this file exists: production LGB crashes on 2026-04-22 with
    ValueError: could not convert string to float: 'HOURLY'
went undetected because every existing LGB test in tests/training/ feeds either:
    1. a pure-numeric pd.DataFrame (np.random.randn columns + numeric target),
    2. or a pl.from_pandas(...) wrap of the same — no pl.Enum, no cat_features,
       no align_polars_categorical_dicts step, no auto-detection of cat columns.

The crash path requires ALL of:
    * Polars input
    * pl.Enum or pl.Categorical columns declared as cat_features
    * full mlframe pipeline (lazy pandas conversion, pre_pipeline, fit_params build)
    * LGB in mlframe_models list

These tests construct exactly that minimal scenario so any future regression
in the Polars→pandas→LGB chain is caught locally instead of on the 9M-row
prod dataset 30 minutes into a run.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

pytest.importorskip("lightgbm")


def _make_prod_like_polars_frame(n: int = 2000, seed: int = 0) -> pl.DataFrame:
    """Mimic the 2026-04-22 prod schema in miniature: pl.Enum cat columns +
    numeric features + binary target, large enough to exercise the bridge but
    small enough to fit in CI wall-clock budget."""
    rng = np.random.default_rng(seed)
    budget_categories = ["HOURLY", "FIXED", "MILESTONE"]
    tier_categories = ["BEGINNER", "INTERMEDIATE", "EXPERT"]
    workload_categories = ["LESS_THAN_30", "MORE_THAN_30", "FULL_TIME"]
    return pl.DataFrame({
        "num_feat_1": rng.standard_normal(n).astype(np.float32),
        "num_feat_2": rng.standard_normal(n).astype(np.float32),
        "num_feat_3": rng.standard_normal(n).astype(np.float32),
        "budget_type": pl.Series([budget_categories[i % 3] for i in range(n)]).cast(pl.Enum(budget_categories)),
        "contractor_tier": pl.Series([tier_categories[i % 3] for i in range(n)]).cast(pl.Enum(tier_categories)),
        "workload": pl.Series([workload_categories[i % 3] for i in range(n)]).cast(pl.Enum(workload_categories)),
        "target": rng.integers(0, 2, n),
    })


def test_lgb_directly_handles_polars_enum_via_pandas_bridge():
    """Minimal repro: skip mlframe entirely, just verify that
    Polars(pl.Enum) → get_pandas_view_of_polars_df → LGBMClassifier.fit
    succeeds (no 'could not convert string to float: HOURLY')."""
    import lightgbm as lgb
    from mlframe.training.utils import get_pandas_view_of_polars_df

    pl_df = _make_prod_like_polars_frame(n=500)
    pd_df = get_pandas_view_of_polars_df(pl_df.drop("target"))
    y = pl_df["target"].to_numpy()

    cat_cols = ["budget_type", "contractor_tier", "workload"]
    # Sanity: bridge must keep cat columns as pd.Categorical
    for c in cat_cols:
        assert pd_df[c].dtype.name == "category", (
            f"Bridge dropped pd.Categorical for {c}; got {pd_df[c].dtype}"
        )

    model = lgb.LGBMClassifier(n_estimators=5, verbose=-1)
    model.fit(pd_df, y, categorical_feature=cat_cols)
    proba = model.predict_proba(pd_df)
    assert proba.shape == (500, 2)


def test_lgb_after_passthrough_wrapper_handles_numpy_inner_fn():
    """If a pre_pipeline transformer returns numpy (default sklearn behavior),
    _passthrough_cols_fit_transform must rebuild a pd.DataFrame so LGB still
    takes the pandas fastpath. This test exercises the rebuild path that
    sandwiches between the bridge and LGB.fit.
    """
    import lightgbm as lgb
    from mlframe.training.utils import get_pandas_view_of_polars_df
    from mlframe.training.trainer import _passthrough_cols_fit_transform

    pl_df = _make_prod_like_polars_frame(n=500)
    pd_df = get_pandas_view_of_polars_df(pl_df.drop("target"))
    y = pl_df["target"].to_numpy()

    # Simulate a transformer that returns numpy (selects only numeric cols).
    def numpy_returning_selector(sub_df):
        return sub_df[["num_feat_1", "num_feat_2", "num_feat_3"]].to_numpy()

    out = _passthrough_cols_fit_transform(
        numpy_returning_selector, pd_df,
        passthrough_cols=["budget_type", "contractor_tier", "workload"],
    )
    assert isinstance(out, pd.DataFrame), (
        f"_passthrough_cols_fit_transform must rebuild pd.DataFrame "
        f"to keep LGB fastpath alive; got {type(out).__name__}"
    )
    for c in ["budget_type", "contractor_tier", "workload"]:
        assert c in out.columns

    cat_cols = ["budget_type", "contractor_tier", "workload"]
    model = lgb.LGBMClassifier(n_estimators=5, verbose=-1)
    model.fit(out, y, categorical_feature=cat_cols)


@pytest.mark.parametrize("model_name", ["lgb", "xgb", "cb"])
def test_suite_polars_with_enum_cats_end_to_end(model_name, tmp_path):
    """The test that should have caught the 2026-04-22 LGB 'HOURLY' crash.

    Builds a Polars frame with pl.Enum cat columns, runs train_mlframe_models_suite
    with single model, asserts training completes without ValueError.

    Parametrized across all three tree models so XGB / CB regressions on the same
    path also get caught locally.
    """
    pytest.importorskip(
        {"lgb": "lightgbm", "xgb": "xgboost", "cb": "catboost"}[model_name]
    )

    from mlframe.training.core import train_mlframe_models_suite
    from .shared import SimpleFeaturesAndTargetsExtractor

    pl_df = _make_prod_like_polars_frame(n=600)
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

    # Minimal config so the run is fast but exercises the cat_features + Enum path.
    common_init_params = {
        "drop_columns": [],
        "verbose": 0,
    }
    config_override = {"iterations": 5}
    if model_name == "lgb":
        config_override["lgb_kwargs"] = {"device_type": "cpu", "verbose": -1}

    models, metadata = train_mlframe_models_suite(
        df=pl_df,
        target_name=f"{model_name}_polars_enum_test",
        model_name=f"{model_name}_polars_enum_test",
        features_and_targets_extractor=fte,
        mlframe_models=[model_name],
        hyperparams_config=config_override,
        init_common_params=common_init_params,
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        data_dir=str(tmp_path),
        models_dir="models",
        verbose=0,
    )

    # Smoke check: at least one model trained, no ValueError on 'HOURLY'.
    assert models, f"train_mlframe_models_suite returned empty models for {model_name}"
