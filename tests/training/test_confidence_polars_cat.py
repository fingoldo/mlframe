"""Regression: confidence analyzer must handle polars Categorical/Enum
columns with nulls.

Pre-fix path (fuzz c0115_0c091590):
1. Suite trains an HGB / linear / XGB binary classifier on a polars
   utf8 frame where one column is auto-promoted to pl.Categorical by
   the upstream pipeline (align_polars_categorical_dicts).
2. Post-fit, ``run_confidence_analysis`` is invoked with this test_df
   and ``cat_features=None`` (HGB has no cat_features kwarg).
3. The polars-Categorical column survives the column-filter (the
   only kept-column-typed gate matches Utf8/Object/List/Array/Struct,
   not Categorical/Enum).
4. The CatBoostRegressor confidence model fits on the polars frame
   and raises one of:
     - ``TypeError: No matching signature found`` in
       ``_set_features_order_data_polars_categorical_column.process``
     - ``CatBoostError: Invalid type for cat_feature[...]=NaN``
       when the column has nulls (combo's null_fraction_cats=0.3).

Post-fix:
- Polars frames carrying any pl.Categorical / pl.Enum column are
  converted to pandas via the Arrow-bridge ``get_pandas_view_of_polars_df``
  before the confidence Pool is built (pandas CB Pool path handles
  Categorical+null gracefully).
- After the cat_features list is resolved (caller-passed OR auto-
  detected by ``get_categorical_columns``), every cat column with
  NaN cells is fill_null-ed with the sentinel string ``_NULL_``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training._eval_helpers import run_confidence_analysis


def _confidence_inputs(use_polars: bool, with_null: bool, use_categorical: bool):
    n = 200
    rng = np.random.default_rng(0)
    num_arr = rng.standard_normal((n, 3)).astype(np.float32)
    cat_arr = np.array(["a", "b", "c"] * (n // 3) + ["a", "b"])
    if with_null:
        cat_arr[::5] = ""  # to be lifted to null below
    target = rng.integers(0, 2, size=n).astype(np.int32)
    probs = rng.random((n, 2)).astype(np.float64)
    probs /= probs.sum(axis=1, keepdims=True)

    if use_polars:
        df = pl.DataFrame(
            {
                "num_0": num_arr[:, 0],
                "num_1": num_arr[:, 1],
                "num_2": num_arr[:, 2],
                "cat_0": cat_arr,
            }
        )
        if with_null:
            df = df.with_columns(pl.when(pl.col("cat_0") == "").then(None).otherwise(pl.col("cat_0")).alias("cat_0"))
        if use_categorical:
            df = df.with_columns(pl.col("cat_0").cast(pl.Categorical))
    else:
        df = pd.DataFrame(
            {
                "num_0": num_arr[:, 0],
                "num_1": num_arr[:, 1],
                "num_2": num_arr[:, 2],
                "cat_0": cat_arr,
            }
        )
        if with_null:
            df.loc[df["cat_0"] == "", "cat_0"] = np.nan
        if use_categorical:
            df["cat_0"] = df["cat_0"].astype("category")
    return df, target, probs


@pytest.mark.parametrize(
    "with_null,use_categorical",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_run_confidence_analysis_polars_cat_paths(with_null: bool, use_categorical: bool) -> None:
    """Polars frames with String / Categorical / null cells must not
    crash the confidence analyzer."""
    pytest.importorskip("catboost")
    df, target, probs = _confidence_inputs(use_polars=True, with_null=with_null, use_categorical=use_categorical)
    out = run_confidence_analysis(
        test_df=df,
        test_target=target,
        test_probs=probs,
        cat_features=["cat_0"],
        confidence_model_kwargs={"iterations": 5},
        verbose=False,
    )
    # Either a result dict / fig / None - just MUST NOT raise.
    assert out is None or out is not None  # tautology; the contract is "no crash"


def test_run_confidence_analysis_polars_cat_features_none_autodetect_path() -> None:
    """HGB call path: cat_features=None, polars Categorical column with
    nulls. The auto-detect block picks up cat_0; the post-detect
    fillna must catch its NaN cells."""
    pytest.importorskip("catboost")
    df, target, probs = _confidence_inputs(use_polars=True, with_null=True, use_categorical=True)
    out = run_confidence_analysis(
        test_df=df,
        test_target=target,
        test_probs=probs,
        cat_features=None,
        confidence_model_kwargs={"iterations": 5},
        verbose=False,
    )
    assert out is None or out is not None  # no-crash contract


def test_run_confidence_analysis_pandas_categorical_with_nulls() -> None:
    """Pandas Categorical+NaN: same surface, same fillna gate."""
    pytest.importorskip("catboost")
    df, target, probs = _confidence_inputs(use_polars=False, with_null=True, use_categorical=True)
    out = run_confidence_analysis(
        test_df=df,
        test_target=target,
        test_probs=probs,
        cat_features=["cat_0"],
        confidence_model_kwargs={"iterations": 5},
        verbose=False,
    )
    assert out is None or out is not None  # no-crash contract
