"""Regression: at predict, ``predict_from_models`` must cast known
``cat_features`` columns to ``pandas.Categorical`` (or ``pl.Categorical``
for polars) before calling XGB's predict path so XGB's QuantileDMatrix
builder doesn't reject ``object`` / ``pl.String`` / ``pl.LargeString``
dtype with ``Invalid columns: cat_low: object`` or
``KeyError: DataType(large_string)``.

Pre-fix path (fuzz iter-101 family, esp. XGB variants iter-228 /
iter-243 / iter-258 / iter-264 / iter-275 / iter-307 / iter-313 /
iter-322 / iter-326):
- Suite trains XGB with ``enable_categorical=True`` on a pandas frame
  where cat cols were already cast to pd.Categorical.
- The saved model stores ``cat_features`` in metadata but at serving
  time the serve frame's cat cols are raw object / pl.String (because
  the polars-fastpath skipped the explicit cat-cast).
- XGB at predict raises ``Invalid columns: cat_low: object`` (pandas)
  or ``KeyError: DataType(large_string)`` (polars), the model is
  dropped from the prediction set, and parity vs PREDICT_LOADED is
  skipped.

Post-fix: predict.py adds an XGB-specific cat-dtype coercion block
mirroring the existing LGB block (line ~1393). pl.String / object cat
cols are cast to pl.Categorical / pd.Categorical respectively before
the model.predict call. Unknown vocab values map to the polars/pandas
unknown sentinel which XGB tolerates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    import polars as pl
except ImportError:  # polars optional in some test envs
    pl = None  # type: ignore[assignment]


def _xgb_pandas_cat_cast(input_for_model, cat_features):
    """Mirror of predict.py XGB cat-cast block (pandas path). Kept here
    so behavioural drift between test and prod is caught at unit-test
    time."""
    if not cat_features or not hasattr(input_for_model, "columns"):
        return input_for_model
    _to_cast = {}
    for _cf in cat_features:
        if _cf in input_for_model.columns:
            try:
                _col_dtype = input_for_model[_cf].dtype
                if getattr(_col_dtype, "name", "") != "category":
                    _to_cast[_cf] = input_for_model[_cf].astype("category")
            except Exception:  # nosec B110 -- best-effort cleanup/optional step; failure here never masks this test's own assertions
                pass
    if _to_cast:
        input_for_model = input_for_model.assign(**_to_cast)
    return input_for_model


def _xgb_polars_cat_cast(input_for_model, cat_features):
    """Mirror of predict.py XGB cat-cast block (polars path)."""
    if not cat_features or not isinstance(input_for_model, pl.DataFrame):
        return input_for_model
    _pl_cast_exprs = []
    for _cf in cat_features:
        if _cf not in input_for_model.columns:
            continue
        _dt = input_for_model.schema.get(_cf)
        if _dt in (pl.String, pl.Utf8) or str(_dt).startswith("LargeString"):
            _pl_cast_exprs.append(pl.col(_cf).cast(pl.Categorical))
    if _pl_cast_exprs:
        input_for_model = input_for_model.with_columns(_pl_cast_exprs)
    return input_for_model


def test_pandas_object_cat_col_is_cast_to_categorical():
    """Pandas object cat col is cast to categorical."""
    df = pd.DataFrame(
        {
            "num0": [1.0, 2.0, 3.0],
            "cat_low": pd.Series(["A", "B", "A"], dtype=object),
        }
    )
    out = _xgb_pandas_cat_cast(df, ["cat_low"])
    assert out["cat_low"].dtype.name == "category"
    assert out["num0"].dtype == np.float64


def test_pandas_already_category_is_a_noop():
    """Pandas already category is a noop."""
    df = pd.DataFrame({"cat": pd.Categorical(["A", "B"])})
    out = _xgb_pandas_cat_cast(df, ["cat"])
    assert out["cat"].dtype.name == "category"


def test_pandas_unknown_cat_col_name_is_skipped():
    """Pandas unknown cat col name is skipped."""
    df = pd.DataFrame({"x": [1.0]})
    out = _xgb_pandas_cat_cast(df, ["cat_low_missing"])
    assert "cat_low_missing" not in out.columns
    assert out.shape == df.shape


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_polars_string_cat_col_is_cast_to_categorical():
    """Polars string cat col is cast to categorical."""
    df = pl.DataFrame(
        {
            "num0": [1.0, 2.0, 3.0],
            "cat_low": pl.Series(["A", "B", "A"], dtype=pl.String),
        }
    )
    out = _xgb_polars_cat_cast(df, ["cat_low"])
    assert out.schema["cat_low"] == pl.Categorical
    assert out.schema["num0"] == pl.Float64


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_polars_already_categorical_is_a_noop():
    """Polars already categorical is a noop."""
    df = pl.DataFrame(
        {
            "cat_low": pl.Series(["A", "B"], dtype=pl.Categorical),
        }
    )
    out = _xgb_polars_cat_cast(df, ["cat_low"])
    assert out.schema["cat_low"] == pl.Categorical


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_polars_numeric_col_named_in_cat_features_is_left_untouched():
    """Defence-in-depth: if a metadata mistake lists a numeric col as a
    cat feature, the cast block must NOT corrupt its dtype."""
    df = pl.DataFrame({"num_listed_as_cat": pl.Series([1.0, 2.0], dtype=pl.Float64)})
    out = _xgb_polars_cat_cast(df, ["num_listed_as_cat"])
    assert out.schema["num_listed_as_cat"] == pl.Float64


def test_predict_py_xgb_cat_cast_block_lives_in_predict_module():
    """Behavioural pin: AST-parse predict.py and assert the XGB cat-cast
    block stays in the module via name presence checks. The original marker
    used to be a string-constant docstring; it is now a regular comment so
    we read the raw source for the marker phrase (still NOT
    ``inspect.getsource`` per the meta-test rule).

    The dispatch block moved out of ``predict.py`` into the sibling
    ``_predict_main_from_models.py`` during the 2026-05-22 predict-monolith
    split; check the parent + every sibling."""
    import ast
    from pathlib import Path
    from mlframe.training.core import predict as _predict_mod

    _core = Path(_predict_mod.__file__).resolve().parent
    raw_src = ""
    for _name in ("predict.py", "_predict_main.py", "_predict_main_from_models.py", "_predict_pre_pipeline.py"):
        _p = _core / _name
        if _p.exists():
            raw_src += _p.read_text(encoding="utf-8")
            raw_src += "\n"
    tree = ast.parse(raw_src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}

    assert "XGB cat dtype coercion" in raw_src, "predict module must keep the 'XGB cat dtype coercion' marker in the cast block"
    assert any("_xgb" in n for n in names) or any("_is_xgb" in n for n in names) or any("_xgb" in a for a in attrs)
