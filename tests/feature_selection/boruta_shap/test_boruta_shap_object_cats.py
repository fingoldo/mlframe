"""Regression: ``BorutaShap.fit`` must auto-ordinal-encode object /
pandas-Categorical columns before the internal surrogate fit so LGB /
XGB / RandomForest don't choke on raw cat strings.

Pre-fix path (fuzz iter-179 family, e.g. iter-237 / iter-239 / iter-308 /
iter-314 / iter-332 / iter-334):
- Suite config sets ``boruta_shap=True``.
- The main cat-encoder runs on the polars-pre frame (or is configured for
  ``cat_enc=ordinal`` but bypassed for polars-fastpath models). BorutaShap
  is appended to ``pre_pipelines`` (``_setup_helpers.py:428``) and receives
  a pandas frame whose cat columns are still ``object`` dtype.
- ``Train_model`` calls ``self.model.fit(X, y)`` (line 289 non-CB branch);
  LGBMRegressor raises ``ValueError: could not convert string to float:
  'A'`` and the entire BorutaShap selection branch is dropped.

Post-fix: ``BorutaShap.fit`` ordinal-encodes object / Categorical cols on
``self.X`` (the internal copy) so the surrogate fit + downstream SHAP
explainer see numeric values. ``transform`` still subsets the
CALLER-supplied frame by column NAME, so the encoding is private to the
internal selection path and never leaks into downstream model inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


pytest.importorskip("shap")
pytest.importorskip("lightgbm")


def _make_mixed_frame(seed: int = 17, n: int = 240) -> tuple[pd.DataFrame, pd.Series]:
    """Make mixed frame."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "num0": rng.standard_normal(n).astype(np.float64),
            "num1": rng.standard_normal(n).astype(np.float64),
            "cat_obj": rng.choice(list("ABCDE"), size=n),
            "cat_pd": pd.Categorical(rng.choice(list("XYZ"), size=n)),
        }
    )
    y = pd.Series(df["num0"] * 1.4 - df["num1"] * 0.6 + (df["cat_obj"] == "A").astype(float) * 0.8 + rng.standard_normal(n) * 0.2)
    return df, y


def test_boruta_shap_encodes_object_cats_before_surrogate_fit():
    """Boruta shap encodes object cats before surrogate fit."""
    from lightgbm import LGBMRegressor
    from mlframe.feature_selection.boruta_shap import BorutaShap

    df, y = _make_mixed_frame()
    selector = BorutaShap(
        model=LGBMRegressor(n_estimators=20, num_leaves=15, verbose=-1),
        importance_measure="shap",
        classification=False,
        n_trials=5,
        sample=False,
        normalize=True,
        verbose=False,
    )
    selector.fit(df, y)
    assert hasattr(selector, "support_")
    assert selector.support_.shape == (df.shape[1],)


def test_boruta_shap_transform_returns_caller_frame_unmodified():
    """Internal encoding must NOT leak: ``transform`` returns columns
    sliced out of the CALLER-supplied frame, preserving the original
    dtype (object string)."""
    from lightgbm import LGBMRegressor
    from mlframe.feature_selection.boruta_shap import BorutaShap

    df, y = _make_mixed_frame()
    selector = BorutaShap(
        model=LGBMRegressor(n_estimators=15, num_leaves=15, verbose=-1),
        importance_measure="shap",
        classification=False,
        n_trials=3,
        sample=False,
        normalize=True,
        verbose=False,
    )
    selector.fit(df, y)
    out = selector.transform(df)
    for col in out.columns:
        original = df[col]
        roundtripped = out[col]
        assert original.dtype == roundtripped.dtype, (
            f"BorutaShap.transform({col!r}) returned dtype {roundtripped.dtype} "
            f"but caller's column dtype is {original.dtype}. The internal "
            "ordinal-encoding must stay private."
        )


def test_ordinal_encode_object_cols_inplace_handles_nan_via_minus_one():
    """Ordinal encode object cols inplace handles nan via minus one."""
    from mlframe.feature_selection.boruta_shap import BorutaShap

    df = pd.DataFrame(
        {
            "obj_col": pd.Series(["A", "B", None, "A", "C"], dtype=object),
            "cat_col": pd.Categorical(["X", None, "Y", "X", None]),
            "num_col": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    touched = BorutaShap._ordinal_encode_object_cols_inplace(df)
    assert set(touched) == {"obj_col", "cat_col"}
    assert df["obj_col"].dtype == np.int32
    assert df["cat_col"].dtype == np.int32
    assert df["num_col"].dtype == np.float64
    # NaN -> -1 sentinel so surrogate can split on missing.
    assert (df["obj_col"] == -1).sum() == 1
    assert (df["cat_col"] == -1).sum() == 2


def test_ordinal_encode_is_a_noop_on_already_numeric_frame():
    """Ordinal encode is a noop on already numeric frame."""
    from mlframe.feature_selection.boruta_shap import BorutaShap

    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3, 4]})
    touched = BorutaShap._ordinal_encode_object_cols_inplace(df)
    assert touched == []
    # untouched
    assert df["a"].dtype == np.float64
