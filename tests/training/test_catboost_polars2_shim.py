"""CatBoost trains and predicts on a polars frame with a Categorical column on polars 2, where its Pool reads the removed get_categories."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

catboost = pytest.importorskip("catboost")

from mlframe.training._catboost_polars2_shim import install_catboost_polars2_shim, polars_lacks_get_categories


def _frame() -> pl.DataFrame:
    """A numeric column and a Categorical column, 60 rows."""
    rng = np.random.default_rng(0)
    return pl.DataFrame({"x": rng.random(60), "c": pl.Series(["u", "v", "w"] * 20, dtype=pl.Categorical)})


def test_catboost_fits_and_predicts_on_a_polars_categorical_column() -> None:
    """With the shim installed the fit and the predict on a Categorical-column polars frame both succeed and give 60 labels."""
    install_catboost_polars2_shim()
    frame, y = _frame(), np.tile([0, 1], 30)
    model = catboost.CatBoostClassifier(iterations=3, verbose=0, cat_features=["c"], allow_writing_files=False).fit(frame, y)
    assert model.predict(frame).shape[0] == 60


def test_the_shim_is_installed_exactly_when_polars_lacks_get_categories() -> None:
    """Polars 1 keeps CatBoost's own constructor; polars 2 gets the wrapper, and a second install is a no-op."""
    installed = install_catboost_polars2_shim()
    assert installed == polars_lacks_get_categories()
    assert install_catboost_polars2_shim() == installed
    assert catboost.Pool.__dict__.get("_mlframe_polars2_categorical_shim", False) == installed
