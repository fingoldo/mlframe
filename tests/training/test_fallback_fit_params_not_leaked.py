"""The CatBoost text/embedding decategorise step rewrites ``eval_set`` with the ``__MISSING__`` sentinel; that rewrite must not reach the caller's ``fit_params``.

``fit_params`` is shared across every model of a target. A rewritten eval set left behind in it hands the sentinel string to the next
model (HGB / XGB), which cannot convert it to float ("could not convert string to float: '__MISSING__'").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_catboost_text_decategorise_does_not_rewrite_callers_fit_params():
    """Fitting CatBoost with a text-routed category column leaves the caller's ``fit_params`` (and its eval set) untouched."""
    catboost = pytest.importorskip("catboost")
    from mlframe.training._training_loop import _train_model_with_fallback

    rng = np.random.default_rng(0)
    n = 200
    words = pd.Categorical(rng.choice(["a", "b", "c"], size=n))
    X = pd.DataFrame({"num_0": rng.normal(size=n), "txt": pd.Series(words).astype("category")})
    X.loc[::5, "txt"] = np.nan
    y = rng.integers(0, 2, size=n)
    Xv, yv = X.iloc[:50].copy(), y[:50]
    eval_set = [(Xv, yv)]
    fit_params = {"text_features": ["txt"], "eval_set": eval_set}
    snapshot = dict(fit_params)

    model = catboost.CatBoostClassifier(iterations=3, verbose=0, allow_writing_files=False)
    _train_model_with_fallback(model, None, "CatBoostClassifier", X, y, fit_params, verbose=False)

    assert set(fit_params) == set(snapshot)
    assert fit_params["text_features"] == snapshot["text_features"]
    assert fit_params["eval_set"] is eval_set
    assert eval_set[0][0]["txt"].isna().any(), "the caller's validation frame must keep its NaN cells"
