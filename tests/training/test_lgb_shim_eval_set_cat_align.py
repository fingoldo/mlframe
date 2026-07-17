"""Regression: the LGB dataset-reuse shim aligns the eval_set frame's category dtype to train.

LightGBM's categorical_feature="auto" re-detects categoricals from each frame's pandas dtypes independently, so a
column that is category-dtype in train X but a DIFFERENT CategoricalDtype (different categories, or object) in the
eval_set frame raised "train and valid dataset categorical_feature do not match" at fit (fuzz c0125). The shim now
casts the (small) eval_set frame's categorical columns to the train frame's exact CategoricalDtype before building
the val Dataset -- the train frame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
from mlframe.training.lgb_shim import LGBMRegressorWithDatasetReuse


def test_lgb_shim_fit_aligns_eval_set_cat_dtype():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "cat": pd.Categorical(rng.choice(list("abc"), n), categories=list("abc")),
        }
    )
    y = rng.normal(size=n)

    nv = 80
    # eval_set 'cat' carries a DIFFERENT category set (only a,b) -> pre-fix raised "categorical_feature do not match".
    X_val = pd.DataFrame(
        {
            "num": rng.normal(size=nv),
            "cat": pd.Categorical(rng.choice(list("ab"), nv), categories=list("ab")),
        }
    )
    y_val = rng.normal(size=nv)

    model = LGBMRegressorWithDatasetReuse(n_estimators=5, verbose=-1)
    model.fit(X, y, eval_set=[(X_val, y_val)])  # must not raise
    assert model.booster_ is not None
    # the train frame must be untouched (no in-place cat mutation)
    assert list(X["cat"].cat.categories) == list("abc")
