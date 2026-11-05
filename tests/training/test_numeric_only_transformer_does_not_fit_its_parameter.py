"""``_NumericOnlyTransformer.fit`` fits a clone of its inner transformer, leaving the constructor parameter unfitted.

The inner transformer is a constructor parameter. Fitting it in place made ``get_params()['inner']`` come back fitted, so a caller that reused the
instance, or a ``clone`` built from it, carried the first fit's statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from mlframe.training.strategies.base import _NumericOnlyTransformer


def test_fit_leaves_the_inner_parameter_unfitted():
    """After fit, the inner transformer passed to the constructor is still unfitted; the fitted clone does the work."""
    scaler = StandardScaler()
    X = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "cat": ["a", "b", "a", "b"]})
    t = _NumericOnlyTransformer(inner=scaler, cat_features=["cat"]).fit(X)
    assert not hasattr(scaler, "mean_")
    try:
        check_is_fitted(scaler)
        raised = False
    except Exception:
        raised = True
    assert raised
    assert hasattr(t, "_fitted_inner")


def test_transform_still_scales_the_numeric_block():
    """The numeric column is scaled with the statistics of the fit; the categorical column passes through."""
    X = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "cat": ["a", "b", "a", "b"]})
    out = _NumericOnlyTransformer(inner=StandardScaler(), cat_features=["cat"]).fit(X).transform(X)
    assert np.isclose(out["num"].to_numpy().mean(), 0.0)
    assert list(out["cat"]) == ["a", "b", "a", "b"]
