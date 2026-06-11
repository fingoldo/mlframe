"""``BorutaShap.fit`` must accept a polars DataFrame: internally we Arrow-bridge it to a pandas view so the rest of the algorithm (pandas-only ``.apply`` / ``.drop(inplace=True)``) keeps working without touching the caller's frame.

Asserts:
  * no AttributeError on polars input
  * sklearn-style ``support_`` boolean mask is set
  * ``selected_features_`` returns the kept-column list
"""

from __future__ import annotations

import numpy as np
import pytest


def test_boruta_shap_accepts_polars_dataframe():
    pl = pytest.importorskip("polars")
    pytest.importorskip("shap")
    from mlframe.feature_selection.boruta_shap import BorutaShap

    rng = np.random.default_rng(0)
    n = 200
    x_inf = rng.normal(size=n)
    x_noise = rng.normal(size=n)
    y = (x_inf > 0).astype(np.int64)

    X_pl = pl.DataFrame({"inf": x_inf, "noise": x_noise})
    y_s = pl.Series("y", y)

    sel = BorutaShap(
        importance_measure="gini",
        classification=True,
        n_trials=5,
        random_state=0,
        verbose=False,
    )
    sel.fit(X_pl, y_s)

    assert hasattr(sel, "support_"), "BorutaShap.fit must set sklearn-style ``support_`` mask"
    assert sel.support_.dtype == bool
    assert sel.support_.shape == (2,)
    assert hasattr(sel, "selected_features_"), "BorutaShap.fit must set ``selected_features_`` list"
    assert isinstance(sel.selected_features_, list)
    # Every entry must be one of the original input column names.
    assert set(sel.selected_features_).issubset({"inf", "noise"})
