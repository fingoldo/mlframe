"""Regression: ``BorutaShap.transform`` must raise the canonical sklearn
``NotFittedError`` instead of a bare ``AttributeError`` when invoked on
an un-fitted instance.

Pre-fix path (fuzz iter-347, axes cb+lgb regression boruta_shap=True
weight=recency_only):
- The per-weight pre_pipeline path can hand a cloned (un-fitted)
  BorutaShap to ``transform`` directly (cache-hit / state-transfer code
  in ``_pipeline_helpers.py`` does not always replicate ``selected_features_``).
- ``transform`` accessed ``self.selected_features_`` and raised
  ``AttributeError: 'BorutaShap' object has no attribute 'selected_features_'``.
- The whole pre_pipeline_fit_transform branch lost the model (status:
  AttributeError after 0.00s).

Post-fix: ``transform`` checks ``hasattr(self, 'selected_features_')`` and
raises ``NotFittedError`` so the caller's check_is_fitted-aware fallback
(the iter-59 recovery branch in predict.py and ``_pipeline_helpers._is_fitted``)
can react consistently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_transform_on_unfit_borutashap_raises_notfittederror():
    from sklearn.exceptions import NotFittedError
    from mlframe.feature_selection.boruta_shap import BorutaShap

    fresh = BorutaShap(classification=False, n_trials=2, verbose=False)
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    with pytest.raises(NotFittedError):
        fresh.transform(df)


def test_transform_after_fit_works_normally():
    from lightgbm import LGBMRegressor
    from mlframe.feature_selection.boruta_shap import BorutaShap

    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "num0": rng.standard_normal(100),
            "num1": rng.standard_normal(100),
        }
    )
    y = pd.Series(df["num0"] * 1.1 + rng.standard_normal(100) * 0.1)
    selector = BorutaShap(
        model=LGBMRegressor(n_estimators=10, num_leaves=15, verbose=-1),
        importance_measure="shap",
        classification=False,
        n_trials=3,
        sample=False,
        normalize=True,
        verbose=False,
    )
    selector.fit(df, y)
    out = selector.transform(df)
    assert out.shape[0] == df.shape[0]


def test_transform_attribute_error_is_replaced_by_notfittederror():
    """Behavioural pin: the bare AttributeError that pre-fix surfaced in
    fuzz iter-347 must NOT come back. NotFittedError is sklearn's
    canonical signal; downstream callers (check_is_fitted-based gates,
    pre_pipeline NotFittedError-aware fallbacks) rely on it."""
    from sklearn.exceptions import NotFittedError
    from mlframe.feature_selection.boruta_shap import BorutaShap

    fresh = BorutaShap(classification=True, n_trials=2, verbose=False)
    df = pd.DataFrame({"a": [1.0]})
    try:
        fresh.transform(df)
    except NotFittedError:
        pass
    except AttributeError as exc:
        pytest.fail(f"Pre-fix AttributeError leaked back: {exc!r}. transform must raise NotFittedError instead.")
