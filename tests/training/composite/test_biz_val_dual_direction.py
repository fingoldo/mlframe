"""biz_value test for ``training.composite.dual_direction.DualDirectionCompositeEstimator``.

Synthetic: ``y = shape(x1) * scale(x2) + noise`` -- a multiplicative interaction between two feature groups
(the LANL "shape-normalized target * scale prediction" pattern). A single linear model predicting ``y`` directly
from ``[x1, x2]`` cannot represent a multiplicative interaction (Ridge only fits additive/linear combinations of
its inputs), so it is systematically biased; the dual-direction estimator factors the problem into two additive
sub-problems (predict ``scale`` from ``x2`` directly; predict the ratio-transformed shape from ``x1``), each of
which Ridge CAN represent, recovering the product cleanly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mlframe.training.composite.dual_direction import DualDirectionCompositeEstimator


def _make_shape_scale_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    shape = np.clip(0.1 + 0.8 * x1, 0, 1)  # linear in x1, so Ridge can fit the shape sub-problem cleanly --
    # isolates the test to the MULTIPLICATIVE-interaction failure mode alone, not an additional nonlinearity
    # a single linear baseline model would ALSO fail at even without the shape*scale coupling.
    scale = 10.0 + 20.0 * x2  # magnitude, linear in x2
    y = shape * scale + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2})
    return df, y, scale


def test_biz_val_dual_direction_beats_single_linear_model_on_multiplicative_target():
    df, y, scale = _make_shape_scale_dataset(n=3000, seed=0)
    df_train, df_test, y_train, y_test, scale_train, _ = train_test_split(df, y, scale, test_size=0.3, random_state=0)

    baseline = Ridge().fit(df_train, y_train)
    rmse_baseline = float(mean_squared_error(y_test, baseline.predict(df_test)) ** 0.5)

    dual = DualDirectionCompositeEstimator(scale_estimator=Ridge(), shape_estimator=Ridge(), n_splits=5, random_state=0)
    dual.fit(df_train, y_train, scale_train)
    rmse_dual = float(mean_squared_error(y_test, dual.predict(df_test)) ** 0.5)

    assert rmse_dual < rmse_baseline * 0.5, f"expected the dual-direction estimator to cut RMSE by >=50% vs a single linear model on the multiplicative target, got dual={rmse_dual:.4f} baseline={rmse_baseline:.4f}"


def test_dual_direction_predict_scale_returns_reasonable_scale_estimate():
    df, y, scale = _make_shape_scale_dataset(n=2000, seed=1)
    df_train, df_test, y_train, _, scale_train, scale_test = train_test_split(df, y, scale, test_size=0.3, random_state=1)

    dual = DualDirectionCompositeEstimator(scale_estimator=Ridge(), shape_estimator=Ridge(), n_splits=5, random_state=1)
    dual.fit(df_train, y_train, scale_train)
    scale_pred = dual.predict_scale(df_test)

    rmse_scale = float(mean_squared_error(scale_test, scale_pred) ** 0.5)
    assert rmse_scale < 2.0, f"expected the scale sub-model to recover the linear scale relationship reasonably well, got rmse={rmse_scale:.4f}"


def test_dual_direction_predict_before_fit_raises():
    import pytest

    df, _, _ = _make_shape_scale_dataset(n=10, seed=2)
    dual = DualDirectionCompositeEstimator(scale_estimator=Ridge(), shape_estimator=Ridge())
    with pytest.raises(ValueError):
        dual.predict(df)


def test_dual_direction_rejects_misaligned_inputs():
    import pytest

    df, y, scale = _make_shape_scale_dataset(n=50, seed=3)
    dual = DualDirectionCompositeEstimator(scale_estimator=Ridge(), shape_estimator=Ridge())
    with pytest.raises(ValueError):
        dual.fit(df, y[:-5], scale)
