"""Tests for ``composite_predictions_as_feature`` + ``composite_oof_predictions`` (R10c brainstorm #10).

Composite x FE-pipeline stacking: expose a composite-target model's predictions as an engineered feature column on the input dataframe. Two variants:
- ``composite_predictions_as_feature``: attach a fitted wrapper's predictions (in-sample warning -- caller responsibility).
- ``composite_oof_predictions``: K-fold out-of-fold predictions for leakage-free downstream stacking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

lgb = pytest.importorskip("lightgbm")

from mlframe.training.composite import (
    CompositeTargetEstimator,
    composite_oof_predictions,
    composite_predictions_as_feature,
)


def _make_dataset(n: int = 400, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=10.0, scale=2.0, size=n)
    x_other = rng.normal(size=n)
    y = 0.9 * base + 0.3 * x_other + rng.normal(scale=0.2, size=n)
    df = pd.DataFrame({"base": base, "x_other": x_other})
    return df, y


def _fit_wrapper(df: pd.DataFrame, y: np.ndarray) -> CompositeTargetEstimator:
    inner = lgb.LGBMRegressor(n_estimators=30, num_leaves=11, verbose=-1, random_state=0)
    wrapper = CompositeTargetEstimator(
        base_estimator=inner,
        transform_name="linear_residual",
        base_column="base",
    )
    wrapper.fit(df, y)
    return wrapper


# ===========================================================================
# composite_predictions_as_feature
# ===========================================================================


class TestPredictionsAsFeature:
    def test_attaches_column_pandas(self) -> None:
        df, y = _make_dataset()
        wrapper = _fit_wrapper(df, y)
        out = composite_predictions_as_feature(wrapper, df)
        # Default column name derived from wrapper attrs.
        assert "composite_pred__linear_residual__base" in out.columns
        assert len(out) == len(df)
        # Original df not mutated.
        assert "composite_pred__linear_residual__base" not in df.columns

    def test_custom_column_name(self) -> None:
        df, y = _make_dataset()
        wrapper = _fit_wrapper(df, y)
        out = composite_predictions_as_feature(wrapper, df, column_name="my_pred")
        assert "my_pred" in out.columns
        assert out["my_pred"].notna().all()

    def test_finite_predictions(self) -> None:
        df, y = _make_dataset()
        wrapper = _fit_wrapper(df, y)
        out = composite_predictions_as_feature(wrapper, df)
        preds = out["composite_pred__linear_residual__base"].to_numpy()
        assert np.all(np.isfinite(preds))

    def test_fallback_on_predict_failure(self) -> None:
        """When the wrapper's predict fails (e.g. missing base column) and ``fallback_value`` is set, return a column filled with the fallback rather than raising."""
        df, y = _make_dataset()
        wrapper = _fit_wrapper(df, y)
        # Predict on a df without the base column -- wrapper.predict raises.
        df_bad = df.drop(columns=["base"])
        out = composite_predictions_as_feature(
            wrapper, df_bad, column_name="pred", fallback_value=0.0,
        )
        assert "pred" in out.columns
        assert (out["pred"] == 0.0).all()

    def test_fallback_none_reraises(self) -> None:
        df, y = _make_dataset()
        wrapper = _fit_wrapper(df, y)
        df_bad = df.drop(columns=["base"])
        with pytest.raises(KeyError):
            composite_predictions_as_feature(wrapper, df_bad)


# ===========================================================================
# composite_oof_predictions
# ===========================================================================


class TestOOFPredictions:
    def test_shape_matches_input(self) -> None:
        df, y = _make_dataset(n=300)
        def factory():
            inner = lgb.LGBMRegressor(n_estimators=15, num_leaves=7, verbose=-1, random_state=0)
            return CompositeTargetEstimator(
                base_estimator=inner,
                transform_name="linear_residual",
                base_column="base",
            )
        oof = composite_oof_predictions(factory, df, y, n_splits=5, random_state=0)
        assert oof.shape == (len(df),)

    def test_oof_predictions_finite(self) -> None:
        df, y = _make_dataset(n=300)
        def factory():
            inner = lgb.LGBMRegressor(n_estimators=15, num_leaves=7, verbose=-1, random_state=0)
            return CompositeTargetEstimator(
                base_estimator=inner,
                transform_name="linear_residual",
                base_column="base",
            )
        oof = composite_oof_predictions(factory, df, y, n_splits=5, random_state=0)
        # All folds should succeed on this clean DGP.
        assert np.all(np.isfinite(oof))

    def test_oof_rmse_higher_than_in_sample(self) -> None:
        """OOF predictions are honest; in-sample predictions are optimistic. Lock: in-sample RMSE < OOF RMSE (so the OOF feature isn't carrying in-sample leakage)."""
        df, y = _make_dataset(n=400)
        # In-sample wrapper.
        wrapper = _fit_wrapper(df, y)
        in_sample_preds = wrapper.predict(df)
        in_sample_rmse = float(np.sqrt(np.mean((in_sample_preds - y) ** 2)))
        # OOF predictions.
        def factory():
            inner = lgb.LGBMRegressor(n_estimators=30, num_leaves=11, verbose=-1, random_state=0)
            return CompositeTargetEstimator(
                base_estimator=inner,
                transform_name="linear_residual",
                base_column="base",
            )
        oof = composite_oof_predictions(factory, df, y, n_splits=5, random_state=0)
        oof_rmse = float(np.sqrt(np.mean((oof - y) ** 2)))
        assert in_sample_rmse <= oof_rmse, (
            f"in-sample RMSE should be <= OOF RMSE; got in_sample={in_sample_rmse:.4f}, oof={oof_rmse:.4f}"
        )
