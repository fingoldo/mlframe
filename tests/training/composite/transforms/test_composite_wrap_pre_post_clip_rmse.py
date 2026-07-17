"""Pre/post-clip RMSE pair contract: post-hoc ``y_clip_low/high`` is a no-op on train (in-envelope by construction),
so a single headline RMSE that compares the wrapped (clipped) prediction to y_train is OPTIMISTICALLY EQUAL TO the
un-clipped one - the clip never had a chance to "improve" the train metric. On val / test rows that drift outside the
train envelope the clip narrows max-error, so wrapped < raw there. Capturing both makes the clip's contribution visible
instead of folding the no-op train case into a falsely "improved" number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import CompositeTargetEstimator


class _NumpyAdapter:
    """Inner stand-in that accepts a DataFrame and forwards to an sklearn estimator trained on ``[x, base]`` columns."""

    def __init__(self, sklearn_est, feature_cols: list[str]) -> None:
        self.sklearn_est = sklearn_est
        self.feature_cols = feature_cols

    def predict(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return self.sklearn_est.predict(X[self.feature_cols].to_numpy())
        return self.sklearn_est.predict(np.asarray(X))


def test_train_raw_equals_wrapped_val_wrapped_better_than_raw() -> None:
    """Clip is a no-op on train (in-envelope by construction) - ``train_rmse_raw == train_rmse_wrapped`` exactly.

    On val the linear extrapolation onto out-of-envelope base values produces predictions outside ``[10, 100]``; the clip
    saturates them at the envelope edges and narrows RMSE - ``val_rmse_wrapped < val_rmse_raw``.
    """
    rng = np.random.default_rng(0)
    n_train = 400
    x_train = rng.uniform(0, 10, size=n_train)
    base_train = rng.uniform(50, 60, size=n_train)
    y_train = np.clip(20 + 3 * x_train + rng.normal(scale=2.0, size=n_train), 10.0, 100.0)
    t_train = y_train - base_train
    X_train_df = pd.DataFrame({"x": x_train, "base": base_train})
    sk = LinearRegression().fit(X_train_df[["x"]].to_numpy(), t_train)
    inner = _NumpyAdapter(sk, ["x"])

    wrapper = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=inner,
        transform_name="diff",
        base_column="base",
        transform_fitted_params={},
        y_train=y_train,
        fallback_predict="y_train_median",
        base_columns=("base",),
    )

    # Train: clip bounds derived from y_train -> in-envelope -> wrapped == raw on train.
    y_pred_train_wrapped = wrapper.predict(X_train_df)
    y_pred_train_raw = wrapper.predict_pre_clip(X_train_df)
    np.testing.assert_allclose(y_pred_train_wrapped, y_pred_train_raw)
    train_rmse_raw = float(np.sqrt(np.mean((y_pred_train_raw - y_train) ** 2)))
    train_rmse_wrapped = float(np.sqrt(np.mean((y_pred_train_wrapped - y_train) ** 2)))
    assert train_rmse_raw == train_rmse_wrapped, "post-hoc clip cannot improve train RMSE; in-envelope by construction"

    # Val: extreme out-of-envelope rows -> linear extrapolates wild, wrapped clip narrows.
    n_val = 200
    x_val_in = rng.uniform(0, 10, size=n_val // 2)
    base_val_in = rng.uniform(50, 60, size=n_val // 2)
    y_val_in = np.clip(20 + 3 * x_val_in + rng.normal(scale=2.0, size=n_val // 2), 10.0, 100.0)
    # Bases must overshoot the actual fitted clip bound. The wrapper extends the train [q001, q999] envelope by safety factors
    # (~0.1 lower, ~10x upper) so the effective high bound on this train (y in [10, 100]) lands near ~910. Push extreme bases to
    # 5_000 - 10_000 so y_hat = T_pred + base = (small) + (5_000..10_000) decisively breaches the clip ceiling.
    x_val_ex = rng.uniform(20, 30, size=n_val // 2)
    base_val_ex = rng.uniform(5_000, 10_000, size=n_val // 2)
    # Inject true y values near the clip endpoints so wrapped (clipped to [low, high]) is closer than raw (~5_000-10_000).
    y_val_ex = np.concatenate([np.full(n_val // 4, 90.0), np.full(n_val // 4, 20.0)])
    x_val = np.concatenate([x_val_in, x_val_ex])
    base_val = np.concatenate([base_val_in, base_val_ex])
    y_val = np.concatenate([y_val_in, y_val_ex])
    X_val_df = pd.DataFrame({"x": x_val, "base": base_val})

    y_pred_val_wrapped = wrapper.predict(X_val_df)
    y_pred_val_raw = wrapper.predict_pre_clip(X_val_df)
    val_rmse_raw = float(np.sqrt(np.mean((y_pred_val_raw - y_val) ** 2)))
    val_rmse_wrapped = float(np.sqrt(np.mean((y_pred_val_wrapped - y_val) ** 2)))
    # Some rows DID get clipped (otherwise the test setup is broken).
    assert not np.allclose(y_pred_val_wrapped, y_pred_val_raw), "expected at least one val row to be clipped"
    # Wrapped val RMSE must be strictly smaller because the clip pulls wild predictions back into the envelope and most
    # extreme y_val are themselves close to the envelope (500 -> 100, -200 -> 10).
    assert val_rmse_wrapped < val_rmse_raw, f"expected val_rmse_wrapped({val_rmse_wrapped:.4f}) < val_rmse_raw({val_rmse_raw:.4f}); clip not contributing"


def test_predict_pre_clip_exposed_on_wrapper() -> None:
    """``predict_pre_clip`` must exist and return a numpy array of the right shape - the post-phase code branches on its presence."""
    rng = np.random.default_rng(1)
    y_train = rng.uniform(0, 50, size=100)
    base_train = rng.uniform(10, 20, size=100)
    t_train = y_train - base_train
    X = pd.DataFrame({"x": rng.uniform(size=100), "base": base_train})
    sk = LinearRegression().fit(X[["x"]].to_numpy(), t_train)
    inner = _NumpyAdapter(sk, ["x"])
    wrapper = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=inner,
        transform_name="diff",
        base_column="base",
        transform_fitted_params={},
        y_train=y_train,
        base_columns=("base",),
    )
    assert hasattr(wrapper, "predict_pre_clip")
    out = wrapper.predict_pre_clip(X)
    assert isinstance(out, np.ndarray)
    assert out.shape == (100,)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-xvs", "--no-cov"])
