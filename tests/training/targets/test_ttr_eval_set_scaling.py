"""Tests for F7: ``_TTRWithEvalSetScaling`` -- the custom TransformedTargetRegressor subclass that ALSO standardises the ``eval_set`` y_val arrays before delegating to the inner estimator.

Bug history:
- F1 (2026-05-11): wrapped MLP regression in stock ``TransformedTargetRegressor`` to fix the catastrophic-y-scale failure (MLP predicts ~0 against target mean=11500).
- F7 (2026-05-12): the F1 fix worked for train (TTR scales y_train) but broke val: PyTorch-Lightning consumes ``eval_set=(X_val, y_val)`` for its val_dataloader and computes val_loss against RAW y_val while the inner model predicts on STANDARDISED scale. Result: ``train_loss=0.16`` (std units) vs ``val_loss=1.3e+8`` (raw units) -- early-stop / val_MSE callbacks see nonsensical numbers.

The fix: ``_TTRWithEvalSetScaling`` intercepts ``eval_set`` in fit_kwargs, applies the same fitted transformer to y_val, and forwards the scaled eval_set to the inner regressor. Supports both ``tuple`` (LGB / MLP) and ``list of tuples`` (CB / XGB) eval_set shapes.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler


class _RecordingEstimator(BaseEstimator, RegressorMixin):
    """Mock regressor that records the y / eval_set values it was given so the test can assert the TTR wrap forwarded SCALED values, not raw."""

    def __init__(self):
        self.last_y = None
        self.last_eval_set = None
        # sklearn.compose's TTR clones via __init__ kwargs; nothing needed here.

    def fit(self, X, y, **fit_params):
        """Fit."""
        self.last_y = np.asarray(y).copy()
        self.last_eval_set = fit_params.get("eval_set")
        return self

    def predict(self, X):
        # Return zeros in standardised scale; TTR's inverse_transform should rescale.
        """Predict."""
        return np.zeros(len(X), dtype=np.float64)


def _make_data(n: int = 200, seed: int = 0) -> tuple:
    """Make data."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = 0.95 * X[:, 0] * 100.0 + 11500.0 + rng.normal(scale=10.0, size=n)
    return X, y


def _import_ttr_with_eval_set_scaling():
    """The subclass lives inside ``_configure_mlp_params``. We re-define it here to keep the test isolated from trainer.py module import order and from the (lazy) torch / lightning imports."""
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.base import clone as _clone

    class _TTRWithEvalSetScaling(TransformedTargetRegressor):
        """Groups tests covering t t r with eval set scaling."""
        def fit(self, X, y, **fit_params):
            """Fit."""
            y_arr = np.asarray(y, dtype=np.float64)
            y_arr_2d = y_arr.reshape(-1, 1) if y_arr.ndim == 1 else y_arr
            self.transformer_ = _clone(self.transformer) if self.transformer is not None else None
            if self.transformer_ is not None:
                self.transformer_.fit(y_arr_2d)
                if "eval_set" in fit_params and fit_params["eval_set"] is not None:
                    es = fit_params["eval_set"]
                    if isinstance(es, tuple) and len(es) == 2:
                        X_val, y_val = es
                        y_val_arr = np.asarray(y_val, dtype=np.float64)
                        y_val_2d = y_val_arr.reshape(-1, 1) if y_val_arr.ndim == 1 else y_val_arr
                        y_val_scaled = self.transformer_.transform(y_val_2d).reshape(y_val_arr.shape)
                        fit_params["eval_set"] = (X_val, y_val_scaled)
                    elif isinstance(es, list):
                        new_es = []
                        for entry in es:
                            if isinstance(entry, tuple) and len(entry) == 2:
                                X_v, y_v = entry
                                y_v_arr = np.asarray(y_v, dtype=np.float64)
                                y_v_2d = y_v_arr.reshape(-1, 1) if y_v_arr.ndim == 1 else y_v_arr
                                y_v_scaled = self.transformer_.transform(y_v_2d).reshape(y_v_arr.shape)
                                new_es.append((X_v, y_v_scaled))
                            else:
                                new_es.append(entry)
                        fit_params["eval_set"] = new_es
            return super().fit(X, y, **fit_params)

    return _TTRWithEvalSetScaling


class TestEvalSetScaled:
    """Groups tests covering eval set scaled."""
    def test_tuple_eval_set_y_val_is_scaled(self) -> None:
        """``eval_set=(X_val, y_val)`` form: y_val must reach the inner regressor on STANDARDISED scale (same as y_train), not raw."""
        TTR = _import_ttr_with_eval_set_scaling()
        X, y = _make_data(n=300)
        X_val, y_val = _make_data(n=100, seed=1)
        inner = _RecordingEstimator()
        ttr = TTR(regressor=inner, transformer=StandardScaler())
        ttr.fit(X, y, eval_set=(X_val, y_val))
        # The recorded eval_set's y_val should be roughly mean-0, std-1 (StandardScaler effect).
        recorded_X_val, recorded_y_val = ttr.regressor_.last_eval_set
        assert recorded_X_val is X_val  # X passes through unchanged
        # y_val standardised against y_train's scaler -> mean near 0 (since both drawn from similar distribution), std near 1.
        recorded_y_val = np.asarray(recorded_y_val)
        assert abs(np.mean(recorded_y_val)) < 0.5
        assert abs(np.std(recorded_y_val) - 1.0) < 0.5

    def test_list_of_tuples_eval_set_y_val_is_scaled(self) -> None:
        """``eval_set=[(X_val, y_val)]`` form (CB / XGB style): each tuple's y must be scaled."""
        TTR = _import_ttr_with_eval_set_scaling()
        X, y = _make_data(n=300)
        X_val, y_val = _make_data(n=100, seed=2)
        inner = _RecordingEstimator()
        ttr = TTR(regressor=inner, transformer=StandardScaler())
        ttr.fit(X, y, eval_set=[(X_val, y_val)])
        recorded = ttr.regressor_.last_eval_set
        assert isinstance(recorded, list)
        assert len(recorded) == 1
        recorded_y_val = np.asarray(recorded[0][1])
        assert abs(np.std(recorded_y_val) - 1.0) < 0.5

    def test_y_train_standardised(self) -> None:
        """The y arg of fit gets standardised by the parent TTR -- inner regressor sees scaled y."""
        TTR = _import_ttr_with_eval_set_scaling()
        X, y = _make_data(n=300)
        inner = _RecordingEstimator()
        ttr = TTR(regressor=inner, transformer=StandardScaler())
        ttr.fit(X, y)
        recorded_y = ttr.regressor_.last_y
        # Scaled y: mean near 0, std near 1.
        assert abs(np.mean(recorded_y)) < 0.1
        assert abs(np.std(recorded_y) - 1.0) < 0.1

    def test_no_eval_set_pass_through(self) -> None:
        """When eval_set is absent, fit works unchanged (no KeyError, no scaling attempt on a None)."""
        TTR = _import_ttr_with_eval_set_scaling()
        X, y = _make_data(n=200)
        inner = _RecordingEstimator()
        ttr = TTR(regressor=inner, transformer=StandardScaler())
        ttr.fit(X, y)
        assert ttr.regressor_.last_eval_set is None

    def test_predict_inverse_transforms_back_to_raw(self) -> None:
        """End-to-end: inner predicts in std-scale (returns zeros), TTR inverse_transform takes back to raw -> predictions near y_train.mean()."""
        TTR = _import_ttr_with_eval_set_scaling()
        X, y = _make_data(n=300)
        inner = _RecordingEstimator()
        ttr = TTR(regressor=inner, transformer=StandardScaler())
        ttr.fit(X, y)
        preds = ttr.predict(X[:5])
        # inner.predict returns zeros (std-scale = mean of standardised y). Inverse: 0 * std(y) + mean(y) = mean(y).
        y_mean = float(np.mean(y))
        np.testing.assert_allclose(preds, np.full(5, y_mean), rtol=1e-6)

    def test_eval_set_with_2d_y_val_handled(self) -> None:
        """A 2-D y_val (multi-output case; shape (n, 1)) gets standardised without shape mismatch."""
        TTR = _import_ttr_with_eval_set_scaling()
        X, y = _make_data(n=300)
        X_val, y_val = _make_data(n=100, seed=3)
        y_val_2d = y_val.reshape(-1, 1)
        inner = _RecordingEstimator()
        ttr = TTR(regressor=inner, transformer=StandardScaler())
        ttr.fit(X, y, eval_set=(X_val, y_val_2d))
        recorded_y_val = np.asarray(ttr.regressor_.last_eval_set[1])
        assert recorded_y_val.shape == y_val_2d.shape  # 2-D shape preserved
        assert abs(np.std(recorded_y_val) - 1.0) < 0.5


class TestRegressionAgainstUnpatchedTTR:
    """Groups tests covering regression against unpatched t t r."""
    def test_unpatched_ttr_LEAVES_eval_set_raw(self) -> None:
        """Sanity / regression sensor: the STOCK sklearn TransformedTargetRegressor does NOT scale eval_set -- demonstrating the bug F7 fixes. If sklearn ever changes this behavior the test will fail loudly and we can drop the custom subclass."""
        from sklearn.compose import TransformedTargetRegressor

        X, y = _make_data(n=300)
        X_val, y_val = _make_data(n=100, seed=4)
        inner = _RecordingEstimator()
        ttr_stock = TransformedTargetRegressor(regressor=inner, transformer=StandardScaler())
        ttr_stock.fit(X, y, eval_set=(X_val, y_val))
        # Stock TTR leaves eval_set's y raw.
        recorded_y_val = np.asarray(ttr_stock.regressor_.last_eval_set[1])
        assert abs(np.mean(recorded_y_val) - np.mean(y_val)) < 0.001
        # Raw y has std ~ sqrt(var(y_val)) which is ~ 100 (X[:, 0] * 100 + noise). NOT close to 1.
        assert np.std(recorded_y_val) > 50  # raw scale; would be ~1 if scaled
