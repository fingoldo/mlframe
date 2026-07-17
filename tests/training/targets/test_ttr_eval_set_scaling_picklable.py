"""#8: _TTRWithEvalSetScaling must be dill-picklable (module-level class).

Pre-fix the class was defined INSIDE _configure_neural_params in trainer.py. As a local class it carried a hidden closure reference to the enclosing function's namespace; dill could not serialise the ABC-metaclass ``_abc._abc_data`` slot through that closure -- production saves failed with ``cannot pickle '_abc._abc_data' object``. Moving to module level fixes that.
"""

from __future__ import annotations

import io

import numpy as np


class TestTTRWithEvalSetScalingPickle:
    def test_module_level_class_importable(self) -> None:
        """The class lives at module level, not inside a function."""
        from mlframe.training.targets._ttr_eval_set_scaling import _TTRWithEvalSetScaling

        # Module-level class: __qualname__ is just the class name, NOT
        # "_configure_neural_params.<locals>._TTRWithEvalSetScaling" (the
        # pre-fix qualname that broke pickle).
        assert "<locals>" not in _TTRWithEvalSetScaling.__qualname__

    def test_unfit_instance_pickles_via_dill(self) -> None:
        """A constructed but UNFIT instance must dill-roundtrip without raising. Pre-fix this raised ``cannot pickle '_abc._abc_data' object``."""
        import dill  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        from mlframe.training.targets._ttr_eval_set_scaling import _TTRWithEvalSetScaling

        m = _TTRWithEvalSetScaling(regressor=Ridge(alpha=1e-3), transformer=StandardScaler())
        buf = io.BytesIO()
        dill.dump(m, buf)
        buf.seek(0)
        m2 = dill.load(buf)  # nosec B301 -- round-trip of a locally-created, trusted object
        assert isinstance(m2, _TTRWithEvalSetScaling)

    def test_fit_then_pickle_roundtrips_and_predicts(self) -> None:
        """End-to-end: fit on small dataset, dill-roundtrip, predict matches the original."""
        import dill  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        from mlframe.training.targets._ttr_eval_set_scaling import _TTRWithEvalSetScaling

        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (200, 4))
        y = X[:, 0] * 2.0 + 5.0 + rng.normal(0, 0.1, 200)
        m = _TTRWithEvalSetScaling(regressor=Ridge(alpha=1e-3), transformer=StandardScaler())
        m.fit(X, y)
        preds_before = m.predict(X[:10])

        buf = io.BytesIO()
        dill.dump(m, buf)
        buf.seek(0)
        m2 = dill.load(buf)  # nosec B301 -- round-trip of a locally-created, trusted object
        preds_after = m2.predict(X[:10])
        np.testing.assert_allclose(preds_after, preds_before, rtol=1e-9, atol=1e-9)

    def test_eval_set_y_val_gets_scaled(self) -> None:
        """The class's main purpose: scale eval_set's y_val through the transformer so inner estimators see y_val on the same scale as y_train."""
        from sklearn.preprocessing import StandardScaler

        from mlframe.training.targets._ttr_eval_set_scaling import _TTRWithEvalSetScaling

        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (100, 3))
        y = rng.normal(loc=1000.0, scale=50.0, size=100)  # huge mean, so scaling matters
        X_val = rng.normal(0, 1, (20, 3))
        y_val = rng.normal(loc=1000.0, scale=50.0, size=20)

        captured_eval_set = {}

        class _RecordingRegressor:
            def fit(self, X, y, **fit_params):
                captured_eval_set["eval_set"] = fit_params.get("eval_set")
                self.coef_ = np.zeros(X.shape[1])
                self.intercept_ = 0.0
                self._is_fitted = True
                return self

            def predict(self, X):
                return np.zeros(X.shape[0])

            def get_params(self, deep=True):
                return {}

            def set_params(self, **kwargs):
                return self

        # We want a sklearn-clone-compatible regressor; the stock Ridge works
        # for the round-trip but won't record eval_set, so we skip it and
        # just check that the override transforms eval_set values via
        # _RecordingRegressor.
        m = _TTRWithEvalSetScaling(
            regressor=_RecordingRegressor(),
            transformer=StandardScaler(),
        )
        m.fit(X, y, eval_set=(X_val, y_val))
        es = captured_eval_set.get("eval_set")
        assert es is not None
        _, y_val_scaled = es
        # y_val should be on standardised scale (mean ~ 0, std ~ 1) after the override.
        assert abs(float(np.mean(y_val_scaled))) < 1.0
        assert abs(float(np.std(y_val_scaled))) < 5.0
