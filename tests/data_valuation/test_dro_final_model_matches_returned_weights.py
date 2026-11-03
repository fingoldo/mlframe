"""dro_reweight_fit returns a model trained on the weights it returns, as its docstring promises."""

from __future__ import annotations

import numpy as np

from mlframe.data_valuation._adversarial_reweighting import dro_reweight_fit


def test_returned_model_was_fit_on_the_returned_weights():
    """fit_fn records the weights of its last call: they must equal the returned final weights (they were the previous round's)."""
    seen = {}

    class _M:
        """Stand-in model."""

        def predict(self, X):
            """Constant predictor."""
            return np.zeros(len(X))

    def fit_fn(X, y, w):
        """Remember the sample weights of the latest fit."""
        seen["w"] = np.asarray(w).copy()
        return _M()

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = (X[:, 0] > 0).astype(float)
    _model, w_final, _info = dro_reweight_fit(fit_fn, lambda yt, yp: (yt - yp) ** 2, X, y, n_rounds=2, rng=np.random.default_rng(1))
    np.testing.assert_allclose(seen["w"], w_final)
