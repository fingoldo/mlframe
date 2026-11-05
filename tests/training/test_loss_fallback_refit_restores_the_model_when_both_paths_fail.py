"""A rejected loss-fallback refit leaves the model exactly as it was, not with the fallback loss half applied.

``set_params`` changed the loss before ``fit`` ran. When ``fit`` and the fresh-instance rebuild both failed, the function returned None
while the model still carried the fallback loss and metric, so saved metadata and later refits described a booster that never trained.
"""

from __future__ import annotations

import logging

from mlframe.training._training_loop_refit import _maybe_refit_on_degenerate_best_iter


class _RejectingBooster:
    """Accepts ``set_params``, rejects ``fit``, and refuses to be rebuilt with new parameters."""

    def __init__(self, **params):
        """Refuse every construction: the fresh-instance path must fail too."""
        raise ValueError("rejected by the backend")

    def get_params(self, deep: bool = True) -> dict:
        """Current parameters."""
        return dict(self.params)

    def set_params(self, **kwargs) -> "_RejectingBooster":
        """Apply parameters in place, as a real estimator does."""
        self.params.update(kwargs)
        return self

    def fit(self, X, y, **kwargs):
        """Reject the fit."""
        raise RuntimeError("degenerate loss cannot be fitted")


def _model():
    """A LightGBM-style regressor on a Huber loss with a 500-iteration budget."""
    m = object.__new__(_RejectingBooster)
    m.params = {"objective": "huber", "n_estimators": 500}
    return m


def test_failed_fallback_refit_restores_the_original_loss():
    """After both refit paths fail, the model reports its original objective and budget, and the call returns None."""
    m = _model()
    out = _maybe_refit_on_degenerate_best_iter(
        model_obj=m,
        model_type_name="LGBMRegressor",
        best_iter=1,
        train_df=[[0.0]],
        train_target=[0.0],
        fit_params={},
        logger_=logging.getLogger("test"),
    )
    assert out is None
    assert m.get_params()["objective"] == "huber"
    assert m.get_params()["n_estimators"] == 500
