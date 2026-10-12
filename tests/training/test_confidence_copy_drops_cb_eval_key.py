"""The confidence regressor's fit parameters must not carry the main CatBoost fit's internal eval-weight key.

The suite stores that key in the fit parameters for the main CatBoost model. The confidence model is a plain CatBoostRegressor,
which rejects it as an unexpected keyword argument, so a combo with confidence analysis enabled failed in its fit.
"""

from __future__ import annotations

from mlframe.training._confidence_analysis import _copy_confidence_fit_params
from mlframe.training.cb._cb_eval_weights import CB_EVAL_WEIGHTS_KEY


def test_confidence_fit_params_drop_the_cb_eval_weight_key():
    """The copied fit parameters keep the caller's ordinary keys and lose the CatBoost eval-weight carrier."""
    fit_params = {CB_EVAL_WEIGHTS_KEY: [1.0, 2.0], "cat_features": ["c"]}

    copied = _copy_confidence_fit_params(fit_params, {})

    assert CB_EVAL_WEIGHTS_KEY not in copied
    assert copied["cat_features"] == ["c"]
