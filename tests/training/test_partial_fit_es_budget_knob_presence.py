"""A model whose category names a budget knob the object does not carry must not be wrapped for dichotomic early stopping.

The category table maps ``ridge`` to ``max_iter``. A Pipeline around a Ridge has no ``max_iter`` of its own, so wrapping it used to
pass the caller's check and then fail inside the wrapper with "has no partial_fit and no budget_param", which failed whole fuzz
combos. The category may only name a knob the object actually exposes.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.training._data_helpers_early_stopping import _detect_budget_param, maybe_wrap_for_partial_fit_es


def _val_split() -> tuple[np.ndarray, np.ndarray]:
    """Small regression validation split shared by the wrap calls."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(40, 3)), rng.normal(size=40)


def test_pipeline_around_ridge_is_not_wrapped_for_budget_search():
    """A Pipeline categorised as ridge exposes no ``max_iter``, so it is left unwrapped rather than refused later."""
    X_val, y_val = _val_split()
    pipe = make_pipeline(StandardScaler(), Ridge())

    model, was_wrapped = maybe_wrap_for_partial_fit_es(pipe, model_category="ridge", X_val=X_val, y_val=y_val, is_classification=False)

    assert was_wrapped is False
    assert model is pipe


def test_bare_ridge_still_gets_the_max_iter_budget_knob():
    """A bare Ridge carries ``max_iter`` and keeps the category's budget knob."""
    assert _detect_budget_param("ridge", Ridge()) == "max_iter"
