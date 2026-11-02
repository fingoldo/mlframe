"""EarlyStoppingWrapper refuses settings that used to return an unfitted-looking result (tracker 2026-11-02, VAL6-6)."""

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from mlframe.estimators.early_stopping import EarlyStoppingWrapper


def _data(n=200, seed=0):
    """A small separable binary problem."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    return X, (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"max_iter": 0}, "max_iter"),
        ({"max_iter": 5, "start_iter": 0}, "start_iter"),
        ({"max_iter": 5, "patience": 0}, "patience"),
        ({"max_iter": 5, "max_runtime_mins": 0}, "max_runtime_mins"),
        ({"max_iter": 5, "max_runtime_mins": -1.0}, "max_runtime_mins"),
        ({"max_iter": 5, "validation_fraction": 0.0}, "validation_fraction"),
        ({"max_iter": 5, "validation_fraction": 1.0}, "validation_fraction"),
    ],
)
def test_degenerate_settings_raise_a_named_value_error(kwargs, match):
    """Each degenerate setting raises before any fitting, naming the parameter."""
    X, y = _data()
    wrapper = EarlyStoppingWrapper(GradientBoostingClassifier(random_state=0), **kwargs)
    with pytest.raises(ValueError, match=match):
        wrapper.fit(X, y)


def test_a_sensible_configuration_still_fits():
    """The validation does not reject a normal setting."""
    X, y = _data()
    wrapper = EarlyStoppingWrapper(GradientBoostingClassifier(random_state=0), max_iter=5, patience=2, warm_start_step=1)
    wrapper.fit(X, y)
    assert wrapper.predict(X).shape == (200,)
