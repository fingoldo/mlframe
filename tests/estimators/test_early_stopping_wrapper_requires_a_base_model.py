"""``EarlyStoppingWrapper`` without a base model raises ``ValueError`` with a clear message, not sklearn's clone error.

A ``None`` base reached ``clone`` and failed with "Cannot clone object 'None'", which does not say what is missing (audit API3-14).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.estimators.early_stopping import EarlyStoppingWrapper


def test_fit_without_a_base_model_names_the_missing_parameter():
    """A None base model is reported as a missing required parameter."""
    X = np.random.default_rng(0).normal(size=(40, 2))
    y = X[:, 0]
    with pytest.raises(ValueError, match="base_model is required"):
        EarlyStoppingWrapper(base_model=None).fit(X, y)
