"""Config classes the suite never reads say so when constructed; FairnessConfig.enabled says it does nothing."""

import warnings

import pytest

from mlframe.training.configs import FairnessConfig, LinearModelConfig


def test_consumed_class_does_not_warn():
    """Consumed class does not warn."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LinearModelConfig()
    assert [w for w in caught if "not read by the training suite" in str(w.message)] == []


def test_fairness_enabled_warns():
    """Fairness enabled warns."""
    with pytest.warns(UserWarning, match="enabled=True has no effect"):
        FairnessConfig(enabled=True)
