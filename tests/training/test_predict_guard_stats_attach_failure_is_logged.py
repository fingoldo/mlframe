"""A model that refuses the train-fit NaN statistics is reported (tracker 2026-11-02 wave 9, EXC9-4)."""

import logging

import numpy as np

from mlframe.training._predict_guards import prime_nan_guard_stats


class _Slotted:
    """Model with __slots__ and no room for extra attributes."""

    __slots__ = ()


def test_prime_on_a_slotted_model_warns(caplog):
    """The guard cannot store the imputer and scaler; the log names the model type."""
    with caplog.at_level(logging.WARNING):
        prime_nan_guard_stats(_Slotted(), np.random.default_rng(0).normal(size=(40, 3)))
    assert any("_Slotted" in r.getMessage() for r in caplog.records)
