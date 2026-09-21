"""Ctrl+C during a fit must stop the suite, not read as a model failure.

CatBoost catches whatever a python-side custom metric raises and re-raises it as ``CatBoostError``, so an interrupt
inside ``ICE.evaluate`` reached the trainer as an ordinary model error: with ``continue_on_model_failure`` the suite
moved on to the next model instead of stopping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training._training_loop import _train_model_with_fallback_unguarded


class _Boom:
    """Stand-in estimator whose fit raises the CatBoost-wrapped interrupt text."""

    def __init__(self, message: str):
        self._message = message

    def fit(self, *args, **kwargs):
        raise RuntimeError(self._message)


_CB_WRAPPED = (
    "catboost/python-package/catboost/helpers.cpp:58: Traceback (most recent call last):\n"
    '  File "_catboost.pyx", line 1614, in _catboost._MetricEval\n'
    '  File "mlframe/metrics/_ice_metric.py", line 401, in evaluate\n'
    "KeyboardInterrupt\n"
)


def _fit(message):
    X = pd.DataFrame({"a": np.arange(10.0)})
    return _train_model_with_fallback_unguarded(_Boom(message), _Boom(message), "CatBoostClassifier", X, np.arange(10.0), {})


def test_wrapped_interrupt_becomes_keyboard_interrupt():
    with pytest.raises(KeyboardInterrupt):
        _fit(_CB_WRAPPED)


def test_other_errors_are_not_turned_into_interrupts():
    with pytest.raises(Exception) as exc:
        _fit("some unrelated CatBoostError about a bad parameter")
    assert not isinstance(exc.value, KeyboardInterrupt)
