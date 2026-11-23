"""Cluster 13 regression: BorutaShap.Train_model retries without verbose only for a verbose-keyword rejection (audit 2026-11-20, finding 13-F17).

A TypeError raised inside fit used to trigger the retry and be hidden. Now only a TypeError naming 'verbose' is retried; any other
TypeError propagates.
"""

import numpy as np
import pytest

from mlframe.feature_selection.boruta_shap import BorutaShap


class _RejectsVerbose:
    """Estimator whose fit does not accept verbose; the retry should succeed."""

    def fit(self, X, y, **kwargs):
        """Fit only when no verbose keyword is passed."""
        if "verbose" in kwargs:
            raise TypeError("fit() got an unexpected keyword argument 'verbose'")
        self.fitted_ = True
        return self


class _InternalTypeError:
    """Estimator whose fit fails internally with an unrelated TypeError."""

    def fit(self, X, y, **kwargs):
        """Raise a TypeError that has nothing to do with the verbose keyword."""
        raise TypeError("bad operand type inside fit")


def _selector(model):
    """A BorutaShap wrapping the given model, with the attributes Train_model reads."""
    selector = BorutaShap.__new__(BorutaShap)
    selector.model_ = model
    selector.X_categorical_ = None
    selector.fit_params = None
    return selector


def test_verbose_rejection_is_retried():
    """An estimator that rejects verbose is refit without it (13-F17 control)."""
    X = np.zeros((4, 2))
    y = np.zeros(4)
    model = _RejectsVerbose()
    _selector(model).Train_model(X, y)
    assert model.fitted_ is True


def test_internal_type_error_is_not_retried_or_hidden():
    """A TypeError raised inside fit surfaces instead of being retried and swallowed (13-F17)."""
    X = np.zeros((4, 2))
    y = np.zeros(4)
    with pytest.raises(TypeError, match="bad operand"):
        _selector(_InternalTypeError()).Train_model(X, y)
