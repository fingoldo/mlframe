"""Cluster 39 regression: CustomHandler.fit retries unsupervised only on a y-signature mismatch (audit 2026-11-20, finding 39-F22)."""

import numpy as np
import pandas as pd
import pytest

from mlframe.training.feature_handling.custom_handler import CustomHandler, CustomParams


class _UnsupervisedOnly:
    """Transformer whose fit takes X only, so passing y is a signature mismatch."""

    def fit(self, X):
        """Fit without a target."""
        return self

    def transform(self, X):
        """Identity transform."""
        return np.asarray(X)


class _InternalTypeError:
    """Supervised transformer that hits a genuine TypeError inside fit."""

    def fit(self, X, y=None):
        """Fail with an unrelated TypeError."""
        raise TypeError("unsupported operand type(s) for +: 'int' and 'str'")

    def transform(self, X):
        """Identity transform."""
        return np.asarray(X)


def _handler(transformer):
    """Build a CustomHandler around ``transformer`` for column ``a``."""
    return CustomHandler("a", CustomParams(transformer=transformer))


def test_genuine_internal_type_error_surfaces():
    """A real TypeError inside a supervised fit is raised, not silently refit unsupervised (39-F22)."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(TypeError, match="unsupported operand"):
        _handler(_InternalTypeError()).fit(df, np.array([0, 1, 0]))


def test_signature_mismatch_still_retries_unsupervised():
    """Control: a fit(X)-only transformer is still fitted unsupervised via the retry (39-F22)."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    h = _handler(_UnsupervisedOnly())
    h.fit(df, np.array([0, 1, 0]))
    assert np.asarray(h.transform(df)).shape[0] == 3
