"""TRAINING_COMPOSITE_CORE_A-8 (2026-08-05 audit): ``sklearn_set_params`` must reject an unrecognized
parameter name with a ``ValueError``, matching sklearn's own ``BaseEstimator.set_params`` contract --
not silently accept and apply any keyword via a bare ``setattr`` loop, which swallows a typo'd kwarg.
"""

from __future__ import annotations

import pytest

from mlframe.training.composite.qrf import _LeafResidualForest


def test_unknown_param_raises_value_error():
    """A typo'd/unknown kwarg to set_params must raise ValueError naming the bad parameter."""
    est = _LeafResidualForest()
    with pytest.raises(ValueError, match="totally_not_a_real_param"):
        est.set_params(totally_not_a_real_param=123)


def test_known_param_still_applies():
    """A real parameter must still be applied (no regression on the happy path)."""
    est = _LeafResidualForest()
    est.set_params(n_estimators=50)
    assert est.n_estimators == 50
