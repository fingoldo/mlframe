"""A recommendation constructs only an estimator class defined in mlframe that derives from sklearn's BaseEstimator.

The refusal list checked the name and the presence of ``fit``. Any mlframe class with a ``fit`` method, whatever it is, was still constructed
with the caller's kwargs. The check is now an allow-list: a BaseEstimator subclass defined inside the mlframe package.
"""

from __future__ import annotations

import sys
import types

from mlframe.training.composite._estimator_dispatch import instantiate_recommended_estimator


def test_a_fit_class_that_is_not_a_base_estimator_is_refused(monkeypatch):
    """An mlframe class with ``fit`` but no BaseEstimator base is not constructed."""
    module_name = "mlframe._dispatch_probe_not_estimator"
    probe = types.ModuleType(module_name)

    class NotAnEstimator:
        """A class with a fit method that is not a scikit-learn estimator."""

        def __init__(self, **kwargs):
            """Record construction so the test can tell whether the class was built."""
            self.kwargs = kwargs

        def fit(self, X, y):
            """Return self; never reached by the dispatch check."""
            return self

    NotAnEstimator.__module__ = module_name
    probe.NotAnEstimator = NotAnEstimator
    monkeypatch.setitem(sys.modules, module_name, probe)

    try:
        instantiate_recommended_estimator({"module": module_name, "estimator": "NotAnEstimator"}, sentinel=1)
    except ValueError as exc:
        assert "not an estimator class" in str(exc)
    else:
        raise AssertionError("a non-BaseEstimator class was constructed from a recommendation")
