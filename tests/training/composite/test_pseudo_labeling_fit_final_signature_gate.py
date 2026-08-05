"""TRAINING_COMPOSITE_CORE_B-2 (2026-08-05 audit): PseudoLabelingLoop._fit_final used
``except TypeError: model.fit(X, y)`` to retry without sample_weight -- the exact anti-pattern
post_shim.py (same directory) documents fixing. A TypeError raised DEEP inside a fit that DOES accept
sample_weight (a bad dtype, a shape mismatch, a downstream bug) was mis-attributed to "no sample_weight
support", silently dropping the confirmation-bias down-weighting of pseudo-labeled rows with no warning,
and hiding the real error. Fixed by signature-gating via post_shim.py's
_model_fit_accepts_sample_weight, matching that module's already-established pattern.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.pseudo_labeling import PseudoLabelingLoop


class _SampleWeightAwareButBuggyModel:
    """A model whose fit() DOES accept sample_weight, but raises TypeError from a genuine internal bug
    that only triggers when sample_weight is actually passed (simulating e.g. a dtype/shape bug in the
    weighting code path) -- the exact scenario the catch-all except TypeError mis-handled: the old code
    caught this TypeError, assumed "no sample_weight support", and silently retried unweighted
    (succeeding and masking the bug); the fix must instead propagate it."""

    def fit(self, X, y, sample_weight=None):
        """Raise a TypeError only when sample_weight is actually passed, to simulate a genuine bug in
        the weighting code path rather than a signature mismatch."""
        if sample_weight is not None:
            raise TypeError("simulated downstream bug: bad dtype in the sample_weight handling code, not a signature mismatch")
        self.fitted_unweighted_ = True

    def predict(self, X):
        """Never reached in this test."""
        return np.zeros(len(X))

    def predict_proba(self, X):
        """Never reached in this test."""
        return np.zeros((len(X), 2))


class _NoSampleWeightModel:
    """A model whose fit() genuinely has no sample_weight parameter -- the legitimate fallback case."""

    def fit(self, X, y):
        """Fit without sample_weight support."""
        self.fitted_ = True
        return self

    def predict(self, X):
        """Return a constant prediction."""
        return np.zeros(len(X))


def test_fit_final_propagates_genuine_typeerror_instead_of_silently_dropping_weights():
    """A TypeError from a model that DOES declare sample_weight must propagate (the real bug must be
    visible), not be silently swallowed into an unweighted retry."""
    model = _SampleWeightAwareButBuggyModel()
    X = np.zeros((10, 2))
    y = np.zeros(10)
    sw = np.ones(10)

    with pytest.raises(TypeError, match="simulated downstream bug"):
        PseudoLabelingLoop._fit_final(model, X, y, sw)


def test_fit_final_still_falls_back_when_fit_genuinely_lacks_sample_weight():
    """A model whose fit() genuinely has no sample_weight parameter must still fit successfully
    without it -- the legitimate fallback case must keep working."""
    model = _NoSampleWeightModel()
    X = np.zeros((10, 2))
    y = np.zeros(10)
    sw = np.ones(10)

    PseudoLabelingLoop._fit_final(model, X, y, sw)
    assert model.fitted_ is True
