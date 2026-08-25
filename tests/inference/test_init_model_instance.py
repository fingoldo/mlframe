"""Coverage for inference.explainability.init_model_instance, previously untested (0% of the file's
non-heavy-SHAP code was exercised). Only this function is targeted here -- compute_shap_on_cv needs a
real fitted model + shap explainer and is out of scope for a cheap smoke test."""

from __future__ import annotations

import pytest

from mlframe.inference.explainability import init_model_instance


class _DummyEstimator:
    """Minimal stand-in for a real estimator: records the constructor kwargs."""

    def __init__(self, **kwargs):
        """Record constructor kwargs."""
        self.kwargs = kwargs


def test_init_model_instance_plain_class():
    """Non-Pipeline model_class: instantiated directly with params."""
    inst = init_model_instance(_DummyEstimator, {"a": 1, "b": "x"})
    assert isinstance(inst, _DummyEstimator)
    assert inst.kwargs == {"a": 1, "b": "x"}


def test_init_model_instance_imblearn_pipeline():
    """Pipeline model_class: only the "est" step is (re)initialized with params; other steps pass through unchanged."""
    pytest.importorskip("imblearn")
    from imblearn.pipeline import Pipeline

    class _Passthrough:
        """A no-op pipeline step."""

        def fit(self, X, y=None):
            """No-op fit."""
            return self

        def transform(self, X):
            """No-op transform."""
            return X

    pipe = Pipeline([("prep", _Passthrough()), ("est", _DummyEstimator)])
    out = init_model_instance(pipe, {"c": 3})
    assert isinstance(out, Pipeline)
    assert out.steps[0][0] == "prep"
    assert out.steps[0][1] is pipe.steps[0][1]  # untouched
    assert out.steps[1][0] == "est"
    assert isinstance(out.steps[1][1], _DummyEstimator)
    assert out.steps[1][1].kwargs == {"c": 3}


def test_init_model_instance_pipeline_without_est_step_raises():
    """A Pipeline with no step named "est" cannot be (re)initialized -- explicit ValueError, not a silent no-op."""
    pytest.importorskip("imblearn")
    from imblearn.pipeline import Pipeline

    class _Passthrough:
        """A no-op pipeline step."""

        def fit(self, X, y=None):
            """No-op fit."""
            return self

        def transform(self, X):
            """No-op transform."""
            return X

    pipe = Pipeline([("prep", _Passthrough()), ("clf", _Passthrough())])
    with pytest.raises(ValueError, match="est"):
        init_model_instance(pipe, {})
