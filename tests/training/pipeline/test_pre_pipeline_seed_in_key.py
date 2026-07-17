"""Regression sensor for A5 P2 #11.

Pre-fix _pipeline_signature_for_cache built the signature only from ``step.get_params(deep=False)``. Two structurally identical pipelines with different ``random_state`` values DID hit the cache slot under that key when ``get_params`` surfaced the seed identically; defence-in-depth was missing for the attribute-only-seed case (some custom transformers set ``random_state`` post-init as an instance attribute). The fix also folds attribute-only seeds so collision is avoided.
"""

from __future__ import annotations

from mlframe.training.pipeline._pipeline_cache import _pipeline_signature_for_cache


class _SimpleStep:
    """Minimal sklearn-shaped step where get_params returns a fixed kwargs dict that does NOT advertise random_state, but the attribute is set on the instance (replicates the rare custom-transformer pattern)."""

    def __init__(self, with_mean: bool = True):
        self.with_mean = with_mean
        # set later by external caller as a side-effect

    def get_params(self, deep: bool = False):
        return {"with_mean": self.with_mean}


class _ToyPipeline:
    """Pipeline-shaped fake so the signature builder walks ``.steps``."""

    def __init__(self, steps):
        self.steps = steps


def test_attribute_only_random_state_changes_signature():
    a = _SimpleStep()
    a.random_state = 11
    b = _SimpleStep()
    b.random_state = 42
    pa = _ToyPipeline([("scaler", a)])
    pb = _ToyPipeline([("scaler", b)])
    sig_a = _pipeline_signature_for_cache(pa)
    sig_b = _pipeline_signature_for_cache(pb)
    assert sig_a != sig_b, f"Differing random_state attributes must produce different cache signatures (got identical: {sig_a!r})"


def test_no_random_state_signatures_remain_stable():
    a = _SimpleStep(with_mean=True)
    b = _SimpleStep(with_mean=True)
    sig_a = _pipeline_signature_for_cache(_ToyPipeline([("s", a)]))
    sig_b = _pipeline_signature_for_cache(_ToyPipeline([("s", b)]))
    assert sig_a == sig_b, "Identical pipelines without seeds must still share the cache slot"
