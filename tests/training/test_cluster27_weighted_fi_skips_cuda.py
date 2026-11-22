"""Audit 2026-11-20 finding 27-F7: a weighted fit's permutation importances never use the unweighted CUDA scorer.

The CUDA batched kernel scores with an unweighted R2. With sample_weight given, the non-native importance path must go to the weighted
CPU permutation. The test patches the CUDA kernel to fail the test if called and checks the weighted fallback runs.
"""

import numpy as np

from mlframe.training import _feature_importances as fi


class _Net:
    """A stand-in network object: the CUDA branch is gated on it being present."""


class _Model:
    """A stand-in estimator with no native importances."""


def test_weighted_fit_does_not_call_the_unweighted_cuda_kernel(monkeypatch):
    """With sample_weight set, the CUDA kernel is not called on either branch."""
    calls = []

    def cuda_kernel(*args, **kwargs):
        """Record a call to the unweighted CUDA kernel."""
        calls.append(1)
        return None, None

    def cpu_permutation(model, X, y, return_std=True, sample_weight=None):
        """Record that the weighted CPU fallback received the weights."""
        calls.append(("cpu", sample_weight is not None))
        return np.zeros(X.shape[1]), np.zeros(X.shape[1])

    monkeypatch.setattr(fi, "_cuda_batched_permutation_importance", cuda_kernel)
    monkeypatch.setattr(fi, "_permutation_feature_importances", cpu_permutation)
    monkeypatch.setattr(fi, "_captum_integrated_gradients_importance", lambda net, X: None)
    X = np.random.default_rng(0).standard_normal((30, 60))
    y = np.zeros(30)
    w = np.ones(30)
    fi._non_native_importances(_Model(), X, y, "permutation_cuda", w)
    assert 1 not in calls
    assert ("cpu", True) in calls
