"""The CPU softmax kernel keeps its guards for non-finite logits (full fastmath lets the compiler assume they never occur)."""
import numpy as np

from mlframe.feature_engineering.transformer._kernels_njit import softmax_stable_inplace


def test_all_minus_inf_logits_fall_back_to_uniform():
    """No reachable neighbour gives all -inf logits; the documented result is uniform 1/k, not nan."""
    logits = np.full(5, -np.inf, dtype=np.float32)
    softmax_stable_inplace(logits)
    assert np.allclose(logits, 0.2)


def test_a_nan_logit_does_not_leak_nan_into_the_other_weights_check_is_uniform():
    """A nan max is non-finite, so the guard must route the row to uniform."""
    logits = np.array([np.nan, 1.0, 2.0], dtype=np.float32)
    softmax_stable_inplace(logits)
    assert np.isfinite(logits).all() and abs(float(logits.sum()) - 1.0) < 1e-5
