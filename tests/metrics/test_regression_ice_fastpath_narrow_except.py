"""Regression: the batched-ICE fastpath must not mask unexpected errors.

Pre-fix a broad ``except Exception`` swallowed ANY error from ``_batch_per_class_ice_kernel`` and silently fell
back to the slow per-class loop at DEBUG, hiding genuine numeric/logic bugs. Post-fix only (ValueError, TypeError)
fall back; any other exception type propagates.
"""

import numpy as np
import pytest

import mlframe.metrics._ice_metric as ice


def _binary_inputs():
    rng = np.random.default_rng(1)
    n = 200
    y_true = (rng.random(n) > 0.5).astype(np.int64)
    y_score = rng.random(n)
    return y_true, y_score


def test_unexpected_error_type_propagates(monkeypatch):
    y_true, y_score = _binary_inputs()

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic kernel failure")

    monkeypatch.setattr(ice, "_batch_per_class_ice_kernel", _boom)
    with pytest.raises(RuntimeError, match="synthetic kernel failure"):
        ice.compute_probabilistic_multiclass_error(y_true, y_score, method="multicrit")


def test_value_error_still_falls_back(monkeypatch):
    y_true, y_score = _binary_inputs()

    def _value_err(*args, **kwargs):
        raise ValueError("recoverable shape/dtype issue")

    monkeypatch.setattr(ice, "_batch_per_class_ice_kernel", _value_err)
    # Falls back to the slow per-class loop and returns a finite scalar (no raise).
    res = ice.compute_probabilistic_multiclass_error(y_true, y_score, method="multicrit")
    assert np.isfinite(float(res))
