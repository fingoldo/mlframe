"""Audit 2026-11-20 finding 35-F22: the auto gate's redundancy test uses residual correlation, as its protocol defines.

Two accurate predictors share the target signal: their predictions correlate near 1, but their errors are independent. Measured on
residuals they are not redundant, so the cheaper NNLS gate fires. Measured on predictions they looked redundant and routed to Shapley.
"""

import numpy as np

from mlframe.training.composite.ensemble import stacking


def test_independent_errors_route_to_nnls_even_when_predictions_correlate():
    """Predictions correlate near 1 but residuals are independent: the gate picks the NNLS path."""
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.normal(size=n) * 10.0
    p1 = y + rng.normal(scale=0.5, size=n)
    p2 = y + rng.normal(scale=0.5, size=n)
    assert np.corrcoef(p1, p2)[0, 1] > 0.99
    _survivors, _weights, method = stacking.auto_aware_gate({"a": p1, "b": p2}, y)
    assert method == "nnls"
