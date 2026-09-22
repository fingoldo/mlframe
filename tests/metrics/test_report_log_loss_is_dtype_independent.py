"""The calibration report's log loss must not depend on the dtype the model emitted.

`fast_log_loss`'s default clip eps follows `y_pred`'s dtype, so a confidently wrong row cost 15.9 for a model emitting
float32 probabilities and 36.0 for a float64 one: a report comparing the two ranked them partly on dtype.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.classification._classification_report import fast_calibration_report


def test_float32_and_float64_predictions_report_the_same_log_loss():
    rng = np.random.default_rng(0)
    n = 2000
    p = rng.random(n)
    y = (rng.random(n) < p).astype(int)
    p[:3] = 0.0
    y[:3] = 1  # confidently wrong rows: exactly where the dtype-dependent eps diverged
    ll64 = fast_calibration_report(y, p.astype(np.float64), show_plots=False).ll
    ll32 = fast_calibration_report(y, p.astype(np.float32), show_plots=False).ll
    assert ll32 == pytest.approx(ll64, abs=1e-6)
