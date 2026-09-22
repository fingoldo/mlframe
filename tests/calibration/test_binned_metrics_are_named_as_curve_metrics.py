"""Metrics computed on the ~20 binned reliability points must not carry per-sample names.

`estimate_calibration_quality_binned` reported "CRPS" and "R2" evaluated on the binned pockets: two length-20 vectors
turn CRPS into the mean absolute reliability-curve gap, roughly an order of magnitude below a per-sample CRPS on the
same data, shown to the operator under the per-sample name.
"""

from __future__ import annotations

import numpy as np

from mlframe.calibration.quality import estimate_calibration_quality_binned


def test_binned_entries_are_labelled_as_curve_metrics():
    rng = np.random.default_rng(0)
    p = rng.random(2000)
    y = (rng.random(2000) < p).astype(np.float64)
    _, _, _, metrics = estimate_calibration_quality_binned(y, p)
    assert "CRPS" not in metrics and "R2" not in metrics
    assert {"CRPS_curve", "R2_curve", "BR"} <= set(metrics)
