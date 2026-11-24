"""Cluster 17 regression: a degenerate calibration report's ICE is the uncomputable sentinel, not 1.0 (audit 2026-11-20, finding 17-F9).

A real model's ICE can exceed 1.0, so a placeholder of 1.0 could beat it. The degenerate report and the empty-input ICE now use
ICE_UNCOMPUTABLE (1e6, already defined for this purpose), which stays finite as the report contract requires.
"""

import numpy as np

from mlframe.metrics.calibration._calibration_metrics import ICE_UNCOMPUTABLE
from mlframe.metrics.classification._classification_report import fast_ice_only


def test_ice_sentinel_is_the_uncomputable_constant():
    """The sentinel is far above any realistic ICE and finite (17-F9)."""
    assert ICE_UNCOMPUTABLE == 1e6
    assert np.isfinite(ICE_UNCOMPUTABLE)


def test_empty_input_ice_is_the_sentinel():
    """fast_ice_only on empty input returns the sentinel, not 1.0 (17-F9)."""
    assert fast_ice_only(np.array([], dtype=float), np.array([], dtype=float)) == ICE_UNCOMPUTABLE
