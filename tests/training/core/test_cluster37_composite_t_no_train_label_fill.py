"""Audit 2026-11-20 finding 37-F1: a training row whose base leaves the transform domain gets no fabricated composite label.

Only validation and test rows take the median fill. The training row stays missing, so it is not trained on a made-up value.
"""

import numpy as np
from types import SimpleNamespace

from mlframe.training.core._phase_composite_discovery import composite_t_full


class _Diff:
    """A transform defined where the base is positive; forward is y minus base."""

    def domain_check(self, y, base):
        """The domain holds where base is positive."""
        return np.asarray(base) > 0

    def forward(self, y, base, params):
        """Residual of y over base."""
        return np.asarray(y, dtype=np.float64) - np.asarray(base, dtype=np.float64)


def test_out_of_domain_train_row_stays_missing_and_non_train_row_is_filled():
    """Row 0 is a training row with base 0 (outside the domain): it stays NaN. Row 3 is a test row with base 0: it gets the median."""
    y = np.array([5.0, 2.0, 3.0, 4.0, 1.0])
    base = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    spec = SimpleNamespace(fitted_params={})
    t_full, t_raw = composite_t_full(_Diff(), spec, y, base, np.array([0, 1, 2]))
    assert np.isnan(t_full[0])
    assert np.isfinite(t_full[3])
    assert np.isnan(t_raw[3])
