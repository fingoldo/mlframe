"""A composite target stays missing where its raw target is missing.

Composite discovery applied the train-fitted transform to every row and filled every non-finite T with the train
median. Rows whose raw label was missing (y is NaN, so T is NaN) got that median too: a made-up composite label that
the per-target loop then trained and scored on. The fill is meant only for rows where y exists but the base column
violates the transform's domain.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlframe.training.composite import get_transform
from mlframe.training.core._phase_composite_discovery import composite_t_full


def test_missing_y_stays_missing_while_a_domain_violation_is_filled():
    """A missing label stays missing in the composite target; only a row outside the transform's domain is filled."""
    rng = np.random.default_rng(0)
    n = 200
    base = rng.uniform(1.0, 5.0, n)
    y = base * rng.uniform(1.0, 2.0, n)
    y[10:20] = np.nan  # missing labels
    base[30] = -1.0  # y exists, but log(y / base) is outside the domain
    transform = get_transform("logratio")
    spec = SimpleNamespace(fitted_params=transform.fit(y[np.isfinite(y)], base[np.isfinite(y)]))
    t_full, t_raw = composite_t_full(transform, spec, y, base, np.arange(n))
    assert np.isnan(t_full[10:20]).all(), "a missing raw label must not get a made-up composite label"
    assert np.isfinite(t_full[30]) and np.isnan(t_raw[30]), "a domain violation with a real label is still filled"
    assert np.isfinite(np.delete(t_full, np.arange(10, 20))).all()
