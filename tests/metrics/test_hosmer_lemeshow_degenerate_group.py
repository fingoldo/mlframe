"""A group predicted at probability 0 that still has positives must raise the statistic, not vanish from it (tracker 2026-11-02 wave 8, NAN8-3)."""

import numpy as np

from mlframe.metrics.classification._classification_calibration import hosmer_lemeshow_test


def test_zero_probability_group_with_positives_is_not_skipped():
    """100 rows scored exactly 0.0 with 60 positives make the model clearly miscalibrated: p-value near zero."""
    rng = np.random.default_rng(0)
    p = np.concatenate([np.zeros(100), rng.uniform(0.1, 0.9, 900)])
    y = np.concatenate([(np.arange(100) < 60).astype(int), (rng.uniform(size=900) < p[100:]).astype(int)])
    _, pval, _ = hosmer_lemeshow_test(y, p, n_groups=10)
    assert pval < 1e-6
