"""Cluster 20 regression: the panel's interval base rate is the table's subsample base rate (audit 2026-11-20, finding 20-F42)."""

import numpy as np

from mlframe.reporting.charts import category_discriminability as cd


def test_base_rate_uses_the_same_rows_as_the_count_pass(monkeypatch):
    """Above the cap the base is the mean of the sampled rows (differs from the full mean); below it, the full mean."""
    monkeypatch.setattr(cd, "_COUNT_SUBSAMPLE_CAP", 100)
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=1000) < 0.3).astype(float)
    idx = cd.count_subsample_rows(1000, seed=4)
    assert idx is not None and idx.size == 100 and np.all(np.diff(idx) > 0)
    assert cd.count_base_rate(y, seed=4) == float(y[idx].mean())
    assert cd.count_base_rate(y[:80], seed=4) == float(y[:80].mean())
