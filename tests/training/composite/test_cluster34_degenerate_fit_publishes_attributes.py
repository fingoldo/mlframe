"""Audit 2026-11-20 finding 34-F18: a degenerate fit (too few rows) still sets the bookkeeping attributes.

Before the fix the early return skipped them, so reading ``elapsed_seconds_`` after such a fit raised AttributeError.
"""

import numpy as np
import pandas as pd

from mlframe.training.composite.discovery import CompositeTargetDiscovery
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def test_fit_with_too_few_rows_sets_bookkeeping_attributes():
    """After a 30-row fit (under the 50-row floor) elapsed_seconds_, val_idx_ and test_idx_ are readable and empty."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.normal(size=30), "x2": rng.normal(size=30), "y": rng.normal(size=30)})
    disc = CompositeTargetDiscovery(CompositeTargetDiscoveryConfig())
    disc.fit(df, "y", ["x1", "x2"], np.arange(30))
    assert disc.specs_ == []
    assert disc.elapsed_seconds_ == 0.0
    assert disc.val_idx_ is None
    assert disc.test_idx_ is None
