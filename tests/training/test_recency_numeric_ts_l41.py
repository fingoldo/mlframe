"""Regression: ``get_sample_weights_by_recency`` must accept numeric
(int / float) ts columns as well as pandas datetime Series.

Pre-fix path (1M-row harness ``_profile_fuzz_1m`` with add_ts=True):
1. The harness emits a strictly-monotonic-increasing ``ts`` column as
   ``int64`` epoch-seconds.
2. The FTE's ``get_weights`` method routes that column to
   ``get_sample_weights_by_recency``.
3. The function computed ``(date_series.max() - date_series.min()).total_seconds()``.
   For int64 input, ``max - min`` is a numeric scalar (no
   ``.total_seconds()`` method) and raises
   ``AttributeError: 'int' object has no attribute 'total_seconds'``.
4. The whole suite aborted before model fit even started.

Post-fix: dtype-kind detection at function entry routes numeric ts
through a separate path that treats the raw difference as
already-seconds; datetime ts keeps its Timedelta-aware path
unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.extractors import get_sample_weights_by_recency


def test_numeric_int_ts_returns_finite_weights() -> None:
    """int64 epoch-seconds ts (the 1M harness's format) must produce
    finite weights without crashing."""
    n = 1000
    _start = 1_700_000_000
    ts = pd.Series(_start + np.arange(n, dtype=np.int64), name="ts")
    weights = get_sample_weights_by_recency(ts)
    assert weights.shape == (n,)
    assert np.all(np.isfinite(weights))
    # Most recent row gets the smallest weight if drop is positive — actually
    # the function's contract is "more recent = higher weight". Verify by
    # checking the last row's weight exceeds the first row's weight.
    assert weights[-1] > weights[0]


def test_numeric_float_ts_returns_finite_weights() -> None:
    """float64 ts (alternative numeric form) must also work."""
    n = 1000
    ts = pd.Series(np.linspace(0.0, 86400.0 * 365, n), name="ts")
    weights = get_sample_weights_by_recency(ts)
    assert weights.shape == (n,)
    assert np.all(np.isfinite(weights))


def test_datetime_ts_path_unchanged() -> None:
    """Baseline: datetime Series continues to work via Timedelta path."""
    n = 100
    ts = pd.Series(pd.date_range("2026-01-01", periods=n, freq="1h"), name="ts")
    weights = get_sample_weights_by_recency(ts)
    assert weights.shape == (n,)
    assert np.all(np.isfinite(weights))
    assert weights[-1] > weights[0]


def test_numeric_ts_zero_span_returns_uniform() -> None:
    """All-equal numeric ts falls back to uniform min_weight."""
    n = 50
    ts = pd.Series(np.full(n, 1_700_000_000, dtype=np.int64), name="ts")
    weights = get_sample_weights_by_recency(ts, min_weight=2.5)
    assert weights.shape == (n,)
    assert np.allclose(weights, 2.5)


def test_pre_fix_simulated_AttributeError() -> None:
    """Lock the bug surface: directly call the int Series's
    ``(max - min).total_seconds()`` to confirm Python raises
    AttributeError. If pandas ever ships .total_seconds() on int
    scalars this sensor catches it."""
    n = 10
    ts = pd.Series(1_700_000_000 + np.arange(n, dtype=np.int64), name="ts")
    diff = ts.max() - ts.min()
    assert isinstance(diff, (int, np.integer))
    with pytest.raises(AttributeError, match="total_seconds"):
        diff.total_seconds()
