"""Cluster 21 regression: the trend fit cache does not serve a stale line after an in-place refill (audit 2026-11-20, finding 21-F13).

The cache key is the arrays' identity. An array refilled in place keeps its id and its weakref, so the old endpoints came back for new data.
The entry now carries a digest of the contents, checked on every hit.
"""

import numpy as np

from mlframe.reporting.renderers import _trend


def test_inplace_refill_is_refitted_not_served_stale():
    """Refilling the same buffer with new data gives the endpoints of the new data, not the cached ones (21-F13)."""
    x = np.linspace(0.0, 10.0, 50)
    y = 2.0 * x + 1.0
    first = _trend.robust_fit_endpoints(x, y, "huber")
    y[:] = -3.0 * x + 5.0  # same buffer, new contents
    second = _trend.robust_fit_endpoints(x, y, "huber")
    assert first is not None and second is not None
    assert not np.allclose(first[0][1], second[0][1])


def test_unchanged_arrays_still_hit_the_cache():
    """Control: the same unchanged arrays return the same cached result object (21-F13)."""
    x = np.linspace(0.0, 10.0, 50)
    y = 2.0 * x + 1.0
    a = _trend.robust_fit_endpoints(x, y, "huber")
    b = _trend.robust_fit_endpoints(x, y, "huber")
    assert a == b
