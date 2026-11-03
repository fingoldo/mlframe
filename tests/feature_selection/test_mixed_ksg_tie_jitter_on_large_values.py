"""The mixed-KSG tie-breaking jitter is scaled to the data, so tied values far from zero no longer collapse the estimate."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._ksg import mixed_ksg_mi


def test_large_offset_discrete_variable_gives_ln3_not_a_blown_up_value():
    """x = y = 1e8 + {0, 1, 2}: the true MI is ln 3 = 1.0986 nats; the absolute 1e-10 jitter vanished at that magnitude and gave ~11.9."""
    rng = np.random.default_rng(0)
    x = 1e8 + rng.integers(0, 3, size=3000).astype(np.float64)
    mi = mixed_ksg_mi(x, x.copy(), k=5, seed=0)
    assert 0.9 < mi < 1.5


def test_small_offset_result_is_unchanged_in_kind():
    """The same variable near zero gives the same answer (negative control)."""
    rng = np.random.default_rng(0)
    x = rng.integers(0, 3, size=3000).astype(np.float64)
    assert 0.9 < mixed_ksg_mi(x, x.copy(), k=5, seed=0) < 1.5
