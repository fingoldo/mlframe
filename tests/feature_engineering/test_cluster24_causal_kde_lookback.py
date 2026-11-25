"""Cluster 24 regression: bounded causal KDE lookback (audit 2026-11-20, finding 24-F17).

``lookback=None`` is the exact full-prefix mode. An int lookback clips each causal prefix to its last L rows; prefixes no longer than L are
unchanged. ``lookback="auto"`` picks a candidate only when its modes match the exact ones on probe rows, and falls back to exact otherwise.
"""

import numpy as np
import pytest

from mlframe.feature_engineering.recency_density import (
    LOOKBACK_CANDIDATES,
    _select_lookback,
    _sort_into_groups,
    per_group_recency_weighted_mode,
)


def _series(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """One long group of a smooth, bounded signal with noise; the values are in a fixed range so tolerances are meaningful."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = np.sin(t / 25.0) + 0.1 * rng.standard_normal(n)
    return values, np.zeros(n, dtype=np.int64)


def test_none_is_exact_and_deterministic():
    """Control: the default is the exact full-prefix mode, the same on every call (24-F17)."""
    v, g = _series()
    a = per_group_recency_weighted_mode(v, g, scheme="exp", param=0.9)
    b = per_group_recency_weighted_mode(v, g, scheme="exp", param=0.9, lookback=None)
    assert np.array_equal(a, b)


def test_lookback_longer_than_every_prefix_is_exact():
    """A lookback at least as long as the series clips nothing, so it equals the exact mode (24-F17)."""
    v, g = _series(n=120)
    exact = per_group_recency_weighted_mode(v, g, scheme="exp", param=0.9)
    bounded = per_group_recency_weighted_mode(v, g, scheme="exp", param=0.9, lookback=500)
    assert np.allclose(exact, bounded, rtol=0, atol=1e-12)


def test_short_lookback_changes_long_prefixes_only():
    """Rows whose prefix is within the lookback match the exact mode; later rows differ and stay finite (24-F17)."""
    v, g = _series(n=200)
    L = 20
    exact = per_group_recency_weighted_mode(v, g, scheme="exp", param=0.9)
    bounded = per_group_recency_weighted_mode(v, g, scheme="exp", param=0.9, lookback=L)
    assert np.isfinite(bounded).all()
    assert np.allclose(exact[:L], bounded[:L], rtol=0, atol=1e-12)
    assert not np.allclose(exact[L + 5 :], bounded[L + 5 :], rtol=0, atol=1e-9)


def test_auto_with_zero_tolerance_falls_back_to_exact():
    """With tolerance 0 no candidate qualifies on noisy data, so auto returns the exact mode (24-F17)."""
    v, g = _series(n=400, seed=3)
    exact = per_group_recency_weighted_mode(v, g, scheme="exp", param=0.9)
    auto = per_group_recency_weighted_mode(v, g, scheme="exp", param=0.9, lookback="auto", lookback_tol=0.0)
    assert np.array_equal(exact, auto)


def test_auto_selects_a_candidate_on_a_long_series():
    """On a long series the selection returns one of the candidates, or None when none qualifies (24-F17)."""
    v, g = _series(n=600, seed=1)
    v_sorted, _sort_idx, starts, ends, _n = _sort_into_groups(v, g, None)
    choice = _select_lookback(v_sorted, starts, ends, 0, 0.9, -1.0, 64, 1e-2)
    assert choice in LOOKBACK_CANDIDATES


def test_lookback_refused_off_the_causal_broadcast_path():
    """A lookback on a non-causal or non-broadcast call raises rather than being silently ignored (24-F17)."""
    v, g = _series(n=50)
    with pytest.raises(ValueError, match="lookback applies only"):
        per_group_recency_weighted_mode(v, g, scheme="exp", param=0.9, causal=False, lookback=10)
    with pytest.raises(ValueError, match="lookback applies only"):
        per_group_recency_weighted_mode(v, g, scheme="exp", param=0.9, broadcast=False, lookback=10)
