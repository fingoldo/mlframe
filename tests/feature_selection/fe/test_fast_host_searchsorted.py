"""The multi-core searchsorted returns exactly ``np.searchsorted(..., side='right')``, NaN and infinities included."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._fast_host_ops import searchsorted_right


@pytest.mark.parametrize("n", [10, 199_999, 200_000, 1_000_003])
@pytest.mark.parametrize("kind", ["normal", "ties", "heavy"])
def test_equals_numpy(n, kind):
    """Same integers, over tie-heavy data where values sit exactly on edges."""
    rng = np.random.default_rng(n % 13)
    x = {"normal": rng.standard_normal(n), "ties": np.round(rng.standard_normal(n), 1), "heavy": rng.standard_normal(n) ** 3}[kind]
    for k in (1, 3, 9, 63):
        edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, k + 2)[1:-1]))
        np.testing.assert_array_equal(searchsorted_right(edges, x), np.searchsorted(edges, x, side="right"))


def test_nan_and_infinities_follow_numpy():
    """NaN lands after every edge; +-inf at the ends."""
    x = np.concatenate([np.array([np.nan, -np.inf, np.inf, 0.0, 1.0, 2.0]), np.random.default_rng(0).standard_normal(300_000)])
    edges = np.array([0.0, 1.0, 2.0])
    np.testing.assert_array_equal(searchsorted_right(edges, x), np.searchsorted(edges, x, side="right"))
    assert searchsorted_right(np.array([]), x).max() == 0


def test_integer_input_falls_back_to_numpy():
    """Non-float columns keep numpy's own path."""
    x = np.random.default_rng(0).integers(0, 50, 300_000)
    edges = np.array([10, 20, 30])
    np.testing.assert_array_equal(searchsorted_right(edges, x), np.searchsorted(edges, x, side="right"))


def test_count_distinct_int_matches_unique_and_content_memo_is_content_keyed():
    """The occupancy count equals ``np.unique(...).size`` on every integer shape; the memo recomputes only for different content and hands out private copies."""
    from mlframe.feature_selection.filters._fast_host_ops import ContentMemo, count_distinct_int

    rng = np.random.default_rng(0)
    for a in (rng.integers(0, 12, 50_000), rng.integers(-5, 5, 1000).astype(np.int8), np.array([2**40, 5, 5]), np.array([], dtype=np.int64), np.array([True, False, True])):
        assert count_distinct_int(a) == np.unique(a).size
    memo, calls = ContentMemo(max_entries=2), []

    def compute(a):
        """Counting compute."""
        calls.append(1)
        return a * 2

    x, y = rng.standard_normal(1000), rng.standard_normal(1000)
    first = memo.get_or_compute("t", x, compute)
    first[:] = 0
    np.testing.assert_array_equal(memo.get_or_compute("t", x.copy(), compute), x * 2)
    assert len(calls) == 1
    memo.get_or_compute("t", y, compute)
    assert len(calls) == 2


def test_content_memo_pickle_round_trip_drops_runtime_state() -> None:
    """A pickled ContentMemo keeps its capacity, restores a usable lock and starts with an empty memo."""
    import pickle

    from mlframe.feature_selection.filters._fast_host_ops import ContentMemo

    memo = ContentMemo(max_entries=3)
    x = np.arange(10, dtype=np.float64)
    memo.get_or_compute("t", x, lambda a: a * 2)
    restored = pickle.loads(pickle.dumps(memo))  # nosec B301 - round-trip of an object this test just pickled
    assert restored._max == 3
    assert len(restored._data) == 0
    np.testing.assert_array_equal(restored.get_or_compute("t", x, lambda a: a * 3), x * 3)
