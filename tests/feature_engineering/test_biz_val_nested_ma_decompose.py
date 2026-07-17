"""biz_value test for ``feature_engineering.nested_ma_decompose.nested_ma_decompose``.

Source: av_top3_rampaging_datahulk_minihack2017.md -- ``MA_last_10_3 = (Ten_Day_MA*10 - Three_Day_MA*3)/7``,
recovering "the average of the seven days preceding the last three days" algebraically from two already-
computed moving averages, instead of a third rolling pass over the raw series. This test confirms the
algebraic decomposition is numerically identical to directly computing the exclusive window's average from
raw data (the actual claimed win: same result, one fewer rolling computation).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mlframe.feature_engineering.nested_ma_decompose import nested_ma_decompose, nested_ma_decompose_chain


def test_biz_val_nested_ma_decompose_matches_direct_exclusive_window_average():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n).cumsum() + 100

    window_short, window_long = 3, 10
    ma_short = pd.Series(x).rolling(window_short).mean().to_numpy()
    ma_long = pd.Series(x).rolling(window_long).mean().to_numpy()

    decomposed = nested_ma_decompose(ma_short, ma_long, window_short, window_long)

    true_exclusive = np.full(n, np.nan)
    for i in range(window_long - 1, n):
        true_exclusive[i] = x[i - window_long + 1 : i - window_short + 1].mean()

    valid = ~np.isnan(decomposed) & ~np.isnan(true_exclusive)
    assert valid.sum() > 100
    np.testing.assert_allclose(decomposed[valid], true_exclusive[valid], atol=1e-10)


def test_nested_ma_decompose_rejects_non_nested_windows():
    import pytest

    with pytest.raises(ValueError):
        nested_ma_decompose(np.array([1.0]), np.array([1.0]), window_short=10, window_long=5)


def test_nested_ma_decompose_simple_hand_computed_case():
    # MA(10)=5.0 (sum=50), MA(3)=6.0 (sum=18) -> exclusive 7-day sum = 50-18=32, avg = 32/7.
    result = nested_ma_decompose(np.array([6.0]), np.array([5.0]), window_short=3, window_long=10)
    np.testing.assert_allclose(result, [32.0 / 7.0])


def _windowed_mas(x: np.ndarray, windows: list) -> list:
    s = pd.Series(x)
    return [s.rolling(w).mean().to_numpy() for w in windows]


def test_nested_ma_decompose_chain_bit_identical_to_pairwise_calls():
    """The opt-in chained path must be bit-identical to k-1 separate pairwise calls, not just numerically close."""
    rng = np.random.default_rng(1)
    n = 500
    x = rng.normal(size=n).cumsum() + 100
    windows = [3, 10, 20, 45]
    mas = _windowed_mas(x, windows)

    chained = nested_ma_decompose_chain(mas, windows)

    pairwise = [nested_ma_decompose(mas[i], mas[i + 1], windows[i], windows[i + 1]) for i in range(len(windows) - 1)]

    assert len(chained) == len(pairwise) == len(windows) - 1
    for c, p in zip(chained, pairwise):
        np.testing.assert_array_equal(c, p)  # exact bit-identity, same arithmetic order per pair


def test_nested_ma_decompose_chain_rejects_non_increasing_windows():
    import pytest

    with pytest.raises(ValueError):
        nested_ma_decompose_chain([np.array([1.0]), np.array([1.0])], [10, 5])


def test_nested_ma_decompose_chain_rejects_mismatched_lengths():
    import pytest

    with pytest.raises(ValueError):
        nested_ma_decompose_chain([np.array([1.0])], [3, 10])


def test_biz_val_nested_ma_decompose_chain_speedup_over_pairwise_calls():
    """Chained multi-window decomposition must genuinely outperform k-1 separate pairwise calls (paired A/B).

    Realistic FE regime: many small MA windows (e.g. a per-symbol short-lookback ladder) rather than one huge
    array -- here the win is real (Python/numpy per-call dispatch overhead amortized across the whole ladder
    in a single vectorized pass) whereas on one giant array the memory-bandwidth-bound arithmetic dominates
    and the two paths are a wash (measured separately; see bench_nested_ma_decompose.py for both regimes).
    """
    rng = np.random.default_rng(2)
    n = 50
    x = rng.normal(size=n).cumsum() + 100
    windows = list(range(3, 3 + 3 * 30, 3))  # 30-rung ladder -> 29 pairwise calls vs. 1 chained call
    mas = _windowed_mas(x, windows)
    pairs = list(zip(windows[:-1], windows[1:]))
    mas_pairs = list(zip(mas[:-1], mas[1:]))

    inner_reps = 300  # amortize Windows' ~15.6ms process_time() tick resolution over many calls per trial

    def run_pairwise() -> None:
        for _ in range(inner_reps):
            for (ma_short, ma_long), (w_short, w_long) in zip(mas_pairs, pairs):
                nested_ma_decompose(ma_short, ma_long, w_short, w_long)

    def run_chain() -> None:
        for _ in range(inner_reps):
            nested_ma_decompose_chain(mas, windows)

    # Warm up both paths before timing.
    run_pairwise()
    run_chain()

    n_trials = 15
    pairwise_times = []
    chain_times = []
    for _ in range(n_trials):
        t0 = time.process_time()
        run_pairwise()
        t1 = time.process_time()
        run_chain()
        t2 = time.process_time()
        pairwise_times.append(t1 - t0)
        chain_times.append(t2 - t1)

    pairwise_median = float(np.median(pairwise_times))
    chain_median = float(np.median(chain_times))

    assert chain_median < pairwise_median * 0.7, (
        f"expected chained decomposition to beat pairwise calls by >=30%, got chain_median={chain_median:.6f}s vs pairwise_median={pairwise_median:.6f}s"
    )
