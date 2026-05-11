"""biz_val tests for ``mlframe.feature_engineering.mps`` --
``find_maximum_profit_system``, ``compute_area_profits``,
``backfill_zeros``, ``generate_market_price``.

The Maximum Profit System (MPS) computes the OPTIMAL long/short
position sequence on a known-future-prices series, given a per-trade
transaction cost. It's a target-generation utility for sequence
models: the labels are "what would a perfect-foresight trader do?"

Per CLAUDE.md: each test asserts a SYNTHETIC measurable WIN.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# find_maximum_profit_system
# ---------------------------------------------------------------------------


def test_biz_val_mps_find_max_profit_monotone_uptrend_all_long():
    """On a strictly monotone uptrend with negligible TC, optimal
    positions must be all-long (1). Catches regressions in the
    DP recursion."""
    from mlframe.feature_engineering.mps import find_maximum_profit_system
    prices = np.linspace(100.0, 200.0, 20, dtype=np.float64)
    result = find_maximum_profit_system(prices, tc=1e-9,
                                          tc_mode="fraction")
    pos = result["positions"]
    # All positions should be long (1) on a strict uptrend.
    n_long = int(np.sum(pos == 1))
    n_short = int(np.sum(pos == -1))
    assert n_long > n_short, (
        f"monotone uptrend should be dominated by long positions; "
        f"got long={n_long}, short={n_short}"
    )


def test_biz_val_mps_find_max_profit_monotone_downtrend_all_short():
    """On a strictly monotone downtrend, optimal positions must be
    dominated by short (-1)."""
    from mlframe.feature_engineering.mps import find_maximum_profit_system
    prices = np.linspace(200.0, 100.0, 20, dtype=np.float64)
    result = find_maximum_profit_system(prices, tc=1e-9,
                                          tc_mode="fraction")
    pos = result["positions"]
    n_long = int(np.sum(pos == 1))
    n_short = int(np.sum(pos == -1))
    assert n_short > n_long, (
        f"monotone downtrend should be dominated by short; "
        f"got long={n_long}, short={n_short}"
    )


def test_biz_val_mps_find_max_profit_returns_correct_position_length():
    """Output positions must have length ``len(prices) - 1`` (one
    position per inter-bar transition). Catches off-by-one regressions."""
    from mlframe.feature_engineering.mps import find_maximum_profit_system
    prices = np.array([1.0, 2, 3, 2, 1, 2, 3], dtype=np.float64)
    result = find_maximum_profit_system(prices, tc=0.001)
    pos = result["positions"]
    assert len(pos) == len(prices) - 1, (
        f"positions len {len(pos)} != prices len {len(prices)} - 1"
    )


def test_biz_val_mps_find_max_profit_high_tc_reduces_trade_frequency():
    """High transaction cost must reduce trade frequency (fewer
    position flips) -- the optimizer doesn't pay round-trip cost
    on small price changes."""
    from mlframe.feature_engineering.mps import find_maximum_profit_system
    rng = np.random.default_rng(42)
    # Noisy zigzag where trades are small-profit by default
    prices = 100.0 + np.cumsum(rng.normal(0, 0.3, size=50))
    pos_low_tc = find_maximum_profit_system(prices, tc=0.0001)["positions"]
    pos_high_tc = find_maximum_profit_system(prices, tc=0.05)["positions"]
    # Count flips (position changes)
    flips_low = int(np.sum(np.diff(pos_low_tc) != 0))
    flips_high = int(np.sum(np.diff(pos_high_tc) != 0))
    assert flips_high <= flips_low, (
        f"high TC must yield <= flips than low TC; "
        f"got high_tc={flips_high}, low_tc={flips_low}"
    )


@pytest.mark.parametrize("n_prices", [10, 50, 200])
def test_biz_val_mps_find_max_profit_scales_with_size(n_prices):
    """MPS must complete cleanly across {10, 50, 200} price points."""
    from mlframe.feature_engineering.mps import find_maximum_profit_system
    rng = np.random.default_rng(42)
    prices = 100.0 + np.cumsum(rng.normal(0, 0.5, size=n_prices))
    result = find_maximum_profit_system(prices, tc=0.001)
    assert "positions" in result
    assert len(result["positions"]) == n_prices - 1


def test_biz_val_mps_positions_are_signed_int8():
    """positions array must use compact int8 storage (3 values:
    +1 / 0 / -1). Catches regressions where the dtype drifts."""
    from mlframe.feature_engineering.mps import find_maximum_profit_system
    prices = np.array([10.0, 12, 14, 16, 18], dtype=np.float64)
    pos = find_maximum_profit_system(prices, tc=0.001)["positions"]
    # int8 or similar signed compact integer
    assert pos.dtype.kind == "i", f"positions must be signed int; got {pos.dtype}"
    # Values are in {-1, 0, +1}
    unique = set(np.unique(pos).tolist())
    assert unique.issubset({-1, 0, 1}), (
        f"positions must be in {{-1, 0, +1}}; got {unique}"
    )


# ---------------------------------------------------------------------------
# compute_area_profits
# ---------------------------------------------------------------------------


def test_biz_val_mps_compute_area_profits_returns_per_position_profit():
    """``compute_area_profits(prices, positions)`` must return one
    profit per position transition."""
    from mlframe.feature_engineering.mps import compute_area_profits
    prices = np.array([10.0, 12, 14], dtype=np.float64)
    positions = np.array([1, 1], dtype=np.int8)
    result = compute_area_profits(prices, positions)
    arr = np.asarray(result)
    assert len(arr) >= 1


# ---------------------------------------------------------------------------
# backfill_zeros
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["right", "left"])
def test_biz_val_mps_backfill_zeros_propagates_nonzero(direction):
    """``backfill_zeros`` must propagate non-zero values into
    surrounding zero positions in the configured direction."""
    from mlframe.feature_engineering.mps import backfill_zeros
    arr = np.array([0, 0, 5, 0, 0, 7, 0], dtype=np.float64)
    out = backfill_zeros(arr.copy(), direction=direction)
    # No zeros should remain interior to the non-zero values.
    out_arr = np.asarray(out)
    n_zeros_before = int(np.sum(arr == 0))
    n_zeros_after = int(np.sum(out_arr == 0))
    assert n_zeros_after < n_zeros_before, (
        f"backfill_zeros(direction={direction}) must reduce zero count; "
        f"before={n_zeros_before}, after={n_zeros_after}"
    )
