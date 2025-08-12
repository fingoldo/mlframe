# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from typing import *  # noqa: F401 pylint: disable=wildcard-import,unused-wildcard-import

import numba
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# INITS
# -----------------------------------------------------------------------------------------------------------------------------------------------------

FASTMATH: bool = False

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# CORE
# -----------------------------------------------------------------------------------------------------------------------------------------------------


@numba.njit(fastmath=FASTMATH)
def _state_index(pos):
    # map position to state index: -1 -> 0, 0 -> 1, +1 -> 2
    if pos == -1:
        return 0
    elif pos == 0:
        return 1
    else:
        return 2


@numba.njit(fastmath=FASTMATH)
def _index_state(idx):
    # inverse mapping
    if idx == 0:
        return -1
    elif idx == 1:
        return 0
    else:
        return 1


@numba.njit(fastmath=FASTMATH)
def _trade_count(prev_pos, new_pos):
    # number of trades executed when switching prev_pos -> new_pos
    # closing prev if prev_pos != 0, opening new if new_pos != 0
    trades = 0
    if prev_pos != 0:
        # closing previous
        trades += 1
    if new_pos != 0:
        # opening new position (if new_pos != prev_pos)
        if prev_pos == new_pos:
            # continuing same position -> no open/close
            pass
        else:
            trades += 1
    return trades


@numba.njit(fastmath=FASTMATH)
def _trade_cost(price_t, trades, tc, tc_mode_is_fraction):
    if trades == 0:
        return 0.0
    if tc_mode_is_fraction:
        return price_t * tc * trades
    else:
        return tc * trades


@numba.njit(fastmath=FASTMATH)
def compute_area_profits(prices, positions):
    n = prices.shape[0]
    profits = np.zeros(n, dtype=prices.dtype)

    # We will find consecutive runs of the same non-zero position,
    # for each run, compute profit = prices[end+1] - prices[start] (or reverse if short)

    # For index i in run, profit[i] = profit of run from prices[i]

    start = 0
    while start < n - 1:
        pos = positions[start]

        # If position is zero, no directional profit, profit = 0
        if pos == 0:
            profits[start] = 0.0
            start += 1
            continue

        # Find the end of this run (where position changes or reaches n-1)
        end = start
        while end + 1 < n - 1 and positions[end + 1] == pos:
            end += 1

        # profit on the run is price difference between prices[end+1] and prices[start]
        price_diff = prices[end + 1] - prices[start]

        # For longs (pos == 1), profit is price_diff
        # For shorts (pos == -1), profit is reversed: prices[start] - prices[end+1]
        # But can write uniformly: pos * price_diff
        run_profit = pos * price_diff

        # Assign profit for each index in run from i=start..end
        # profit[i] = profit from prices[i] to prices[end+1] in direction pos

        # So for i in [start..end]:
        # profit[i] = pos * (prices[end+1] - prices[i])

        for i in range(start, end + 1):
            profits[i] = pos * (prices[end + 1] - prices[i])

        # For the last price index n-1, there is no next interval, so profits[n-1] = 0 by default

        start = end + 1

    # For last index n-1 (no position interval), profit = 0
    profits[n - 1] = 0.0
    return profits


@numba.njit(fastmath=FASTMATH)
def find_best_mps_sequence(prices: np.ndarray, tc: float, tc_mode_is_fraction: bool, optimize_consecutive_regions: bool = True, dtype: object = np.float64):
    """
    prices: 1D numpy array float64 (closing prices)
    tc: transaction cost parameter (if tc_mode_is_fraction True -> fraction of price per trade,
        else fixed currency per trade)
    tc_mode_is_fraction: boolean (True => fraction-of-price, False => fixed)
    Returns:
      positions: int8 array length (n-1) with values -1,0,1 representing position held on interval t->t+1
      total_profit: float64 cumulative profit (after transaction costs)
      cumulative_profits: float64 array length (n-1) of running cum profit after each interval
    """
    n = prices.shape[0]
    if n < 2:
        return np.empty(0, dtype=np.int8), np.empty(0, dtype=dtype)

    m = n - 1  # number of intervals
    deltas = np.empty(m, dtype=dtype)
    for i in range(m):
        deltas[i] = prices[i + 1] - prices[i]

    # 3 states: idx 0 -> pos -1, 1 -> pos 0, 2 -> pos 1
    dp = np.full(3, -1e300, dtype=dtype)  # current best cumul. profit up to previous interval
    dp_next = np.full(3, -1e300, dtype=dtype)
    # backpointers: for each time and state, store prev_state index
    back = np.empty((m, 3), dtype=np.int8)

    # initialize for first interval t = 0 (we consider prev_pos implicitly as 0 at time -1)
    # We'll assume starting prev_pos = 0 (flat) and no cost for initial flat.
    prev_pos = 0
    # compute dp for t=0 (after first interval)
    price_t = prices[0]
    for new_idx in range(3):
        new_pos = _index_state(new_idx)
        trades = _trade_count(prev_pos, new_pos)
        cost = _trade_cost(price_t, trades, tc, tc_mode_is_fraction)
        reward = new_pos * deltas[0] - cost
        dp[new_idx] = reward
        back[0, new_idx] = _state_index(prev_pos)  # all point to initial flat (state idx 1)

    # iterate intervals 1..m-1
    for t in range(1, m):
        price_t = prices[t]  # trades executed at price[t]
        for new_idx in range(3):
            new_pos = _index_state(new_idx)
            best_val = -1e300
            best_prev = 0
            for prev_idx in range(3):
                prev_val = dp[prev_idx]
                prev_pos_local = _index_state(prev_idx)
                trades = _trade_count(prev_pos_local, new_pos)
                cost = _trade_cost(price_t, trades, tc, tc_mode_is_fraction)
                cand = prev_val + new_pos * deltas[t] - cost
                if cand > best_val:
                    best_val = cand
                    best_prev = prev_idx
            dp_next[new_idx] = best_val
            back[t, new_idx] = best_prev
        # swap
        for k in range(3):
            dp[k] = dp_next[k]

    # find best final state
    best_final_idx = 0
    best_final_val = dp[0]
    for k in range(1, 3):
        if dp[k] > best_final_val:
            best_final_val = dp[k]
            best_final_idx = k

    # reconstruct positions per interval
    positions = np.empty(m, dtype=np.int8)
    cur_idx = best_final_idx
    for t in range(m - 1, -1, -1):
        positions[t] = _index_state(cur_idx)
        cur_idx = back[t, cur_idx]

    if optimize_consecutive_regions:
        positions = backfill_zeros_from_right(positions)

    # compute profits from current idx till the end of current area
    profits = compute_area_profits(prices=prices, positions=positions)

    return positions, profits


@numba.njit(fastmath=FASTMATH)
def backfill_zeros_from_right(arr):
    """
    >>>
    a = np.array([0, 0, 1, 0, 0, -1, 0, 0])
    print(backfill_zeros_from_right(a))

    [ 1  1  1 -1 -1 -1  0  0]
    """
    arr = np.asarray(arr)
    out = arr.copy()
    # Go from right to left, filling zeros with the last seen non-zero
    mask = out == 0
    # Find the last non-zero to the right for each position
    last = 0
    for i in range(len(out) - 1, -1, -1):
        if out[i] != 0:
            last = out[i]
        elif last != 0:
            out[i] = last
    return out


# public wrapper to call from normal Python (non-numba callers)
def find_maximum_profit_system(
    prices: np.ndarray, tc: float = 1e-10, tc_mode: str = "fraction", optimize_consecutive_regions: bool = True, dtype: object = np.float64
):
    """
    prices: 1D array-like of closing prices
    tc: transaction cost (fraction-of-price if tc_mode='fraction', else fixed currency)
    tc_mode: 'fraction' or 'fixed'
    returns: dict with keys 'positions', 'profits'

    prices = np.array([100.0, 101.5, 100.0, 99.0, 100.5, 102.0, 101.0])
    r = find_maximum_profit_system(prices, tc=0.005, tc_mode='fraction')
    print("positions:", r['positions'])
    print("profits:", r['profits'])

    >>>
    positions: [ 1 -1 -1  1  1 -1]
    profits: [ 1.      0.485   0.5     0.51    0.9975 -0.02  ]

    """
    arr = np.asarray(prices, dtype=dtype)
    if tc_mode not in ("fraction", "fixed"):
        raise ValueError("tc_mode must be 'fraction' or 'fixed'")
    positions, profits = find_best_mps_sequence(
        arr, tc=float(tc), tc_mode_is_fraction=(tc_mode == "fraction"), optimize_consecutive_regions=optimize_consecutive_regions, dtype=dtype
    )
    return {
        "positions": positions,  # length n-1 array of -1/0/1
        "profits": profits,  # running rel profit (%) form cur_idx till the end of the area
    }


def show_mps_regions(prices: np.ndarray, positions: np.ndarray = None, tc: float = 1e-10, tc_mode: str = "fraction", figsize=(10, 5)):

    if positions is None:
        # Get optimal positions
        res = find_maximum_profit_system(prices, tc=tc, tc_mode=tc_mode)
        positions = res["positions"]  # length n-1

    # Plot price
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(prices, color="k", linewidth=1.5, label="Price")

    # Overlay background colors per interval
    for i, pos in enumerate(positions):
        if pos == 1:
            color = "green"
        elif pos == -1:
            color = "red"
        else:
            color = "black"
        ax.axvspan(i, i + 1, facecolor=color, alpha=0.2)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Price")
    ax.set_title("Optimal Long/Short/Flat Positions")
    ax.legend()
    plt.show()


def generate_market_price(n_days=100, base_price=100.0, trend=0.1, start_date=datetime(2024, 1, 1), base_volume=5000, random_seed: int = 42) -> tuple:
    # Generate sample data with more interesting patterns & slight upward trend
    np.random.seed(random_seed)

    # Create date range
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    dates_array = np.array(dates)

    # Generate more realistic price data with trends and volatility

    prices = np.zeros(n_days)
    prices[0] = base_price

    for i in range(1, n_days):
        # Add trend, volatility, and some mean reversion
        change = np.random.normal(trend, 2.5)
        if i > 1:
            # Add some mean reversion
            change += (base_price - prices[i - 1]) * 0.01

        prices[i] = max(prices[i - 1] + change, 1.0)

        # Add some occasional big moves (news events)
        if np.random.random() < 0.05:
            prices[i] *= np.random.choice([0.95, 1.05])

    # Generate correlated volume data (higher volume on big price moves)

    volumes = np.zeros(n_days)

    for i in range(n_days):
        # Base volume with random variation
        vol = base_volume * np.random.uniform(0.5, 2.0)

        # Increase volume on big price moves
        if i > 0:
            price_change_pct = abs(prices[i] - prices[i - 1]) / prices[i - 1]
            vol *= 1 + price_change_pct * 10

        volumes[i] = vol

    return dates, prices, volumes
