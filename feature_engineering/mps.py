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
import polars as pl
from os.path import exists

import matplotlib.pyplot as plt
import plotly.graph_objects as go

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
    return np.nan_to_num(profits / prices, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


@numba.njit(fastmath=FASTMATH)
def find_best_mps_sequence(
    prices: np.ndarray,
    raw_prices: np.ndarray,
    tc: float,
    tc_mode_is_fraction: bool,
    optimize_consecutive_regions: bool = True,
    shift: int = 0,
    dtype: object = np.float64,
):
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
        positions = backfill_zeros(positions, direction="right")
        positions = backfill_zeros(positions, direction="left")

    if shift > 0:
        positions[:-shift] = positions[shift:]
        positions = backfill_zeros(positions, direction="left")
    # compute profits from current idx till the end of current area
    profits = compute_area_profits(prices=raw_prices, positions=positions)

    return positions, profits


@numba.njit(fastmath=True)
def backfill_zeros(arr, direction="right"):
    """
    Backfill zeros in an array from either right or left based on direction parameter.

    Parameters:
    arr : numpy.ndarray
        Input array containing zeros to be backfilled
    direction : str
        Direction of backfill, either 'right' or 'left'

    Returns:
    numpy.ndarray
        Array with zeros backfilled from specified direction

    Examples:
    >>> a = np.array([0, 0, 1, 0, 0, -1, 0, 0])
    >>> print(backfill_zeros(a, direction='right'))
    [ 1  1  1 -1 -1 -1  0  0]
    >>> print(backfill_zeros(a, direction='left'))
    [ 0  0  1  1  1 -1 -1 -1]
    """
    arr = np.asarray(arr)
    out = arr.copy()

    if direction == "right":
        # Go from right to left, filling zeros with the last seen non-zero
        last = 0
        for i in range(len(out) - 1, -1, -1):
            if out[i] != 0:
                last = out[i]
            elif last != 0:
                out[i] = last
    else:  # direction == 'left'
        # Go from left to right, filling zeros with the last seen non-zero
        last = 0
        for i in range(len(out)):
            if out[i] != 0:
                last = out[i]
            elif last != 0:
                out[i] = last

    return out


# public wrapper to call from normal Python (non-numba callers)
def find_maximum_profit_system(
    prices: np.ndarray,
    raw_prices: np.ndarray = None,
    tc: float = 3e-4,
    tc_mode: str = "fraction",
    optimize_consecutive_regions: bool = True,
    shift: int = 0,
    dtype: object = np.float64,
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
    prices_arr = np.asarray(prices, dtype=dtype)

    if raw_prices is not None:
        raw_prices_arr = np.asarray(raw_prices, dtype=dtype)
    else:
        raw_prices_arr = prices_arr

    if tc_mode not in ("fraction", "fixed"):
        raise ValueError("tc_mode must be 'fraction' or 'fixed'")

    positions, profits = find_best_mps_sequence(
        prices=prices_arr,
        raw_prices=raw_prices_arr,
        tc=float(tc),
        tc_mode_is_fraction=(tc_mode == "fraction"),
        shift=shift,
        optimize_consecutive_regions=optimize_consecutive_regions,
        dtype=dtype,
    )

    return {
        "positions": positions,  # length n-1 array of -1/0/1
        "profits": profits,  # running rel profit (%) form cur_idx till the end of the area
    }


def plot_positions(
    prices: Union[np.ndarray, list],
    positions: Union[np.ndarray, list],
    raw_prices: Optional[np.ndarray] = None,
    profits: Optional[np.ndarray] = None,
    use_plotly: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Optimal Positions",
    xlabel: str = "Time step",
    ylabel: str = "Price",
    price_label: str = "Price",
    raw_price_label: str = "Raw Price",
    price_line_width: float = 1.5,
    raw_price_line_width: float = 1.0,
    price_line_color: str = "black",
    raw_price_line_color: str = "gray",
    raw_price_opacity: float = 0.7,
    background_opacity: float = 0.2,
    plotly_size_multiplier: int = 80,
) -> Union[plt.Figure, go.Figure]:
    """
    Plot price with position background colors using either Plotly or Matplotlib.

    Parameters:
    -----------
    prices : np.ndarray or list
        Price data to plot
    positions : np.ndarray or list
        Position data (1=long/green, -1=short/red, 0=flat/black)
    raw_prices : np.ndarray, optional
        Raw price data to plot with different style/color. If None, not plotted.
    profits : np.ndarray, optional
        Profit data for tooltips (only used in Plotly mode). If None, no profit tooltips shown.
    use_plotly : bool, default=True
        If True, use Plotly; if False, use Matplotlib
    figsize : tuple of int, default=(10, 6)
        Figure size (width, height)
    title : str
        Plot title
    xlabel : str, default="Time step"
        X-axis label
    ylabel : str, default="Price"
        Y-axis label
    price_label : str, default="Price"
        Label for main price line
    raw_price_label : str, default="Raw Price"
        Label for raw price line
    price_line_width : float, default=1.5
        Width of main price line
    raw_price_line_width : float, default=1.0
        Width of raw price line
    price_line_color : str, default="black"
        Color of main price line
    raw_price_line_color : str, default="gray"
        Color of raw price line
    raw_price_opacity : float, default=0.7
        Opacity of raw price line
    background_opacity : float, default=0.2
        Opacity of position background colors
    plotly_size_multiplier : int, default=80
        Multiplier to convert figsize to pixels for Plotly

    Returns:
    --------
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        The created figure object
    """

    # Common data preparation
    x_data = list(range(len(prices)))

    # Common color mapping
    color_map = {
        1: {"name": "Long", "color": "green"},
        -1: {"name": "Short", "color": "red"},
        0: {"name": "Flat", "color": "black"},
    }

    if use_plotly:

        # Create Plotly figure
        fig = go.Figure()

        # Prepare hover text for profits if provided
        hover_text = None
        if profits is not None:
            hover_text = [
                f"{price_label}: {price:.2f}<br>Profit: {profit*100:.2f}%<br>Position: {position}"
                for price, profit, position in zip(prices, profits, positions)
            ]

        # Add raw prices if provided
        if raw_prices is not None:
            raw_hover_text = None
            if profits is not None:
                raw_hover_text = [
                    f"{raw_price_label}: {price:.2f}<br>Profit: {profit*100:.2f}%<br>Position: {position}"
                    for price, profit, position in zip(raw_prices, profits, positions)
                ]

            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=raw_prices,
                    mode="lines",
                    line=dict(color=raw_price_line_color, width=raw_price_line_width, dash="dot"),
                    name=raw_price_label,
                    opacity=raw_price_opacity,
                    hovertext=raw_hover_text,
                    hoverinfo="text" if raw_hover_text else "x+y",
                )
            )

        # OPTIMIZED: Create background using bar trace instead of individual vrects
        if len(positions) > 0:
            # Get y-axis range for background bars
            all_prices = list(prices) + (list(raw_prices) if raw_prices is not None else [])
            y_min, y_max = min(all_prices), max(all_prices)
            y_range = y_max - y_min
            bar_height = y_range * 1.2  # Extend slightly beyond data range
            bar_base = y_min - y_range * 0.1

            # Create color array for bars
            bar_colors = [color_map.get(pos, color_map[0])["color"] for pos in positions]

            fig.add_trace(
                go.Bar(
                    x=x_data,
                    y=[bar_height] * len(positions),
                    base=bar_base,
                    marker=dict(color=bar_colors, opacity=background_opacity, line=dict(width=0)),
                    name="Background",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Add price line
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=prices,
                mode="lines",
                line=dict(color=price_line_color, width=price_line_width),
                name=price_label,
                hovertext=hover_text,
                hoverinfo="text" if hover_text else "x+y",
            )
        )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            showlegend=False,
            width=figsize[0] * plotly_size_multiplier,
            height=figsize[1] * plotly_size_multiplier,
            bargap=0,  # Remove gaps between bars
            bargroupgap=0,
        )

    else:

        # Create Matplotlib figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot raw prices if provided
        if raw_prices is not None:
            ax.plot(
                x_data, raw_prices, color=raw_price_line_color, linewidth=raw_price_line_width, linestyle="--", alpha=raw_price_opacity, label=raw_price_label
            )

        # Plot price line
        ax.plot(x_data, prices, color=price_line_color, linewidth=price_line_width, label=price_label)

        # Add background colors
        for i, pos in enumerate(positions):
            color_info = color_map.get(pos, color_map[0])
            ax.axvspan(i, i + 1, facecolor=color_info["color"], alpha=background_opacity)

        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    return fig


def show_mps_regions(
    prices: np.ndarray,
    raw_prices: np.ndarray = None,
    positions: np.ndarray = None,
    tc: float = 3e-4,
    shift: int = 0,
    profit_quantile: float = 0.95,
    tc_mode: str = "fraction",
    figsize=(10, 5),
    use_plotly: bool = True,
    show_chart: bool = True,
    title: str = "Optimal Position",
) -> dict:

    profits = None
    if positions is None:
        # Get optimal positions
        res = find_maximum_profit_system(prices=prices, raw_prices=raw_prices, tc=tc, tc_mode=tc_mode, shift=shift)
        positions = res["positions"]  # length n-1

        max_profit = res["profits"].max()
        profit_quantile_value = np.quantile(res["profits"], profit_quantile)

        title = title + f" tc={tc*100:.2f}%, {profit_quantile*100:.0f}_perc_profit={profit_quantile_value*100:.2f}%, max_profit={max_profit*100:.2f}%"

        if use_plotly:
            profits = res["profits"]

    if show_chart:

        fig = plot_positions(prices=prices, raw_prices=raw_prices, positions=positions, profits=profits, figsize=figsize, use_plotly=use_plotly, title=title)
        fig.show()

    return dict(profit_quantile=profit_quantile_value, max_profit=max_profit, **res)


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


def safely_compute_mps(f, **kwargs):

    if not exists(f):
        return None
    try:
        res = compute_mps_targets(f, **kwargs)
        if res is not None and len(res) > 0:
            return res
    except Exception as e:
        print(f"Error with {f}: {e}")
    return None


def compute_mps_targets(
    fpath: str = None,
    fo_df: pl.DataFrame = None,
    ts_field: str = "ts",
    group_field: str = "secid",
    price_field: str = "pr_close",
    tc: float = 1e-10,
    sma_size: int = 0,
    ewm_alpha: float = 0.3,
    dtype: object = np.float64,
    tc_mode_is_fraction: bool = True,
    optimize_consecutive_regions: bool = True,
    final_price_alias: str = "final_price",
) -> pl.DataFrame:

    if fo_df is None:
        try:
            fo_df = (
                pl.read_parquet(fpath, columns=[ts_field, group_field, price_field], allow_missing_columns=True)
                .unique(subset=[ts_field, group_field], keep="first")
                .sort(ts_field)
            )
        except Exception as e:
            logger.warning(f"File {fpath}, error {e}")
            return

    basic_expr = pl.col(price_field).fill_null(strategy="forward").fill_null(strategy="backward")

    if sma_size:
        final_expr = basic_expr.rolling_mean(window_size=sma_size, min_samples=1)
    elif ewm_alpha:
        final_expr = basic_expr.ewm_mean(alpha=ewm_alpha)
    else:
        final_expr = basic_expr

    grouped_df = fo_df.sort(group_field, ts_field).group_by(group_field).agg(pl.col(ts_field), basic_expr, final_expr.alias(final_price_alias))

    targets_df = []
    for row in grouped_df.iter_rows(named=True):
        raw_prices = np.array(row[price_field])
        final_prices = np.array(row[final_price_alias])
        if final_prices[0] is not None:
            positions, profits = find_best_mps_sequence(
                prices=final_prices,
                raw_prices=raw_prices,
                tc=float(tc),
                tc_mode_is_fraction=tc_mode_is_fraction,
                optimize_consecutive_regions=optimize_consecutive_regions,
                dtype=dtype,
            )
            targets_df.append(
                pl.DataFrame(
                    dict(
                        ts=row[ts_field][:-1],
                        secid=[row[group_field]] * (len(final_prices) - 1),
                        OPTIMAL_POSITION=positions,
                        OPTIMAL_PROFIT=profits[:-1],
                    )
                )
            )

    # targets_df=pl.concat(targets_df)
    targets_df = pl.concat([el for el in targets_df if el is not None and len(el) > 0])
    return targets_df
