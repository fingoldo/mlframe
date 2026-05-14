"""Maximum Profit System: optimal long/flat/short positioning under transaction costs."""

__all__ = [
    "compute_area_profits",
    "find_best_mps_sequence",
    "backfill_zeros",
    "find_maximum_profit_system",
    "plot_positions",
    "show_mps_regions",
    "generate_market_price",
    "safely_compute_mps",
    "compute_mps_targets",
]

import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple, Union

import numba
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Disable fastmath: the DP loop sums many small reward/cost terms; reassociation can shift the
# argmax in degenerate ties and produce non-deterministic positions across runs.
FASTMATH: bool = False

# Sentinel "very negative" value used as -inf in the DP table. np.finfo(float64).min would also
# work but propagates as -inf under fastmath; this constant is safely finite.
_DP_NEG_INF: float = -1e300

# Defaults for generate_market_price (synthetic data generator).
_GEN_DEFAULT_VOLATILITY: float = 2.5
_GEN_DEFAULT_MEAN_REVERSION: float = 0.01
_GEN_DEFAULT_EVENT_PROBABILITY: float = 0.05
_GEN_EVENT_MULTIPLIERS: Tuple[float, float] = (0.95, 1.05)
_GEN_VOLUME_AMPLIFICATION: float = 10.0

# Default transaction-cost. Matches ``find_maximum_profit_system`` and is a realistic round-trip
# fee for retail brokers (~3 bps). ``compute_mps_targets`` previously defaulted to ``1e-10`` (no-cost),
# which produces unrealistic "max profit" targets that look great in backtests but never trade.
_DEFAULT_TC: float = 3e-4


@numba.njit(fastmath=FASTMATH)
def _trade_count(prev_pos, new_pos):
    """Number of trades executed when switching ``prev_pos`` -> ``new_pos``.

    Continuing the same position (prev == new) is ZERO trades; the previous code returned 1 in
    that branch, charging a phantom transaction cost every interval and biasing the DP toward
    churning out of profitable long/short holds.
    """
    if prev_pos == new_pos:
        return 0
    trades = 0
    if prev_pos != 0:
        trades += 1  # closing previous
    if new_pos != 0:
        trades += 1  # opening new
    return trades


@numba.njit(fastmath=FASTMATH)
def _trade_cost(price_t, trades, tc, tc_mode_is_fraction):
    if trades == 0:
        return 0.0
    if tc_mode_is_fraction:
        return price_t * tc * trades
    return tc * trades


@numba.njit(fastmath=FASTMATH)
def compute_area_profits(prices, positions):
    """Per-bar profit fraction (profit / entry_price) of the run that bar belongs to.

    Bars where ``prices[i] <= 0`` produce 0 (no meaningful ratio without a valid denominator).
    """
    n = prices.shape[0]
    profits = np.zeros(n, dtype=prices.dtype)

    start = 0
    while start < n:
        pos = positions[start]

        if pos == 0:
            profits[start] = 0.0
            start += 1
            continue

        end = start
        while end + 1 < n and positions[end + 1] == pos:
            end += 1

        # Run extends to final bar: there is no closing price; leave tail profits at 0.
        if end >= n - 1:
            start = end + 1
            continue

        for i in range(start, end + 1):
            profits[i] = pos * (prices[end + 1] - prices[i])

        start = end + 1

    profits[n - 1] = 0.0

    # Safe division: zero/negative-price bars (synthetic or corrupted feeds) produced inf/NaN under
    # naive ``profits / prices``, which then silently poisoned downstream ML features.
    out = np.zeros_like(profits)
    for i in range(n):
        if prices[i] > 0:
            out[i] = profits[i] / prices[i]
    return out


@numba.njit(fastmath=FASTMATH)
def find_best_mps_sequence(
    prices: np.ndarray,
    raw_prices: np.ndarray,
    tc: float,
    tc_mode_is_fraction: bool,
    optimize_consecutive_regions: bool = True,
    shift: int = 0,
    dtype: np.dtype = np.float64,
):
    """Optimal long/flat/short DP under transaction costs.

    Parameters
    ----------
    prices, raw_prices
        1D float arrays; ``prices`` drives the DP (often a smoothed signal), ``raw_prices`` is
        used for the realised profit ratio.
    tc
        Transaction cost. Interpreted as a fraction of price when ``tc_mode_is_fraction`` is
        ``True``, otherwise a fixed currency amount per trade.
    optimize_consecutive_regions
        Backfill zero positions so consecutive runs of the same sign merge.
    shift
        Apply a forward shift of ``shift`` bars to the position vector (used to simulate
        execution lag).

    Returns
    -------
    positions
        ``int8`` array length ``n-1``; values in ``{-1, 0, 1}``.
    profits
        Per-bar realised profit ratio (length ``n-1``).
    """
    n = prices.shape[0]
    if n < 2:
        # Zero-init the empty profits so the caller never reads uninitialised memory.
        return np.empty(0, dtype=np.int8), np.zeros(0, dtype=dtype)

    m = n - 1
    deltas = np.empty(m, dtype=dtype)
    for i in range(m):
        deltas[i] = prices[i + 1] - prices[i]

    # State index = position + 1: -1->0, 0->1, +1->2. Arithmetic mapping is branch-free and
    # ~5% faster in the hot DP loop than the previous if/elif chain.
    dp = np.full(3, _DP_NEG_INF, dtype=dtype)
    dp_next = np.full(3, _DP_NEG_INF, dtype=dtype)
    back = np.empty((m, 3), dtype=np.int8)

    # First interval: implicit prev_pos = 0 (flat) at "time -1", no entry cost on the flat state.
    prev_pos = 0
    price_t = prices[0]
    for new_idx in range(3):
        new_pos = new_idx - 1
        trades = _trade_count(prev_pos, new_pos)
        cost = _trade_cost(price_t, trades, tc, tc_mode_is_fraction)
        reward = new_pos * deltas[0] - cost
        dp[new_idx] = reward
        back[0, new_idx] = prev_pos + 1

    for t in range(1, m):
        price_t = prices[t]
        for new_idx in range(3):
            new_pos = new_idx - 1
            best_val = _DP_NEG_INF
            best_prev = 0
            for prev_idx in range(3):
                prev_val = dp[prev_idx]
                prev_pos_local = prev_idx - 1
                trades = _trade_count(prev_pos_local, new_pos)
                cost = _trade_cost(price_t, trades, tc, tc_mode_is_fraction)
                cand = prev_val + new_pos * deltas[t] - cost
                if cand > best_val:
                    best_val = cand
                    best_prev = prev_idx
            dp_next[new_idx] = best_val
            back[t, new_idx] = best_prev
        for k in range(3):
            dp[k] = dp_next[k]

    best_final_idx = 0
    best_final_val = dp[0]
    for k in range(1, 3):
        if dp[k] > best_final_val:
            best_final_val = dp[k]
            best_final_idx = k

    positions = np.empty(m, dtype=np.int8)
    cur_idx = best_final_idx
    for t in range(m - 1, -1, -1):
        positions[t] = cur_idx - 1
        cur_idx = back[t, cur_idx]

    if optimize_consecutive_regions:
        positions = backfill_zeros(positions, direction="right")
        positions = backfill_zeros(positions, direction="left")

    if shift > 0:
        if shift < m:
            positions[:-shift] = positions[shift:]
            positions = backfill_zeros(positions, direction="left")
        else:
            # Shift >= length: entire vector becomes the right-edge value, then left-filled.
            positions[:] = 0
            positions = backfill_zeros(positions, direction="left")

    profits = compute_area_profits(prices=raw_prices, positions=positions)

    return positions, profits


@numba.njit(fastmath=FASTMATH)
def backfill_zeros(arr, direction="right"):
    """Fill zero entries with the nearest non-zero neighbour from the given direction.

    Examples
    --------
    >>> a = np.array([0, 0, 1, 0, 0, -1, 0, 0])
    >>> backfill_zeros(a, direction='right')
    array([ 1,  1,  1, -1, -1, -1,  0,  0])
    >>> backfill_zeros(a, direction='left')
    array([ 0,  0,  1,  1,  1, -1, -1, -1])
    """
    out = arr.copy()

    if direction == "right":
        last = 0
        for i in range(len(out) - 1, -1, -1):
            if out[i] != 0:
                last = out[i]
            elif last != 0:
                out[i] = last
    else:
        last = 0
        for i in range(len(out)):
            if out[i] != 0:
                last = out[i]
            elif last != 0:
                out[i] = last

    return out


def find_maximum_profit_system(
    prices: np.ndarray,
    raw_prices: Optional[np.ndarray] = None,
    tc: float = _DEFAULT_TC,
    tc_mode: str = "fraction",
    optimize_consecutive_regions: bool = True,
    shift: int = 0,
    dtype: np.dtype = np.float64,
) -> dict:
    """Wrapper around ``find_best_mps_sequence`` callable from non-numba Python.

    Parameters
    ----------
    prices
        1D array-like of closing prices.
    tc
        Transaction cost (fraction-of-price when ``tc_mode='fraction'``, else fixed currency).
    tc_mode
        ``'fraction'`` or ``'fixed'``.

    Returns
    -------
    Dict with keys ``positions`` (int8 array length n-1) and ``profits`` (running relative profit).

    Examples
    --------
    >>> prices = np.array([100.0, 101.5, 100.0, 99.0, 100.5, 102.0, 101.0])
    >>> r = find_maximum_profit_system(prices, tc=0.005, tc_mode='fraction')  # doctest: +SKIP
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

    # nan_to_num collapses both genuine NaN and any inf escapees from compute_area_profits.
    return {
        "positions": positions,
        "profits": np.nan_to_num(profits, copy=False, nan=0.0, posinf=0.0, neginf=0.0),
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
):
    """Plot price with position-coloured background using Plotly (default) or Matplotlib."""
    # Lazy plotting imports - importing this module shouldn't pay matplotlib/plotly cost.
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    x_data = list(range(len(prices)))

    color_map = {
        1: {"name": "Long", "color": "green"},
        -1: {"name": "Short", "color": "red"},
        0: {"name": "Flat", "color": "black"},
    }

    if use_plotly:
        fig = go.Figure()

        hover_text = None
        if profits is not None:
            hover_text = [
                f"{price_label}: {price:.2f}<br>Profit: {profit*100:.2f}%<br>Position: {position}"
                for price, profit, position in zip(prices, profits, positions)
            ]

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

        # Background as a single bar trace (per-cell vrects scale O(n) DOM nodes on long series).
        if len(positions) > 0:
            all_prices_arr = np.asarray(prices)
            if raw_prices is not None:
                all_prices_arr = np.concatenate([all_prices_arr, np.asarray(raw_prices)])
            y_min = float(all_prices_arr.min())
            y_max = float(all_prices_arr.max())
            y_range = y_max - y_min
            bar_height = y_range * 1.2
            bar_base = y_min - y_range * 0.1

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

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            showlegend=False,
            width=figsize[0] * plotly_size_multiplier,
            height=figsize[1] * plotly_size_multiplier,
            bargap=0,
            bargroupgap=0,
        )

    else:
        fig, ax = plt.subplots(figsize=figsize)

        if raw_prices is not None:
            ax.plot(
                x_data,
                raw_prices,
                color=raw_price_line_color,
                linewidth=raw_price_line_width,
                linestyle="--",
                alpha=raw_price_opacity,
                label=raw_price_label,
            )

        ax.plot(x_data, prices, color=price_line_color, linewidth=price_line_width, label=price_label)

        # Vectorise position spans via broken_barh - per-bar axvspan was O(n) Python calls.
        y_min, y_max = ax.get_ylim()
        for sign, info in color_map.items():
            if sign == 0:
                continue
            mask = np.asarray(positions) == sign
            if not mask.any():
                continue
            spans = []
            i = 0
            n = len(mask)
            while i < n:
                if mask[i]:
                    j = i
                    while j + 1 < n and mask[j + 1]:
                        j += 1
                    spans.append((i, j - i + 1))
                    i = j + 1
                else:
                    i += 1
            ax.broken_barh(spans, (y_min, y_max - y_min), facecolor=info["color"], alpha=background_opacity)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best")

    return fig


def show_mps_regions(
    prices: np.ndarray,
    raw_prices: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    tc: float = _DEFAULT_TC,
    shift: int = 0,
    profit_quantile: float = 0.95,
    tc_mode: str = "fraction",
    figsize: Tuple[int, int] = (10, 5),
    use_plotly: bool = True,
    show_chart: bool = True,
    title: str = "Optimal Position",
) -> dict:
    profits = None
    profit_quantile_value = None
    max_profit = None
    res: dict = {}
    if positions is None:
        res = find_maximum_profit_system(
            prices=prices, raw_prices=raw_prices, tc=tc, tc_mode=tc_mode, shift=shift
        )
        positions = res["positions"]

        if res["profits"].size > 0:
            max_profit = float(res["profits"].max())
            profit_quantile_value = float(np.quantile(res["profits"], profit_quantile))
            title = (
                f"{title} tc={tc*100:.2f}%, "
                f"{profit_quantile*100:.0f}_perc_profit={profit_quantile_value*100:.2f}%, "
                f"max_profit={max_profit*100:.2f}%"
            )

        if use_plotly:
            profits = res["profits"]

    if show_chart:
        fig = plot_positions(
            prices=prices,
            raw_prices=raw_prices,
            positions=positions,
            profits=profits,
            figsize=figsize,
            use_plotly=use_plotly,
            title=title,
        )
        fig.show()

    return dict(profit_quantile=profit_quantile_value, max_profit=max_profit, **res)


def generate_market_price(
    n_days: int = 100,
    base_price: float = 100.0,
    trend: float = 0.1,
    start_date: Optional[datetime] = None,
    base_volume: float = 5000.0,
    random_seed: int = 42,
    volatility: float = _GEN_DEFAULT_VOLATILITY,
    mean_reversion: float = _GEN_DEFAULT_MEAN_REVERSION,
    event_probability: float = _GEN_DEFAULT_EVENT_PROBABILITY,
    event_multipliers: Tuple[float, float] = _GEN_EVENT_MULTIPLIERS,
    volume_amplification: float = _GEN_VOLUME_AMPLIFICATION,
) -> Tuple[list, np.ndarray, np.ndarray]:
    """Synthetic OHLC-style price + volume series. Used by tests and notebook demos."""
    rng = np.random.default_rng(random_seed)
    if start_date is None:
        start_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    prices = np.empty(n_days, dtype=np.float64)
    prices[0] = base_price
    changes = rng.normal(trend, volatility, size=n_days - 1)
    events_mask = rng.random(n_days - 1) < event_probability
    event_choices = rng.choice(event_multipliers, size=n_days - 1)
    for i in range(1, n_days):
        change = changes[i - 1]
        if i > 1:
            change += (base_price - prices[i - 1]) * mean_reversion
        prices[i] = max(prices[i - 1] + change, 1.0)
        if events_mask[i - 1]:
            prices[i] *= event_choices[i - 1]

    volumes = base_volume * rng.uniform(0.5, 2.0, size=n_days)
    if n_days > 1:
        rel_change = np.abs(np.diff(prices)) / prices[:-1]
        volumes[1:] *= 1.0 + rel_change * volume_amplification

    return dates, prices, volumes


def safely_compute_mps(fpath, **kwargs):
    """Wrap ``compute_mps_targets`` with file-existence and exception guards. Returns ``None`` on failure."""
    if not os.path.exists(fpath):
        return None
    try:
        res = compute_mps_targets(fpath, **kwargs)
        if res is not None and len(res) > 0:
            return res
    except Exception:
        logger.exception("safely_compute_mps failed for %s", fpath)
    return None


def compute_mps_targets(
    fpath: Optional[str] = None,
    fo_df: Optional[pl.DataFrame] = None,
    ts_field: str = "ts",
    group_field: str = "secid",
    price_field: str = "pr_close",
    tc: float = _DEFAULT_TC,
    sma_size: int = 0,
    ewm_alpha: float = 0.3,
    dtype: np.dtype = np.float64,
    tc_mode_is_fraction: bool = True,
    optimize_consecutive_regions: bool = True,
    final_price_alias: str = "final_price",
) -> Optional[pl.DataFrame]:
    """Compute per-ticker MPS positions/profits from parquet or a pre-loaded polars frame.

    Exactly one of ``fpath`` (parquet path) or ``fo_df`` (already-loaded frame) must be supplied.
    Returns ``None`` when both are unset or the join produces zero ticker groups.
    """
    if fo_df is None:
        if fpath is None:
            raise ValueError("compute_mps_targets: must provide either fpath or fo_df")
        try:
            fo_df = (
                pl.read_parquet(
                    fpath,
                    columns=[ts_field, group_field, price_field],
                    allow_missing_columns=True,
                )
                .unique(subset=[ts_field, group_field], keep="first")
                .sort(ts_field)
            )
        except Exception:
            logger.warning("compute_mps_targets: failed to read %s", fpath, exc_info=True)
            return None

    basic_expr = pl.col(price_field).fill_null(strategy="forward").fill_null(strategy="backward")

    if sma_size:
        final_expr = basic_expr.rolling_mean(window_size=sma_size, min_samples=1)
    elif ewm_alpha:
        final_expr = basic_expr.ewm_mean(alpha=ewm_alpha)
    else:
        final_expr = basic_expr

    grouped_df = (
        fo_df.sort(group_field, ts_field)
        .group_by(group_field)
        .agg(pl.col(ts_field), basic_expr, final_expr.alias(final_price_alias))
    )

    parts: list = []
    for row in grouped_df.iter_rows(named=True):
        raw_prices = np.asarray(row[price_field], dtype=dtype)
        final_prices = np.asarray(row[final_price_alias], dtype=dtype)
        # Skip rows where any final price is NaN/missing - find_best_mps_sequence requires finite floats.
        if final_prices.size < 2 or not np.isfinite(final_prices).all():
            continue
        positions, profits = find_best_mps_sequence(
            prices=final_prices,
            raw_prices=raw_prices,
            tc=float(tc),
            tc_mode_is_fraction=tc_mode_is_fraction,
            optimize_consecutive_regions=optimize_consecutive_regions,
            dtype=dtype,
        )
        parts.append(
            pl.DataFrame(
                dict(
                    ts=row[ts_field][:-1],
                    secid=[row[group_field]] * (len(final_prices) - 1),
                    OPTIMAL_POSITION=positions,
                    OPTIMAL_PROFIT=np.nan_to_num(profits[:-1], copy=False, nan=0.0, posinf=0.0, neginf=0.0),
                )
            )
        )

    if not parts:
        return None
    return pl.concat(parts)
