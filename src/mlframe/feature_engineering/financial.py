"""Features for financial modelling."""

from __future__ import annotations


__all__ = [
    "add_ohlcv_ratios_rlags",
    "add_fast_rolling_stats",
    "apply_ta_indicator",
    "add_ohlcv_ta_indicators",
    "create_ohlcv_wholemarket_features",
    "merge_perticker_and_wholemarket_features",
]

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import polars as pl
import polars.selectors as cs

# polars_talib wraps libta-lib (C extension) and only matters for the
# add_ohlcv_ta_indicators() path. Deferred to that function so consumers
# that only want the rolling-stats / market-wide helpers can import
# mlframe.feature_engineering.financial without TA-Lib installed.
try:
    import polars_talib as plta  # type: ignore
except ImportError:
    plta = None  # noqa: F841 - guarded use inside the TA-indicator functions

import pyutilz.polarslib as pllib
from pyutilz.system import clean_ram

logger = logging.getLogger(__name__)

# Default values placed at module scope so they're not rebuilt per call. None-defaults inside
# function bodies use these; we never mutate them.
_DEFAULT_OHLCV_FIELDS: Dict[str, str] = dict(
    qty="qty", open="open", high="high", low="low", close="close", volume="volume"
)
_DEFAULT_TA_OHLCV_FIELDS: Dict[str, str] = dict(open="open", high="high", low="low", close="close", volume="volume")
_DEFAULT_MARKET_ACTION_PREFIXES: Tuple[str, ...] = ("",)
_DEFAULT_LAGS: Tuple[int, ...] = (1,)
_DEFAULT_TA_WINDOWS: Tuple[int, ...] = (5, 10)
_DEFAULT_FSS_WINDOWS: Tuple[Tuple[int, int, int], ...] = ((12, 26, 9),)
_DEFAULT_ROLLING_WINDOWS: Tuple[int, ...] = (5,)
_DEFAULT_ROLLING_NUMAGGS: Tuple[str, ...] = (
    "rolling_min",
    "rolling_max",
    "rolling_mean",
    "rolling_std",
    "rolling_skew",
    "rolling_kurtosis",
)
_DEFAULT_WHOLEMARKET_NUMAGGS: Tuple[str, ...] = (
    "min", "max", "mean", "median", "std", "skew", "kurtosis", "entropy", "n_unique",
)
_DEFAULT_WEIGHTING_COLUMNS: Tuple[str, ...] = ("volume", "qty")


def _group_if_needed(expr: pl.Expr, over: str = "") -> pl.Expr:
    """Apply ``.over(over)`` only when ``over`` is truthy; otherwise pass through unchanged."""
    return expr.over(over) if over else expr


def add_ohlcv_ratios_rlags(
    ohlcv: pl.DataFrame,
    columns_selector: str = "",
    lags: Optional[Sequence[int]] = None,
    crossbar_ratios_lags: Optional[Sequence[int]] = None,
    add_ratios: bool = True,
    add_rlags: bool = True,
    ticker_column: str = "ticker",
    exclude_fields: Optional[Sequence[str]] = None,
    market_action_prefixes: Optional[Sequence[str]] = None,
    ohlcv_fields_mapping: Optional[Dict[str, str]] = None,
    nans_filler: float = 0.0,
    cast_f64_to_f32: bool = True,
) -> pl.DataFrame:
    """Add nuanced features to raw OHLCV (interbar ratios + relative lags).

    Assumes the dataframe is sorted by timestamp. Grouping uses polars ``.over(ticker_column)``.
    Pass an empty list (``[]``) to ``market_action_prefixes`` to legitimately request "no
    prefixes"; ``None`` selects the default single-empty-prefix ``[""]``.
    """
    if lags is None:
        lags = list(_DEFAULT_LAGS)
    if crossbar_ratios_lags is None:
        crossbar_ratios_lags = list(_DEFAULT_LAGS)

    # Negative/zero lag pulls future prices into current row -> look-ahead leakage. Guard
    # explicitly rather than let the polars expression succeed silently.
    for _lag in lags:
        if _lag <= 0:
            raise ValueError(
                f"add_ohlcv_ratios_rlags: lag must be > 0, got {_lag} (negative/zero causes look-ahead leakage)"
            )
    for _lag in crossbar_ratios_lags:
        if _lag <= 0:
            raise ValueError(
                f"add_ohlcv_ratios_rlags: crossbar_ratios_lag must be > 0, got {_lag} (negative/zero causes look-ahead leakage)"
            )
    if market_action_prefixes is None:
        market_action_prefixes = list(_DEFAULT_MARKET_ACTION_PREFIXES)
    if ohlcv_fields_mapping is None:
        ohlcv_fields_mapping = dict(_DEFAULT_OHLCV_FIELDS)

    all_num_cols = cs.numeric()
    if columns_selector:
        all_num_cols = all_num_cols & cs.contains(columns_selector)
    if exclude_fields:
        all_num_cols = all_num_cols - cs.by_name(exclude_fields)

    interbar_ratios_features: list = []
    if add_ratios:
        for prefix in market_action_prefixes:

            qty = pl.col(f"{prefix}{ohlcv_fields_mapping.get('qty')}")
            low = pl.col(f"{prefix}{ohlcv_fields_mapping.get('low')}")
            high = pl.col(f"{prefix}{ohlcv_fields_mapping.get('high')}")
            open_col = pl.col(f"{prefix}{ohlcv_fields_mapping.get('open')}")
            close = pl.col(f"{prefix}{ohlcv_fields_mapping.get('close')}")
            volume = pl.col(f"{prefix}{ohlcv_fields_mapping.get('volume')}")

            interbar_ratios_features.extend(
                [
                    (close / open_col - 1).alias(f"{prefix}close_to_open"),
                    (high / open_col - 1).alias(f"{prefix}high_to_open"),
                    (open_col / low - 1).alias(f"{prefix}open_to_low"),
                    (close / low - 1).alias(f"{prefix}close_to_low"),
                    (high / low - 1).alias(f"{prefix}high_to_low"),
                    (high / close - 1).alias(f"{prefix}high_to_close"),
                    pllib.clean_numeric((volume / qty), nans_filler=nans_filler).alias(f"{prefix}avg_trade_size"),
                ]
            )
            if ticker_column:
                for period_shift in crossbar_ratios_lags:
                    interbar_ratios_features.extend(
                        [
                            (close / open_col.shift(period_shift).over(ticker_column) - 1).alias(f"{prefix}close_to_open-{period_shift}"),
                            (high / open_col.shift(period_shift).over(ticker_column) - 1).alias(f"{prefix}high_to_open-{period_shift}"),
                            (open_col / low.shift(period_shift).over(ticker_column) - 1).alias(f"{prefix}open_to_low-{period_shift}"),
                            (close / low.shift(period_shift).over(ticker_column) - 1).alias(f"{prefix}close_to_low-{period_shift}"),
                            (high / low.shift(period_shift).over(ticker_column) - 1).alias(f"{prefix}high_to_low-{period_shift}"),
                            (high / close.shift(period_shift).over(ticker_column) - 1).alias(f"{prefix}high_to_close-{period_shift}"),
                            pllib.clean_numeric(
                                (volume / qty.shift(period_shift).over(ticker_column)),
                                nans_filler=nans_filler,
                            ).alias(f"{prefix}avg_trade_size-{period_shift}"),
                        ]
                    )

    if add_ratios:
        ohlcv = ohlcv.with_columns(*interbar_ratios_features)

    if add_rlags:
        ohlcv = ohlcv.with_columns(
            *[
                pllib.clean_numeric(
                    (all_num_cols / _group_if_needed(all_num_cols.shift(lag), over=ticker_column) - 1),
                    nans_filler=nans_filler,
                ).name.suffix(f"_rlag{lag}")
                for lag in lags
            ],
        )

    if cast_f64_to_f32:
        ohlcv = pllib.cast_f64_to_f32(ohlcv)

    return ohlcv


def add_fast_rolling_stats(
    df: pl.DataFrame,
    columns_selector: Optional[str] = None,
    rolling_windows: Optional[Sequence[int]] = None,
    numaggs: Optional[Sequence[str]] = None,
    relative: bool = True,
    min_samples: int = 1,
    groupby_column: Optional[str] = None,
    exclude_fields: Optional[Sequence[str]] = None,
    nans_filler: float = 0.0,
    cast_f64_to_f32: bool = True,
) -> pl.DataFrame:
    """Add rolling-window statistics to a frame. Assumes the frame is sorted by timestamp."""
    if not rolling_windows:
        rolling_windows = list(_DEFAULT_ROLLING_WINDOWS)
    if numaggs is None:
        numaggs = list(_DEFAULT_ROLLING_NUMAGGS)

    all_num_cols = cs.numeric()
    if columns_selector:
        all_num_cols = all_num_cols & cs.contains(columns_selector)
    if exclude_fields:
        all_num_cols = all_num_cols - cs.by_name(exclude_fields)

    exprs: list = []
    for func in numaggs:
        short_name = func.replace("rolling_", "")
        if relative:
            exprs.extend(
                [
                    pllib.clean_numeric(
                        (
                            all_num_cols
                            / _group_if_needed(getattr(all_num_cols, func)(window, min_samples=min_samples), over=groupby_column)
                            - 1
                        ),
                        nans_filler=nans_filler,
                    ).name.suffix(f"_r{short_name}{window}")
                    for window in rolling_windows
                ]
            )
        else:
            exprs.extend(
                [
                    _group_if_needed(
                        getattr(all_num_cols, func)(window, min_samples=min_samples), over=groupby_column
                    ).name.suffix(f"_{short_name}{window}")
                    for window in rolling_windows
                ]
            )

    df = df.with_columns(exprs)

    if cast_f64_to_f32:
        df = pllib.cast_f64_to_f32(df)

    return df


def apply_ta_indicator(
    expr: pl.Expr,
    func: str,
    window: Union[int, str],
    ticker_column: str,
    unnests: Sequence[str],
    prefix: str,
    fastperiod: int = 0,
    slowperiod: int = 0,
    signalperiod: int = 0,
    suffix: str = "",
) -> pl.Expr:
    """Name a TA indicator expression and lift struct fields when listed in ``unnests``.

    ``window`` accepts either an int (the rolling window) or an empty string meaning "no
    rolling window" (used for cyclic/static indicators).
    """
    if window:
        col = f"{prefix}{func}{window}{suffix}"
    else:
        if not fastperiod:
            col = f"{prefix}{func}"
        else:
            col = f"{prefix}{func}{fastperiod}-{slowperiod}-{signalperiod}"
    expr = expr.over(ticker_column).alias(col)
    if col in unnests:
        return expr.name.map_fields(lambda x: f"{col}_{x}")
    return expr


# TA categories - hoisted to module scope so they're computed once.
_CYCLIC_TA_INDICATORS: Tuple[str, ...] = (
    "ht_dcperiod", "ht_dcphase", "ht_phasor", "ht_sine", "ht_trendmode", "ht_trendline", "mama",
)
_OHLC_ONLY_TA_INDICATORS: Tuple[str, ...] = (
    "bop", "avgprice",
    "cdl2crows", "cdl3blackcrows", "cdl3inside", "cdl3linestrike", "cdl3outside",
    "cdl3starsinsouth", "cdl3whitesoldiers", "cdlabandonedbaby", "cdladvanceblock", "cdlbelthold",
    "cdlbreakaway", "cdlclosingmarubozu", "cdlconcealbabyswall", "cdlcounterattack",
    "cdldarkcloudcover", "cdldoji", "cdldojistar", "cdldragonflydoji", "cdlengulfing",
    "cdleveningdojistar", "cdleveningstar", "cdlgapsidesidewhite", "cdlgravestonedoji",
    "cdlhammer", "cdlhangingman", "cdlharami", "cdlharamicross", "cdlhighwave", "cdlhikkake",
    "cdlhikkakemod", "cdlhomingpigeon", "cdlidentical3crows", "cdlinneck", "cdlinvertedhammer",
    "cdlkicking", "cdlkickingbylength", "cdlladderbottom", "cdllongleggeddoji", "cdllongline",
    "cdlmarubozu", "cdlmatchinglow", "cdlmathold", "cdlmorningdojistar", "cdlmorningstar",
    "cdlonneck", "cdlpiercing", "cdlrickshawman", "cdlrisefall3methods", "cdlseparatinglines",
    "cdlshootingstar", "cdlshortline", "cdlspinningtop", "cdlstalledpattern", "cdlsticksandwich",
    "cdltakuri", "cdltasukigap", "cdlthrusting", "cdltristar", "cdlunique3river",
    "cdlupsidegap2crows", "cdlxsidegap3methods",
)
_CV_ONLY_TA_INDICATORS: Tuple[str, ...] = ("obv",)
_HLC_ONLY_TA_INDICATORS: Tuple[str, ...] = ("typprice", "wclprice", "trange", "stoch", "stochf", "ultosc")
_HLCV_ONLY_TA_INDICATORS: Tuple[str, ...] = ("adosc",)
_HL_ONLY_TA_INDICATORS: Tuple[str, ...] = ("sar", "sarext", "medprice")
_TIMEPERIOD_ONLY_TA_INDICATORS: Tuple[str, ...] = (
    "apo", "cmo", "mom", "rsi", "trix", "bbands", "dema", "ema", "kama", "sma", "t3", "tema",
    "trima", "wma", "midpoint", "ppo", "linearreg", "linearreg_angle", "linearreg_intercept",
    "linearreg_slope", "tsf", "stochrsi",
)
_TIMEPERIOD_HLC_TA_INDICATORS: Tuple[str, ...] = (
    "adx", "adxr", "cci", "dx", "minus_di", "plus_di", "willr", "atr", "natr",
)
_TIMEPERIOD_HL_TA_INDICATORS: Tuple[str, ...] = ("aroon", "aroonosc", "minus_dm", "plus_dm", "midprice")
_TIMEPERIOD_P01_TA_INDICATORS: Tuple[str, ...] = ("correl", "beta")
_TIMEPERIOD_HLCV_TA_INDICATORS: Tuple[str, ...] = ("mfi",)


def _build_unnests(prefix: str, ta_windows: Sequence[int]) -> List[str]:
    """Enumerate every TA-output column that ``apply_ta_indicator`` should unnest into struct fields.

    Pre-computing the full list up-front (rather than appending inside the per-indicator loop)
    fixes a path-dependence bug: earlier ``apply_ta_indicator`` calls used to miss later
    ``unnests.append(...)`` mutations and silently skip the struct-flattening for those indicators.
    """
    names = [f"{prefix}ht_phasor", f"{prefix}ht_sine", f"{prefix}mama", f"{prefix}stoch", f"{prefix}stochf"]
    for window in ta_windows:
        names.extend(
            [
                f"{prefix}aroon{window}",
                f"{prefix}bbands{window}close",
                f"{prefix}bbands{window}volume",
                f"{prefix}stochrsi{window}close",
                f"{prefix}stochrsi{window}volume",
            ]
        )
    return names


def add_ohlcv_ta_indicators(
    ohlcv: pl.DataFrame,
    ta_windows: Optional[Sequence[int]] = None,
    fss_rolling_windows: Optional[Sequence[Sequence[int]]] = None,
    ticker_column: str = "ticker",
    market_action_prefixes: Optional[Sequence[str]] = None,
    ohlcv_fields_mapping: Optional[Dict[str, str]] = None,
    nans_filler: float = 0.0,
    cast_f64_to_f32: bool = True,
) -> pl.DataFrame:
    """Apply a rich set of polars-talib indicators to multi-asset OHLCV.

    The frame must be sorted by timestamp. ``market_action_prefixes`` allows applying TA per
    buy/sell side separately, e.g. ``["", "buy_", "sell_"]``.
    """
    if plta is None:
        raise ImportError(
            "polars_talib is required for add_ohlcv_ta_indicators(); install "
            "mlframe[polars_ext] (which pulls polars-talib + libta-lib) to use this function."
        )
    if not ta_windows:
        ta_windows = list(_DEFAULT_TA_WINDOWS)
    if not fss_rolling_windows:
        fss_rolling_windows = [list(t) for t in _DEFAULT_FSS_WINDOWS]
    if market_action_prefixes is None:
        market_action_prefixes = list(_DEFAULT_MARKET_ACTION_PREFIXES)
    if ohlcv_fields_mapping is None:
        ohlcv_fields_mapping = dict(_DEFAULT_TA_OHLCV_FIELDS)

    ta_expressions: list = []
    unnests: List[str] = []

    for prefix in market_action_prefixes:
        # Build the COMPLETE unnest set BEFORE constructing any apply_ta_indicator expression.
        # The previous code appended inside the loops, so earlier expressions saw a smaller list.
        unnests.extend(_build_unnests(prefix, ta_windows))

    for prefix in market_action_prefixes:
        # Prices are zero-filled here (not forward-filled) because polars-talib indicators apply
        # their own .over(ticker_column) downstream, and a nested window expression
        # (forward-fill-over inside `getattr(close.ta, func)()`) raises
        # `InvalidOperationError: window expression not allowed in aggregation` in polars 1.x.
        # If your input contains sporadic null prices, forward-fill them on the caller side
        # BEFORE passing to add_ohlcv_ta_indicators.
        low = pl.col(f"{prefix}{ohlcv_fields_mapping.get('low')}").fill_null(0.0)
        high = pl.col(f"{prefix}{ohlcv_fields_mapping.get('high')}").fill_null(0.0)
        open_col = pl.col(f"{prefix}{ohlcv_fields_mapping.get('open')}").fill_null(0.0)
        close = pl.col(f"{prefix}{ohlcv_fields_mapping.get('close')}").fill_null(0.0)
        volume = pl.col(f"{prefix}{ohlcv_fields_mapping.get('volume')}").fill_null(0.0)

        ta_expressions.extend(
            [
                apply_ta_indicator(
                    getattr(close.ta, func)(), func=func, window="", ticker_column=ticker_column, unnests=unnests, prefix=prefix
                )
                for func in _CYCLIC_TA_INDICATORS
            ]
        )

        ta_expressions.extend(
            [
                apply_ta_indicator(
                    getattr(plta, func)(open=open_col, high=high, low=low, close=close),
                    func=func,
                    window="",
                    ticker_column=ticker_column,
                    unnests=unnests,
                    prefix=prefix,
                )
                for func in _OHLC_ONLY_TA_INDICATORS
            ]
        )

        ta_expressions.extend(
            [
                apply_ta_indicator(
                    getattr(plta, func)(volume=volume, close=close),
                    func=func,
                    window="",
                    ticker_column=ticker_column,
                    unnests=unnests,
                    prefix=prefix,
                )
                for func in _CV_ONLY_TA_INDICATORS
            ]
        )

        ta_expressions.extend(
            [
                apply_ta_indicator(
                    getattr(plta, func)(high=high, low=low, close=close),
                    func=func,
                    window="",
                    ticker_column=ticker_column,
                    unnests=unnests,
                    prefix=prefix,
                )
                for func in _HLC_ONLY_TA_INDICATORS
            ]
        )

        ta_expressions.extend(
            [
                apply_ta_indicator(
                    getattr(plta, func)(high=high, low=low, close=close, volume=volume),
                    func=func,
                    window="",
                    ticker_column=ticker_column,
                    unnests=unnests,
                    prefix=prefix,
                )
                for func in _HLCV_ONLY_TA_INDICATORS
            ]
        )

        ta_expressions.extend(
            [
                apply_ta_indicator(
                    getattr(plta, func)(high=high, low=low),
                    func=func,
                    window="",
                    ticker_column=ticker_column,
                    unnests=unnests,
                    prefix=prefix,
                )
                for func in _HL_ONLY_TA_INDICATORS
            ]
        )

        for window in ta_windows:
            ta_expressions.extend(
                [
                    apply_ta_indicator(
                        getattr(close.ta, func)(window),
                        func=func,
                        window=window,
                        ticker_column=ticker_column,
                        unnests=unnests,
                        prefix=prefix,
                        suffix="close",
                    )
                    for func in _TIMEPERIOD_ONLY_TA_INDICATORS
                ]
            )
            ta_expressions.extend(
                [
                    apply_ta_indicator(
                        getattr(volume.ta, func)(window),
                        func=func,
                        window=window,
                        ticker_column=ticker_column,
                        unnests=unnests,
                        prefix=prefix,
                        suffix="volume",
                    )
                    for func in _TIMEPERIOD_ONLY_TA_INDICATORS
                ]
            )
            ta_expressions.extend(
                [
                    apply_ta_indicator(
                        getattr(plta, func)(price0=high, price1=low, timeperiod=window),
                        func=func,
                        window=window,
                        ticker_column=ticker_column,
                        unnests=unnests,
                        prefix=prefix,
                    )
                    for func in _TIMEPERIOD_P01_TA_INDICATORS
                ]
            )
            ta_expressions.extend(
                [
                    apply_ta_indicator(
                        getattr(plta, func)(high=high, low=low, timeperiod=window),
                        func=func,
                        window=window,
                        ticker_column=ticker_column,
                        unnests=unnests,
                        prefix=prefix,
                    )
                    for func in _TIMEPERIOD_HL_TA_INDICATORS
                ]
            )
            ta_expressions.extend(
                [
                    pllib.clean_numeric(
                        apply_ta_indicator(
                            getattr(plta, func)(high=high, low=low, close=close, timeperiod=window),
                            func=func,
                            window=window,
                            ticker_column=ticker_column,
                            unnests=unnests,
                            prefix=prefix,
                        ),
                        nans_filler=nans_filler,
                    )
                    for func in _TIMEPERIOD_HLC_TA_INDICATORS
                ]
            )
            ta_expressions.extend(
                [
                    apply_ta_indicator(
                        getattr(plta, func)(high=high, low=low, close=close, volume=volume, timeperiod=window),
                        func=func,
                        window=window,
                        ticker_column=ticker_column,
                        unnests=unnests,
                        prefix=prefix,
                    )
                    for func in _TIMEPERIOD_HLCV_TA_INDICATORS
                ]
            )

        # The MACD/FSS block was historically present but disabled (empty indicator list); leave
        # the for-loop in place so adding indicators later requires only one edit.
        timeperiod_only_fss_indicators: Tuple[str, ...] = ()
        for fastperiod, slowperiod, signalperiod in fss_rolling_windows:
            ta_expressions.extend(
                [
                    apply_ta_indicator(
                        getattr(close.ta, func)(
                            fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod
                        ),
                        func=func,
                        window=0,
                        fastperiod=fastperiod,
                        slowperiod=slowperiod,
                        signalperiod=signalperiod,
                        ticker_column=ticker_column,
                        unnests=unnests,
                        prefix=prefix,
                    )
                    for func in timeperiod_only_fss_indicators
                ]
            )

    res = ohlcv.with_columns(ta_expressions)

    if unnests:
        res = res.unnest(unnests)

    # fill_null then fill_nan: polars distinguishes the two; TA warmup periods produce nulls
    # while divide-by-zero in some indicators produces NaN. Both must be scrubbed.
    res = res.with_columns(cs.numeric().fill_null(nans_filler).fill_nan(nans_filler))

    if cast_f64_to_f32:
        res = pllib.cast_f64_to_f32(res)

    return res


def create_ohlcv_wholemarket_features(
    ohlcv: pl.DataFrame,
    timestamp_column: str = "date",
    exclude_fields: Optional[Sequence[str]] = None,
    weighting_columns: Optional[Sequence[str]] = None,
    numaggs: Optional[Sequence[str]] = None,
    nans_filler: float = 0.0,
    cast_f64_to_f32: bool = True,
) -> pl.DataFrame:
    """Cross-ticker aggregates (min/max/std/mean/quantiles + weighted variants) per timestamp.

    Should run AFTER ``add_ohlcv_ratios_rlags`` and ``add_ohlcv_ta_indicators`` so that all derived
    columns are included. The result is joined back per-bar via timestamp; ranks
    ``(val - min)/(max - min)`` can be added on the joined frame.
    """
    if weighting_columns is None:
        weighting_columns = list(_DEFAULT_WEIGHTING_COLUMNS)
    if not numaggs:
        numaggs = list(_DEFAULT_WHOLEMARKET_NUMAGGS)

    all_num_cols = cs.numeric()
    if exclude_fields:
        all_num_cols = all_num_cols - cs.by_name(exclude_fields)

    wcols = pllib.add_weighted_aggregates(
        columns_selector=all_num_cols, weighting_columns=weighting_columns, fpref="wm_"
    )

    res = ohlcv.group_by(timestamp_column).agg(
        [pl.len().alias("wm_size")]
        + [getattr(all_num_cols, func)().name.suffix(f"_wm_{func}") for func in numaggs]
        + wcols
    )
    # cs.float() (not cs.numeric()) - clean_numeric returns Float64 via pl.when/otherwise, which
    # would promote integer timestamp columns to f64 and break downstream joins where the
    # per-ticker frame still has int64 timestamps. Integer columns can't contain NaN/inf
    # anyway, so cleaning only floats is correct and dtype-preserving.
    res = res.with_columns(pllib.clean_numeric(cs.float(), nans_filler=nans_filler))
    if cast_f64_to_f32:
        res = pllib.cast_f64_to_f32(res)

    return res.sort(timestamp_column)


def merge_perticker_and_wholemarket_features(
    perticker_features: Union[pl.DataFrame, pl.LazyFrame],
    wholemarket_features: Union[pl.DataFrame, pl.LazyFrame],
    timestamp_column: str = "date",
    add_rankings: bool = True,
) -> pl.DataFrame:
    """Join per-ticker features with whole-market aggregates and add ``(val-min)/(max-min)`` ranks."""
    rankings: list = []
    if add_rankings:
        wholemarket_cols = set(wholemarket_features.collect_schema().names())
        for col in perticker_features.collect_schema().names():
            if f"{col}_wm_min" in wholemarket_cols and f"{col}_wm_max" in wholemarket_cols:
                rankings.append(
                    pllib.clean_numeric(  # market-constant column → wm_max==wm_min → division by zero yields inf/NaN
                        (pl.col(col) - pl.col(f"{col}_wm_min")) / (pl.col(f"{col}_wm_max") - pl.col(f"{col}_wm_min"))
                    )
                    .alias(f"{col}_wm_rnk")
                )

    joined = perticker_features.join(wholemarket_features, on=timestamp_column, how="left").sort(timestamp_column)

    if rankings:
        joined = joined.with_columns(rankings)

    clean_ram()
    return joined.collect() if isinstance(joined, pl.LazyFrame) else joined
