"""Features for financial modelling."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import polars_talib as plta
import polars as pl, polars.selectors as cs

import pyutilz.polarslib as pllib

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def add_ohlcv_ratios_rlags_rollings(
    ohlcv: pl.DataFrame,
    columns_selector: str = "",
    lags: list = [1],
    rolling_windows: list = [5],
    crossbar_ratios_lags: list = [1],
    min_samples: int = 1,
    ticker_column: str = "ticker",
    target_columns_shift: int = 1,
    target_columns_prefix: str = "target",
    market_action_prefixes: list = [""],
    ohlcv_fields_mapping: dict = dict(qty="qty", open="open", high="high", low="low", close="close", volume="volume"),
    nans_filler: float = 0.0,
    cast_f64_to_f32: bool = True,
) -> pl.DataFrame:
    """Adds more nuanced features to raw ohlcv."""

    all_num_cols = cs.numeric()
    if columns_selector:
        all_num_cols = all_num_cols & cs.contains(columns_selector)
    if target_columns_prefix:
        all_num_cols = all_num_cols - cs.starts_with(target_columns_prefix)

    interbar_ratios_features = []
    for prefix in market_action_prefixes:

        qty = pl.col(f"{prefix}{ohlcv_fields_mapping.get('qty')}")
        low = pl.col(f"{prefix}{ohlcv_fields_mapping.get('low')}")
        high = pl.col(f"{prefix}{ohlcv_fields_mapping.get('high')}")
        open = pl.col(f"{prefix}{ohlcv_fields_mapping.get('open')}")
        close = pl.col(f"{prefix}{ohlcv_fields_mapping.get('close')}")
        volume = pl.col(f"{prefix}{ohlcv_fields_mapping.get('volume')}")

        interbar_ratios_features.extend(
            [
                (close / open - 1).alias(f"{prefix}close_to_open"),
                (high / open - 1).alias(f"{prefix}high_to_open"),
                (open / low - 1).alias(f"{prefix}open_to_low"),
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
                        (close / open.shift(period_shift).over(ticker_column) - 1).alias(f"{prefix}close_to_open-{period_shift}"),
                        (high / open.shift(period_shift).over(ticker_column) - 1).alias(f"{prefix}high_to_open-{period_shift}"),
                        (open / low.shift(period_shift).over(ticker_column) - 1).alias(f"{prefix}open_to_low-{period_shift}"),
                        (close / low.shift(period_shift).over(ticker_column) - 1).alias(f"{prefix}close_to_low-{period_shift}"),
                        (high / low.shift(period_shift).over(ticker_column) - 1).alias(f"{prefix}high_to_low-{period_shift}"),
                        (high / close.shift(period_shift).over(ticker_column) - 1).alias(f"{prefix}high_to_close-{period_shift}"),
                        pllib.clean_numeric((volume / qty.shift(period_shift).over(ticker_column)), nans_filler=nans_filler).alias(
                            f"{prefix}avg_trade_size-{period_shift}"
                        ),
                    ]
                )

    def group_if_needed(expr: pl.Expr, over: str = "") -> pl.Expr:
        return expr.over(over) if over else expr

    prevtarget_columns = []
    if ticker_column and target_columns_prefix:
        prevtarget_columns.append(
            cs.starts_with(target_columns_prefix).shift(target_columns_shift).over(ticker_column).name.prefix(f"prev{target_columns_shift}")
        )

    ohlcv = ohlcv.with_columns(
        *prevtarget_columns,
        # interbar ohlcv ratios features
        *interbar_ratios_features,
    ).with_columns(
        # relative lags
        *[
            pllib.clean_numeric((all_num_cols / group_if_needed(all_num_cols.shift(lag), over=ticker_column) - 1), nans_filler=nans_filler).name.suffix(
                f"_rlag{lag}"
            )
            for lag in lags
        ],
        # relative means over rolling_windows
        *[
            pllib.clean_numeric(
                (all_num_cols / group_if_needed(all_num_cols.rolling_mean(window, min_samples=min_samples), over=ticker_column) - 1), nans_filler=nans_filler
            ).name.suffix(f"_rmean{window}")
            for window in rolling_windows
        ],
        # relative standard deviations over rolling_windows
        *[
            pllib.clean_numeric(
                (group_if_needed(all_num_cols.rolling_std(window, min_samples=min_samples), over=ticker_column) / all_num_cols), nans_filler=nans_filler
            ).name.suffix(f"_rstd{window}")
            for window in rolling_windows
        ],
    )

    if cast_f64_to_f32:
        ohlcv = pllib.cast_f64_to_f32(ohlcv)

    return ohlcv


def apply_ta_indicator(
    expr: pl.Expr,
    func: str,
    window: int,
    ticker_column: str,
    unnests: list,
    prefix: str,
    fastperiod: int = 0,
    slowperiod: int = 0,
    signalperiod: int = 0,
    suffix: str = "",
) -> pl.Expr:
    """Decides if fields are struct and need prefixing.
    Also creates common naming for TA indicators applied over specific rolling_windows, applies grouping."""
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
    else:
        return expr


def add_ohlcv_ta_indicators(
    ohlcv: pl.DataFrame,
    rolling_windows: list = [5, 10],
    fss_rolling_windows=[[12, 26, 9]],
    ticker_column: str = "ticker",
    market_action_prefixes: list = [""],
    ohlcv_fields_mapping: dict = dict(open="open", high="high", low="low", close="close", volume="volume"),
    nans_filler: float = 0.0,
    cast_f64_to_f32: bool = True,
) -> pl.DataFrame:
    """Applies a rich set of Technical Analysis indicators from polars-talib to ohlcv data of multiple assets.
    ohlcv dataframe must be sorted by timestamp.
    market_action_prefixes allow to apply TA per buy/sell groups separately: market_action_prefixes = ["", "buy_", "sell_"]
    """
    ta_expressions = []
    unnests = []

    for prefix in market_action_prefixes:
        low = pl.col(f"{prefix}{ohlcv_fields_mapping.get('low')}")
        high = pl.col(f"{prefix}{ohlcv_fields_mapping.get('high')}")
        open = pl.col(f"{prefix}{ohlcv_fields_mapping.get('open')}")
        close = pl.col(f"{prefix}{ohlcv_fields_mapping.get('close')}")
        volume = pl.col(f"{prefix}{ohlcv_fields_mapping.get('volume')}")

        cyclic_indicators = "ht_dcperiod ht_dcphase ht_phasor ht_sine ht_trendmode ht_trendline mama".split()
        unnests.extend([f"{prefix}ht_phasor", f"{prefix}ht_sine"])
        unnests.extend(f"{prefix}mama".split())
        ta_expressions.extend(
            [
                apply_ta_indicator(getattr(close.ta, func)(), func=func, window="", ticker_column=ticker_column, unnests=unnests, prefix=prefix)
                for func in cyclic_indicators
            ]
        )

        ohlc_only_indicators = "bop avgprice".split()

        ohlc_only_indicators = ohlc_only_indicators + [
            "cdl2crows",
            "cdl3blackcrows",
            "cdl3inside",
            "cdl3linestrike",
            "cdl3outside",
            "cdl3starsinsouth",
            "cdl3whitesoldiers",
            "cdlabandonedbaby",
            "cdladvanceblock",
            "cdlbelthold",
            "cdlbreakaway",
            "cdlclosingmarubozu",
            "cdlconcealbabyswall",
            "cdlcounterattack",
            "cdldarkcloudcover",
            "cdldoji",
            "cdldojistar",
            "cdldragonflydoji",
            "cdlengulfing",
            "cdleveningdojistar",
            "cdleveningstar",
            "cdlgapsidesidewhite",
            "cdlgravestonedoji",
            "cdlhammer",
            "cdlhangingman",
            "cdlharami",
            "cdlharamicross",
            "cdlhighwave",
            "cdlhikkake",
            "cdlhikkakemod",
            "cdlhomingpigeon",
            "cdlidentical3crows",
            "cdlinneck",
            "cdlinvertedhammer",
            "cdlkicking",
            "cdlkickingbylength",
            "cdlladderbottom",
            "cdllongleggeddoji",
            "cdllongline",
            "cdlmarubozu",
            "cdlmatchinglow",
            "cdlmathold",
            "cdlmorningdojistar",
            "cdlmorningstar",
            "cdlonneck",
            "cdlpiercing",
            "cdlrickshawman",
            "cdlrisefall3methods",
            "cdlseparatinglines",
            "cdlshootingstar",
            "cdlshortline",
            "cdlspinningtop",
            "cdlstalledpattern",
            "cdlsticksandwich",
            "cdltakuri",
            "cdltasukigap",
            "cdlthrusting",
            "cdltristar",
            "cdlunique3river",
            "cdlupsidegap2crows",
            "cdlxsidegap3methods",
        ]

        ta_expressions.extend(
            [
                apply_ta_indicator(
                    getattr(plta, func)(open=open, high=high, low=low, close=close),
                    func=func,
                    window="",
                    ticker_column=ticker_column,
                    unnests=unnests,
                    prefix=prefix,
                )
                for func in ohlc_only_indicators
            ]
        )

        cv_only_indicators = "obv".split()
        ta_expressions.extend(
            [
                apply_ta_indicator(
                    getattr(plta, func)(volume=volume, close=close), func=func, window="", ticker_column=ticker_column, unnests=unnests, prefix=prefix
                )
                for func in cv_only_indicators
            ]
        )

        hlc_only_indicators = "typprice wclprice trange stoch stochf ultosc".split()
        ta_expressions.extend(
            [
                apply_ta_indicator(
                    getattr(plta, func)(high=high, low=low, close=close), func=func, window="", ticker_column=ticker_column, unnests=unnests, prefix=prefix
                )
                for func in hlc_only_indicators
            ]
        )
        unnests.append(f"{prefix}stoch")
        unnests.append(f"{prefix}stochf")

        hlcv_only_indicators = "adosc".split()
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
                for func in hlcv_only_indicators
            ]
        )

        hl_only_indicators = "sar sarext medprice".split()
        ta_expressions.extend(
            [
                apply_ta_indicator(getattr(plta, func)(high=high, low=low), func=func, window="", ticker_column=ticker_column, unnests=unnests, prefix=prefix)
                for func in hl_only_indicators
            ]
        )

        excluded = "mavp roc stoch"
        simplified = "bbands t3 apo sar sarext ppo stochrsi adosc"  # Can be improved with more parameters

        timeperiod_only_indicators = "apo cmo mom rsi trix bbands dema ema kama sma t3 tema trima wma midpoint ppo linearreg linearreg_angle linearreg_intercept linearreg_slope tsf stochrsi".split()
        timeperiod_hlc_indicators = "adx adxr cci dx minus_di plus_di willr atr natr".split()
        timeperiod_hl_indicators = "aroon aroonosc minus_dm plus_dm midprice".split()
        timeperiod_p01_indicators = "correl beta".split()
        timeperiod_hlcv_indicators = "mfi".split()

        for window in rolling_windows:

            unnests.append(f"{prefix}aroon{window}")
            unnests.append(f"{prefix}bbands{window}close")
            unnests.append(f"{prefix}bbands{window}volume")
            unnests.append(f"{prefix}stochrsi{window}close")
            unnests.append(f"{prefix}stochrsi{window}volume")

            ta_expressions.extend(
                [
                    *[
                        apply_ta_indicator(
                            getattr(close.ta, func)(window),
                            func=func,
                            window=window,
                            ticker_column=ticker_column,
                            unnests=unnests,
                            prefix=prefix,
                            suffix="close",
                        )
                        for func in timeperiod_only_indicators
                    ],
                    *[
                        apply_ta_indicator(
                            getattr(volume.ta, func)(window),
                            func=func,
                            window=window,
                            ticker_column=ticker_column,
                            unnests=unnests,
                            prefix=prefix,
                            suffix="volume",
                        )
                        for func in timeperiod_only_indicators
                    ],
                    *[
                        apply_ta_indicator(
                            getattr(plta, func)(price0=high, price1=low, timeperiod=window),
                            func=func,
                            window=window,
                            ticker_column=ticker_column,
                            unnests=unnests,
                            prefix=prefix,
                        )
                        for func in timeperiod_p01_indicators
                    ],
                    *[
                        apply_ta_indicator(
                            getattr(plta, func)(high=high, low=low, timeperiod=window),
                            func=func,
                            window=window,
                            ticker_column=ticker_column,
                            unnests=unnests,
                            prefix=prefix,
                        )
                        for func in timeperiod_hl_indicators
                    ],
                    *[
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
                        for func in timeperiod_hlc_indicators
                    ],
                    *[
                        apply_ta_indicator(
                            getattr(plta, func)(high=high, low=low, close=close, volume=volume, timeperiod=window),
                            func=func,
                            window=window,
                            ticker_column=ticker_column,
                            unnests=unnests,
                            prefix=prefix,
                        )
                        for func in timeperiod_hlcv_indicators
                    ],
                ]
            )

        timeperiod_only_fss_indicators = []  # "macd".split()
        for fastperiod, slowperiod, signalperiod in fss_rolling_windows:
            # unnests.append(f"macd{fastperiod}-{slowperiod}-{signalperiod}")
            ta_expressions.extend(
                [
                    *[
                        apply_ta_indicator(
                            getattr(close.ta, func)(fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod),
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
                    ],
                ]
            )

    res = ohlcv.with_columns(ta_expressions)

    if unnests:
        res = res.unnest(unnests)

    res = res.with_columns(cs.numeric().fill_nan(nans_filler))

    if cast_f64_to_f32:
        res = pllib.cast_f64_to_f32(res)

    return res


def create_ohlcv_wholemarket_features(
    ohlcv: pl.DataFrame,
    timestamp_column: str = "date",
    target_columns_prefix: str = "target",
    weighting_columns: list = "volume qty".split(),
    numaggs: list = "min max mean median std skew kurtosis entropy n_unique".split(),
    nans_filler: float = 0.0,
    cast_f64_to_f32: bool = True,
) -> pl.DataFrame:
    """For all columns of a bar, regardless of a ticker, finds min, max, std, quantiles, etc.
    Also performs a few weighted calculations (for mean and std).
    Should be applied AFTER add_ohlcv_ratios_rlags_rollings and add_ohlcv_ta_indicators, to cover as many columns as possible.
    Then joined with main ohlcv by bar's timestamp (ideally ranks across tickers should be added: (val-min)/(max-min)), using cs.expand_selector(ohlcv,cs.numeric()) to get exact col names..
    Can rlags & rolling means be applied to wholemarket features also, after everything else???
    """
    all_num_cols = cs.numeric()
    if target_columns_prefix:
        all_num_cols = all_num_cols - cs.starts_with(target_columns_prefix)

    wcols = []
    for wcol in weighting_columns:
        all_other_num_cols = all_num_cols - pl.col(wcol)
        weighted_mean = ((all_other_num_cols * pl.col(wcol)).sum() / pl.col(wcol).sum()).name.suffix(f"_wmean_{wcol}")
        wcols.append(weighted_mean)
        # !TODO causes error for now
        # weighted_std = ((pl.col(wcol) * (all_other_num_cols - weighted_mean) ** 2).sum() / pl.col(wcol).sum()).sqrt().name.suffix(f"_wstd_{wcol}")
        # wcols.append(weighted_std)

    res = ohlcv.group_by(timestamp_column).agg(
        [pl.len().alias("wm_ntickers")] + [getattr(all_num_cols, func)().name.suffix(f"_wm_{func}") for func in numaggs] + wcols
    )
    res = res.with_columns(pllib.clean_numeric(cs.float(), nans_filler=nans_filler))
    if cast_f64_to_f32:
        res = pllib.cast_f64_to_f32(res)

    return res.sort(timestamp_column)
