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
from pyutilz.system import clean_ram


# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def add_ohlcv_ratios_rlags(
    ohlcv: pl.DataFrame,
    columns_selector: str = "",
    lags: list = None,
    crossbar_ratios_lags: list = None,
    add_ratios: bool = True,
    add_rlags: bool = True,
    ticker_column: str = "ticker",
    exclude_fields: list = None,
    market_action_prefixes: list = None,
    ohlcv_fields_mapping=None,
    nans_filler: float = 0.0,
    cast_f64_to_f32: bool = True,
) -> pl.DataFrame:
    """Adds more nuanced features to raw ohlcv. Dataframe assumed to be sorted by timestamp.
    Grouping implemented with 'over' mechanics."""

    # ----------------------------------------------------------------------------------------------------------------------------
    # Inits
    # ----------------------------------------------------------------------------------------------------------------------------

    if lags is None:
        lags: list = [1]
    if crossbar_ratios_lags is None:
        crossbar_ratios_lags: list = [1]
    if not market_action_prefixes:
        market_action_prefixes: list = [""]
    if not ohlcv_fields_mapping:
        ohlcv_fields_mapping: dict = dict(qty="qty", open="open", high="high", low="low", close="close", volume="volume")

    # ----------------------------------------------------------------------------------------------------------------------------
    # Columns to work with
    # ----------------------------------------------------------------------------------------------------------------------------

    all_num_cols = cs.numeric()
    if columns_selector:
        all_num_cols = all_num_cols & cs.contains(columns_selector)
    if exclude_fields:
        all_num_cols = all_num_cols - cs.by_name(exclude_fields)

    # ----------------------------------------------------------------------------------------------------------------------------
    # Ratios
    # ----------------------------------------------------------------------------------------------------------------------------

    if add_ratios:
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

    # ----------------------------------------------------------------------------------------------------------------------------
    # Computing
    # ----------------------------------------------------------------------------------------------------------------------------

    if add_ratios:
        ohlcv = ohlcv.with_columns(
            # interbar ohlcv ratios features
            *interbar_ratios_features,
        )

    if add_rlags:

        def group_if_needed(expr: pl.Expr, over: str = "") -> pl.Expr:
            return expr.over(over) if over else expr

        ohlcv = ohlcv.with_columns(
            # relative lags
            *[
                pllib.clean_numeric((all_num_cols / group_if_needed(all_num_cols.shift(lag), over=ticker_column) - 1), nans_filler=nans_filler).name.suffix(
                    f"_rlag{lag}"
                )
                for lag in lags
            ],
        )

    if cast_f64_to_f32:
        ohlcv = pllib.cast_f64_to_f32(ohlcv)

    return ohlcv


def add_fast_rolling_stats(
    df: pl.DataFrame,
    columns_selector: str = None,
    rolling_windows: list = None,
    numaggs: list = None,
    quantiles: list = None,
    relative: bool = True,
    min_samples: int = 1,
    groupby_column: str = None,
    exclude_fields: list = None,
    nans_filler: float = 0.0,
    cast_f64_to_f32: bool = True,
) -> pl.DataFrame:
    """Adds more nuanced features to raw df. Dataframe assumed to be sorted by timestamp.
    Grouping implemented with 'over' mechanics."""

    # ----------------------------------------------------------------------------------------------------------------------------
    # Inits
    # ----------------------------------------------------------------------------------------------------------------------------

    if not rolling_windows:
        rolling_windows: list = [5]

    if numaggs is None:
        numaggs: list = "rolling_min rolling_max rolling_mean rolling_std rolling_skew rolling_kurtosis".split()

    if quantiles is None:
        quantiles: list = [0.1, 0.25, 0.5, 0.75, 0.9]

    # ----------------------------------------------------------------------------------------------------------------------------
    # Columns to work with
    # ----------------------------------------------------------------------------------------------------------------------------

    all_num_cols = cs.numeric()
    if columns_selector:
        all_num_cols = all_num_cols & cs.contains(columns_selector)
    if exclude_fields:
        all_num_cols = all_num_cols - cs.by_name(exclude_fields)

    def group_if_needed(expr: pl.Expr, over: str = "") -> pl.Expr:
        return expr.over(over) if over else expr

    # ----------------------------------------------------------------------------------------------------------------------------
    # Computing
    # ----------------------------------------------------------------------------------------------------------------------------

    exprs = []
    for func in numaggs:
        if relative:
            exprs.extend(
                [
                    pllib.clean_numeric(
                        (all_num_cols / group_if_needed(get(all_num_cols, func)(window, min_samples=min_samples), over=groupby_column) - 1),
                        nans_filler=nans_filler,
                    ).name.suffix(f"_r{func.replace('rolling_','')}{window}")
                    for window in rolling_windows
                ]
            )
        else:
            exprs.extend(
                [
                    group_if_needed(get(all_num_cols, func)(window, min_samples=min_samples), over=groupby_column).name.suffix(
                        f"_{func.replace('rolling_','')}{window}"
                    )
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
    ta_windows: list = None,
    fss_rolling_windows=None,
    ticker_column: str = "ticker",
    market_action_prefixes: list = None,
    ohlcv_fields_mapping: dict = None,
    nans_filler: float = 0.0,
    cast_f64_to_f32: bool = True,
) -> pl.DataFrame:
    """Applies a rich set of Technical Analysis indicators from polars-talib to ohlcv data of multiple assets.
    ohlcv dataframe must be sorted by timestamp.
    market_action_prefixes allow to apply TA per buy/sell groups separately: market_action_prefixes = ["", "buy_", "sell_"]
    """

    # ----------------------------------------------------------------------------------------------------------------------------
    # Inits
    # ----------------------------------------------------------------------------------------------------------------------------

    if not ta_windows:
        ta_windows: list = [5, 10]
    if not fss_rolling_windows:
        fss_rolling_windows = [[12, 26, 9]]
    if not market_action_prefixes:
        market_action_prefixes: list = [""]
    if not ohlcv_fields_mapping:
        ohlcv_fields_mapping: dict = dict(open="open", high="high", low="low", close="close", volume="volume")

    ta_expressions = []
    unnests = []

    for prefix in market_action_prefixes:

        low = pl.col(f"{prefix}{ohlcv_fields_mapping.get('low')}").fill_null(strategy="forward").over(ticker_column).fill_null(0.0).over(ticker_column)
        high = pl.col(f"{prefix}{ohlcv_fields_mapping.get('high')}").fill_null(strategy="forward").over(ticker_column).fill_null(0.0).over(ticker_column)
        open = pl.col(f"{prefix}{ohlcv_fields_mapping.get('open')}").fill_null(strategy="forward").over(ticker_column).fill_null(0.0).over(ticker_column)
        close = pl.col(f"{prefix}{ohlcv_fields_mapping.get('close')}").fill_null(strategy="forward").over(ticker_column).fill_null(0.0).over(ticker_column)
        volume = pl.col(f"{prefix}{ohlcv_fields_mapping.get('volume')}").fill_null(strategy="forward").over(ticker_column).fill_null(0.0).over(ticker_column)

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

        for window in ta_windows:

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
    exclude_fields: list = None,
    weighting_columns: list = None,
    numaggs: list = None,
    nans_filler: float = 0.0,
    cast_f64_to_f32: bool = True,
) -> pl.DataFrame:
    """For all columns of a bar, regardless of a ticker, finds min, max, std, quantiles, etc.
    Also performs a few weighted calculations (for mean and std).
    Should be applied AFTER add_ohlcv_ratios_rlags_rollings and add_ohlcv_ta_indicators, to cover as many columns as possible.
    Then joined with main ohlcv by bar's timestamp (ideally ranks across tickers should be added: (val-min)/(max-min)), using cs.expand_selector(ohlcv,cs.numeric()) to get exact col names..
    rlags & rolling means can be applied one more time to wholemarket features also, after everything else.
    """

    # ----------------------------------------------------------------------------------------------------------------------------
    # Inits
    # ----------------------------------------------------------------------------------------------------------------------------

    if weighting_columns is None:
        weighting_columns: list = "volume qty".split()
    if not numaggs:
        numaggs: list = "min max mean median std skew kurtosis entropy n_unique".split()

    all_num_cols = cs.numeric()
    if exclude_fields:
        all_num_cols = all_num_cols - cs.by_name(exclude_fields)

    wcols = pllib.add_weighted_aggregates(columns_selector=all_num_cols, weighting_columns=weighting_columns, fpref="wm_")  # .name.suffix(f"_wm_")

    res = ohlcv.group_by(timestamp_column).agg(
        [pl.len().alias("wm_size")] + [getattr(all_num_cols, func)().name.suffix(f"_wm_{func}") for func in numaggs] + wcols
    )
    res = res.with_columns(pllib.clean_numeric(cs.float(), nans_filler=nans_filler))
    if cast_f64_to_f32:
        res = pllib.cast_f64_to_f32(res)

    return res.sort(timestamp_column)


def merge_perticker_and_wholemarket_features(
    perticker_features: pl.DataFrame,
    wholemarket_features: pl.DataFrame,
    timestamp_column: str = "date",
    add_rankings: bool = True,
) -> pl.DataFrame:
    """Merges per-ticker and wholemarket features. Add ranks using formula  (val-min)/(max-min)."""

    rankings = []
    if add_rankings:
        wholemarket_cols = set(wholemarket_features.collect_schema().names())
        for col in perticker_features.collect_schema().names():
            if f"{col}_wm_min" in wholemarket_cols and f"{col}_wm_max" in wholemarket_cols:
                rankings.append(((pl.col(col) - pl.col(f"{col}_wm_min")) / (pl.col(f"{col}_wm_max") - pl.col(f"{col}_wm_min"))).alias(f"{col}_wm_rnk"))

    clean_ram()
    joined = perticker_features.join(wholemarket_features, on=timestamp_column, how="left").sort(timestamp_column)
    clean_ram()

    if rankings:
        joined = joined.with_columns(rankings)
        clean_ram()

    return joined.collect()
