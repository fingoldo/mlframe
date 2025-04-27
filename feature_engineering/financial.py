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

import polars as pl

import polars_talib as plta


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
    Also creates common naming for TA indicators applied over specific windows, applies grouping."""
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


def add_ta_features(
    ohlcv: pl.DataFrame,
    windows: list = [5, 10],
    fss_windows=[[12, 26, 9]],
    ticker_column: str = "ticker",
    market_action_prefixes: list = [""],
    ohlcv_fields_mapping: dict = dict(open="open", high="high", low="low", close="close", volume="volume"),
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
        unnests.extend(f"{prefix}mama".split())
        ta_expressions.extend(
            [
                apply_ta_indicator(getattr(close.ta, func)(), func=func, window="", ticker_column=ticker_column, unnests=unnests, prefix=prefix)
                for func in cyclic_indicators
            ]
        )

        ohlc_only_indicators = "bop avgprice".split() + [
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

        unnests.extend([f"{prefix}ht_phasor", f"{prefix}ht_sine"])

        excluded = "mavp roc stoch"
        simplified = "bbands t3 apo sar sarext ppo stochrsi adosc"  # Can be improved with more parameters

        timeperiod_only_indicators = "apo cmo mom rsi trix bbands dema ema kama sma t3 tema trima wma midpoint ppo linearreg linearreg_angle linearreg_intercept linearreg_slope tsf stochrsi".split()
        timeperiod_hlc_indicators = "adx adxr cci dx minus_di plus_di willr atr natr".split()
        timeperiod_hl_indicators = "aroon aroonosc minus_dm plus_dm midprice".split()
        timeperiod_p01_indicators = "correl beta".split()
        timeperiod_hlcv_indicators = "mfi".split()

        for window in windows:

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
                        apply_ta_indicator(
                            getattr(plta, func)(high=high, low=low, close=close, timeperiod=window),
                            func=func,
                            window=window,
                            ticker_column=ticker_column,
                            unnests=unnests,
                            prefix=prefix,
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
        for fastperiod, slowperiod, signalperiod in fss_windows:
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

    return res.with_columns(pl.col(pl.Float64).cast(pl.Float32))
