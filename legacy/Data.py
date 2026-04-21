
########################################################################################################################################################################################################################################
# Load data
########################################################################################################################################################################################################################################
import pandas as pd


# Load data from csv
def LoadDataFromCsv(sHistoricalDataPath, sTicker, lTimeFrameInMinutes):
    ds = pd.read_csv(
        open(sHistoricalDataPath + sTicker + '_' + str(lTimeFrameInMinutes) + '_2010-01-01_2017-07-01', 'r'));
    del ds[ds.columns[0]];
    ds['date'] = ds['date'].astype('datetime64[ns]');
    return ds


# Download different timeframes and currency pairs from Poloniex website and save them on the disk as csv format

def DownloadCurrencyPairCandlesPoloniex(dtFrom, dtTo, sCurrencyPair, lPeriod):
    tmp_pd = pd.read_json(
        'https://poloniex.com/public?command=returnChartData&currencyPair=' + sCurrencyPair + '&start=' + str(
            int(time.mktime(dtFrom.timetuple()))) + '&end=' + str(
            int(time.mktime(dtTo.timetuple()))) + '&period=' + str(lPeriod));
    tmp_pd.to_csv('CryptoCurrency\\history\\candles\poloniex\\' + sCurrencyPair + '_' + str(lPeriod) + '_' + str(
        dtFrom) + '_' + str(dtTo))


def DownloadAllPoloniexCandles():
    pdTickers = pd.read_json('https://poloniex.com/public?command=returnTicker');
    for NextTicker in pdTickers.columns:
        print(NextTicker)
        if (NextTicker > 'BTC_FLDC'):
            for lPeriod in [300, 900, 1800, 7200, 14400, 86400]:
                print('\t ' + str(lPeriod))
                DownloadCurrencyPairCandlesPoloniex(datetime.date(2010, 1, 1), datetime.date(2017, 7, 1), NextTicker,
                                                    lPeriod)