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
import mlflow

########################################################################################################################################################################################################################################
# Helper functions
########################################################################################################################################################################################################################################

def MakeSureBlasAndLaPackAreInstalled():
    from numpy.distutils.system_info import get_info
    print(get_info('blas_opt'))
    print(get_info('lapack_opt'))


def ListAllSkLearnClassifiers():
    from sklearn.utils.testing import all_estimators
    for name, Class in all_estimators():
        if name.find('Class') > 0:
            print(Class.__module__, name)


def PrintTimeSeriesSplitExample():
    tscv = TimeSeriesSplit(n_splits=3, max_train_size=50)
    TimeSeriesSplit(n_splits=3)
    for train, test in tscv.split(range(100)):
        print("%s %s" % (train, test))

        # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24] [25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49]
        # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49] [50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74]
        # [25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74] [75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99]


########################################################################################################################################################################################################################################
# Assessing output distribution
########################################################################################################################################################################################################################################

def PlotTargetClassesDistribution(y):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.hist(y);
    print(np.mean(y))


def test_stationarity(timeseries, w):
    import pandas as pd
    from statsmodels.tsa.stattools import adfuller

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    if dfoutput[0] < dftest[4]['1%']:
        print('This time series is stationary')
    else:
        print('This time series is NON-stationary')

########################################################################################################################################################################################################################################
# MLFLOW
########################################################################################################################################################################################################################################

def flatten_classification_report(cr: dict, separate_metrics=("accuracy",),source:str="")->dict:
    res={}
    for metric in separate_metrics:
        if metric in cr:
            res[source+metric]= cr.pop(metric)
    for class_or_avg, metrics_dict in cr.items():
        prefix=class_or_avg if class_or_avg in ('macro avg', 'weighted avg') else 'class '+str(class_or_avg)
        for metric, value in metrics_dict.items():
            res[source+prefix + "_" + metric]= value
    return res
            
def log_classification_report_to_mlflow(cr: dict, step: int,separate_metrics=("accuracy",),source:str=""):
    """Logging all metrics from a dict-like classification_report as flat MLFlow entries."""

    for metric in separate_metrics:
        if metric in cr:
            mlflow.log_metric(source+metric, cr.pop(metric), step=step)
    for class_or_avg, metrics_dict in cr.items():
        prefix=class_or_avg if class_or_avg in ('macro avg', 'weighted avg') else 'class '+str(class_or_avg)
        for metric, value in metrics_dict.items():
            mlflow.log_metric(source+prefix + "_" + metric, value, step=step)

def embed_website_to_mlflow(url:str,fname:str="url",extension:str='.html',width:int=700,height:int=450):
    """Creates a html file with desired url embedded to be shown nicely in MLFlow UI."""

    website_embed = f'''<!DOCTYPE html>
    <html>
    <iframe src="{url}" style='width: {width}px; height: {height}px' sandbox='allow-same-origin allow-scripts'>
    </iframe>
    </html>'''

    if fname[:-len(extension)].lower()==extension:
        extension=""

    with open(fname+extension, "w") as f:
        f.write(website_embed)   