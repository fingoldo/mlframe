"""Statistics used in ML."""

# pylint: disable=wrong-import-order,wrong-import-position,unidiomatic-typecheck,pointless-string-statement

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.pythonlib import ensure_installed  # lint: disable=ungrouped-imports,disable=wrong-import-order

# ensure_installed("numpy scipy")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import numpy as np
from scipy.stats import norm, t
from scipy.stats._continuous_distns import norm_gen
from functools import lru_cache


def get_dist_percentage_span_for_sd(sd_sigma: float, dist: norm_gen = norm, **dist_kwargs) -> float:
    """Compute percentage of values lying within sigma std deviations from the mean (of a normal distribution).

    >>>np.isclose(get_dist_percentage_span_for_sd(3),0.9973002039367398)
    True

    >>>np.isclose(get_dist_percentage_span_for_sd(3, dist=t, df=1e20), 0.9973002039367398)
    True
    """
    return 1 - 2 * dist.cdf(-sd_sigma, **dist_kwargs)


def get_sd_for_dist_percentage(dist_percentage: float, dist: norm_gen = norm, **dist_kwargs) -> float:
    """Compute sigma std deviations from the mean where desired percentage of (normally distributed) values lies.

    >>>np.isclose(get_sd_for_dist_percentage(0.9973002039367398), 3.0)
    True
    """
    return -dist.ppf(-(dist_percentage - 1) / 2, **dist_kwargs)


@lru_cache
def get_tukey_fences_multiplier_for_quantile(
    quantile: float, sd_sigma: float = 2.7, nonoutlying_dist_percentage: float = None, dist: norm_gen = norm, **dist_kwargs
) -> float:
    """Compute Tukey fences [https://en.wikipedia.org/wiki/John_Tukey] multiplier for a desired quantile or nonoutlying dist coverage percent.

    For some nonnegative constant k John Tukey proposed this test, where k=1.5 indicates an "outlier", and k=3 indicates data that is "far out".
    Reasoning: https://math.stackexchange.com/questions/966331/why-john-tukey-set-1-5-iqr-to-detect-outliers-instead-of-1-or-2/

    >>>np.isclose(get_tukey_fences_multiplier_for_quantile(quantile=0.25, sd_sigma=2.7),1.5015129949825627)
    True

    >>>np.isclose(get_tukey_fences_multiplier_for_quantile(quantile=0.25, sd_sigma=None, nonoutlying_dist_percentage=0.9930660523939187),1.5015129949825627)
    True

    >>>np.isclose(get_tukey_fences_multiplier_for_quantile(quantile=0.1, sd_sigma=2.7),0.5534105971977119)
    True
    """
    assert quantile > 0 and quantile < 1.0
    if quantile > 0.5:
        quantile = 1 - quantile

    if sd_sigma is None:
        assert nonoutlying_dist_percentage > 0 and nonoutlying_dist_percentage < 1.0
        sd_sigma = get_sd_for_dist_percentage(nonoutlying_dist_percentage, dist=dist, dist_kwargs=dist_kwargs)

    ppf = np.abs(dist.ppf(quantile, **dist_kwargs))
    return (sd_sigma - ppf) / (2 * ppf)


def get_expected_unique_random_numbers_qty(span_size: int, sample_size: int) -> float:
    """Get expected number of unique elements drawn uniformly with replacement.

    https://stats.stackexchange.com/questions/296005/the-expected-number-of-unique-elements-drawn-with-replacement

    >>>get_expected_unique_random_numbers_qty(span_size=2000, sample_size=10)
    10.0
    >>>
    get_expected_unique_random_numbers_qty(span_size=200, sample_size=100)
    79.0

    how to check numerically?

    a, b = 0, 0
    for _ in range(1000):
        values = np.random.randint(0, 200, size=100)
        a += len(values)
        b += len(np.unique(values))
    print(b / a)
    0.78861
    """
    return np.ceil(span_size * (1 - np.exp(-sample_size / span_size)))
