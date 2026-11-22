"""Business value of the ``min_support`` categorical fold: a categorical with a real effect and a long tail of thin levels is found again.

The column has 6 real levels (an effect on the target) and, on 8% of the rows, one of 200 thin levels with no effect. Under the frequency cap alone (the column is under the cap, so nothing folds)
the thin cells inflate the column's cell count and its plug-in MI, and the screen drops the real signal; folding the levels under 5 rows into one bucket brings back the 6 real levels and the signal.
Measured on 12 seeds at this size: ``frequency`` 0 / 12, ``min_support`` 12 / 12; four seeds are asserted.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


def _frame(seed: int, n: int = 1500, tail: int = 200, effect: float = 0.4):
    """Real levels 0-5 with an effect, 8% of the rows in a thin tail level with none."""
    rng = np.random.default_rng(200 + seed)
    main = rng.integers(0, 6, n)
    in_tail = rng.random(n) < 0.08
    level = np.where(in_tail, 6 + rng.integers(0, tail, n), main)
    logit = np.where(in_tail, 0.0, np.array([-1, -0.5, -0.2, 0.2, 0.5, 1.0])[main] * effect)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"noise": rng.standard_normal(n), "cat_sig": pd.Series(level.astype(str)).astype("category")})
    return df, y


def _selected(strategy: str, seeds) -> int:
    """How many of the ``seeds`` keep ``cat_sig`` in the support."""
    hits = 0
    for s in seeds:
        df, y = _frame(s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = MRMR(verbose=0, random_seed=s, fe_max_steps=0, categorical_fold_strategy=strategy).fit(df, y)
        hits += "cat_sig" in list(m.get_feature_names_out())
    return hits


@pytest.mark.slow
def test_min_support_recovers_the_real_categorical_that_the_thin_tail_hides():
    """``min_support`` keeps the real column on every seed; the frequency cap alone keeps it on at most one of four."""
    seeds = range(4)
    folded = _selected("min_support", seeds)
    plain = _selected("frequency", seeds)
    assert folded == 4, (folded, plain)
    assert plain <= 1, (folded, plain)


def _rare_signal_frame(seed: int, n_big: int = 1380, thin: int = 300, fraud: int = 6):
    """Five big levels at a 3 % positive rate and 300 levels of three rows each, six of which (random ones) are positive on all three rows."""
    rng = np.random.default_rng(300 + seed)
    big = rng.integers(0, 5, n_big)
    level = np.concatenate([big, np.repeat(np.arange(5, 5 + thin), 3)])
    fraud_levels = set(rng.choice(thin, fraud, replace=False).tolist())
    y = np.concatenate([(rng.random(n_big) < 0.03).astype(int), np.concatenate([np.ones(3, int) if k in fraud_levels else np.zeros(3, int) for k in range(thin)])])
    order = rng.permutation(level.size)
    df = pd.DataFrame({"noise": rng.standard_normal(level.size), "cat": pd.Series(level[order].astype(str)).astype("category")})
    return df, y[order]


def _rare_selected(strategy: str, seeds) -> int:
    """How many of the ``seeds`` keep ``cat`` in the support."""
    hits = 0
    for s in seeds:
        df, y = _rare_signal_frame(s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = MRMR(verbose=0, random_seed=s, fe_max_steps=0, categorical_fold_strategy=strategy).fit(df, y)
        hits += "cat" in list(m.get_feature_names_out())
    return hits


@pytest.mark.slow
def test_target_stat_keeps_the_rare_informative_levels_that_the_other_folds_lose():
    """Six three-row levels that are all positive (against a 3 % base rate) among 300 thin levels: ``target_stat`` finds the column on every seed, the frequency cap on none, ``min_support`` on at most half.

    Measured on 8 seeds: frequency 0 / 8, min_support 3 / 8, target_stat 8 / 8; four seeds are asserted. ``min_support`` folds the informative levels together with the 294 uninformative ones; the frequency cap
    (305 levels, cap 127) keeps the first 126 by count, which ties at three rows, so the informative levels are a matter of order."""
    seeds = range(4)
    by_target = _rare_selected("target_stat", seeds)
    by_support = _rare_selected("min_support", seeds)
    by_frequency = _rare_selected("frequency", seeds)
    assert by_target == 4, (by_target, by_support, by_frequency)
    assert by_frequency <= 1 and by_support <= 2, (by_target, by_support, by_frequency)
