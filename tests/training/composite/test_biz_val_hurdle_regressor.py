"""Business-value tests for ``HurdleRegressor``.

The synthetic reproduces the production target that motivated the class: ~71% of rows exactly zero, the event and
its magnitude driven by DIFFERENT features, and a log-normal magnitude whose tail weight ``sigma`` is varied. At
``sigma=1.8`` a single model on ``log1p(y)`` -- the production ``logY`` composite -- collapses to a prediction spread
a few percent of the target's, the failure logged as ``pred_std=6.44 (1.8% of target_std=349)``.

Each model is fit ONCE per ``sigma`` in a module-scoped fixture; the tests only read the cached predictions, which
keeps every test well inside the 5 s budget. Everything is deterministic: ``HistGradientBoosting*`` below 10k rows
does not early-stop, and all seeds are fixed. Thresholds sit 12-15% below the value measured on seed 0.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from mlframe.training.composite import HurdleRegressor

N_TRAIN = 4000
_KW = dict(max_iter=100, random_state=0)


def _zero_inflated(n: int, sigma: float, seed: int = 0, zero_frac: float = 0.71):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    logit = 1.6 * X[:, 0] - 1.2 * X[:, 1] + np.log((1 - zero_frac) / zero_frac)
    happens = rng.random(n) < 1.0 / (1.0 + np.exp(-logit))
    magnitude = np.exp(3.5 + 0.9 * X[:, 2] - 0.6 * X[:, 3] + rng.normal(0.0, sigma, n))
    return X, np.where(happens, magnitude, 0.0)


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def _fit_all(sigma: float) -> dict:
    X, y = _zero_inflated(2 * N_TRAIN, sigma)
    Xtr, ytr, Xte, yte = X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]
    hurdle = HurdleRegressor(classifier=HistGradientBoostingClassifier(**_KW), regressor=HistGradientBoostingRegressor(**_KW)).fit(Xtr, ytr)
    return {
        "y": yte,
        "hurdle": hurdle.predict(Xte),
        "log1p": np.expm1(HistGradientBoostingRegressor(**_KW).fit(Xtr, np.log1p(ytr)).predict(Xte)),
        "raw": HistGradientBoostingRegressor(**_KW).fit(Xtr, ytr).predict(Xte),
    }


@pytest.fixture(scope="module")
def moderate_tail() -> dict:
    return _fit_all(0.6)


@pytest.fixture(scope="module")
def heavy_tail() -> dict:
    return _fit_all(1.8)


# ------------------------------------------------------------------------------------------------------------
# Moderate tail: the hurdle beats both single-model alternatives outright.
# ------------------------------------------------------------------------------------------------------------


def test_biz_val_hurdle_regressor_default_moderate_tail_beats_log1p_single_model(moderate_tail):
    """Measured seed-0 R2: hurdle 0.3661 vs log1p 0.2054, a gap of 0.1607. Floor 0.14."""
    d = moderate_tail
    gap = _r2(d["y"], d["hurdle"]) - _r2(d["y"], d["log1p"])
    assert gap >= 0.14, gap


def test_biz_val_hurdle_regressor_default_moderate_tail_beats_raw_mse_single_model(moderate_tail):
    """Measured seed-0 R2: hurdle 0.3661 vs raw-MSE 0.2765, a gap of 0.0896. Floor 0.078.

    The honest size of the win over a single raw-MSE regressor: real but modest, and it shrinks as n grows, since a
    flexible booster eventually learns E[y|x] directly.
    """
    d = moderate_tail
    gap = _r2(d["y"], d["hurdle"]) - _r2(d["y"], d["raw"])
    assert gap >= 0.078, gap


# ------------------------------------------------------------------------------------------------------------
# Heavy tail (the production shape): the hurdle does not collapse where the log1p composite did.
# ------------------------------------------------------------------------------------------------------------


def test_biz_val_hurdle_regressor_default_heavy_tail_keeps_the_spread_log1p_loses(heavy_tail):
    """The production failure was a collapsed spread. Measured seed-0: hurdle 0.1298 of target std vs log1p
    0.0439, a ratio of 2.959. Floor 2.6."""
    d = heavy_tail
    ratio = float(np.std(d["hurdle"]) / np.std(d["log1p"]))
    assert ratio >= 2.6, ratio


def test_biz_val_hurdle_regressor_default_heavy_tail_beats_log1p_single_model(heavy_tail):
    """Measured seed-0 R2: hurdle 0.0515 vs log1p 0.0064, a gap of 0.0451. Floor 0.039."""
    d = heavy_tail
    gap = _r2(d["y"], d["hurdle"]) - _r2(d["y"], d["log1p"])
    assert gap >= 0.039, gap


def test_biz_val_hurdle_regressor_default_heavy_tail_beats_predicting_the_mean(heavy_tail):
    """On the production-shaped tail the hurdle still carries signal. Measured seed-0 R2 0.0515. Floor 0.044."""
    d = heavy_tail
    assert _r2(d["y"], d["hurdle"]) >= 0.044
