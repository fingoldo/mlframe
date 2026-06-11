"""Regression sensors for the deferred (FUTURE) audit finding I7 on
``src/mlframe/training/core/_phase_composite_post_xt_ensemble/_phase_composite_post_xt_mtr_oof.py``.

I7 (cheap-partial landed 2026-06-11): ``compute_mtr_oof_nnls_weights`` returns ``None`` -- forfeiting the
benched ~9% honest-OOF NNLS win (``bench_mtr_nnls_oof.py``) and silently degrading the MTR cross-target
ensemble to equal-mean -- on (a) any single component fold-refit failure, (b) a non-finite OOF cell, or
(c) any catch-all exception. Pre-fix all three forfeitures were logged at DEBUG (and the non-finite path
logged NOTHING at all), so an operator never saw that the honest weighting had been silently abandoned.

This change bumps every *failure* exit to WARNING (and adds the missing log on the non-finite path). It is a
pure-observability change: the numeric return contract is unchanged (still ``None`` -> equal-mean), so it is
bit-identical on the success path and on the "not applicable" (too-few-components / too-few-rows) exit, which
stays quiet because equal-mean is the correct, non-degraded answer there.

These tests pin the WARNING visibility (they FAIL on the pre-fix DEBUG/silent logic) AND pin that the
"not applicable" and success paths stay quiet, so a future "log everything loudly" cannot regress the small-
data signal-to-noise.

The full per-component-exclusion fix (drop the failing component, NNLS-weight the rest) remains FUTURE: it
changes the return contract (the caller injects weights shaped to ALL components) and must be re-validated
against ``bench_mtr_nnls_oof.py`` before shipping.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.training.core._phase_composite_post_xt_ensemble._phase_composite_post_xt_mtr_oof import (
    compute_mtr_oof_nnls_weights,
)

_MTR_LOGGER = "mlframe.training.core._phase_composite_post"


class _FixedMTRComponent:
    """sklearn-cloneable component emitting a fixed per-row (K,) prediction.

    ``get_params``/``set_params`` make ``sklearn.base.clone`` happy without
    pulling in a real estimator (keeps the test fast + dependency-light).
    """

    def __init__(self, preds=None):
        self.preds = preds

    def get_params(self, deep=True):
        return {"preds": self.preds}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        self._k = int(np.asarray(y).reshape(len(y), -1).shape[1])
        return self

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else self.preds.shape[0]
        base = np.asarray(self.preds, dtype=np.float64)
        return np.tile(base, (n, 1))


class _RaisingComponent(_FixedMTRComponent):
    """Raises inside ``fit`` so the OOF fold-refit hits the failure exit."""

    def fit(self, X, y):
        raise RuntimeError("synthetic fold-refit failure")


class _NonFiniteComponent(_FixedMTRComponent):
    """Fits fine but predicts a NaN column so the OOF stack is non-finite."""

    def predict(self, X):
        out = super().predict(X)
        out[:, 0] = np.nan
        return out


def _make_problem(n=80, k=2):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 3))
    y = rng.normal(size=(n, k))
    return X, y


# ---------------------------------------------------------------------------
# Failure exits MUST be visible at WARNING (the I7 cheap-partial contract).
# These FAIL on the pre-fix DEBUG/silent logic.
# ---------------------------------------------------------------------------

def test_fold_refit_failure_warns_and_returns_none(caplog):
    """A single component whose fold-refit raises -> ``None`` AND a WARNING
    naming the forfeited honest-OOF win. Pre-fix this was a DEBUG line."""
    X, y = _make_problem()
    good = _FixedMTRComponent(np.array([0.0, 0.0]))
    bad = _RaisingComponent(np.array([0.0, 0.0]))

    with caplog.at_level(logging.WARNING, logger=_MTR_LOGGER):
        w = compute_mtr_oof_nnls_weights([good, bad], X, y, kfold=3, random_state=0)

    assert w is None
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("fold refit failed" in r.getMessage() for r in warnings), (
        "fold-refit failure must surface at WARNING, not DEBUG"
    )
    assert any("forfeiting" in r.getMessage() for r in warnings), (
        "the WARNING must state the benched honest-OOF win is being forfeited"
    )


def test_nonfinite_oof_warns_and_returns_none(caplog):
    """A component that fits but predicts NaN -> non-finite OOF -> ``None``
    AND a WARNING. Pre-fix this exit logged NOTHING at all."""
    X, y = _make_problem()
    good = _FixedMTRComponent(np.array([0.0, 0.0]))
    nan_comp = _NonFiniteComponent(np.array([0.0, 0.0]))

    with caplog.at_level(logging.WARNING, logger=_MTR_LOGGER):
        w = compute_mtr_oof_nnls_weights([good, nan_comp], X, y, kfold=3, random_state=0)

    assert w is None
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("non-finite" in r.getMessage() for r in warnings), (
        "non-finite OOF stack must surface at WARNING (pre-fix it was silent)"
    )
    assert any("forfeiting" in r.getMessage() for r in warnings)


# ---------------------------------------------------------------------------
# The "not applicable" + success exits MUST stay quiet (no spurious WARNING).
# These pin that the fix did not over-log on the expected small-data path.
# ---------------------------------------------------------------------------

def test_too_few_components_is_silent(caplog):
    """One component -> not an ensemble -> ``None`` with NO warning: equal-mean
    is the correct answer here, not a degraded forfeiture."""
    X, y = _make_problem()
    only = _FixedMTRComponent(np.array([0.0, 0.0]))
    with caplog.at_level(logging.WARNING, logger=_MTR_LOGGER):
        w = compute_mtr_oof_nnls_weights([only], X, y, kfold=3, random_state=0)
    assert w is None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], (
        "the too-few-components path must stay quiet"
    )


def test_too_few_rows_is_silent(caplog):
    """Below the min-row floor -> ``None`` with NO warning."""
    X, y = _make_problem(n=10, k=2)  # n < max(50, kfold*2)
    c1 = _FixedMTRComponent(np.array([0.0, 0.0]))
    c2 = _FixedMTRComponent(np.array([1.0, 1.0]))
    with caplog.at_level(logging.WARNING, logger=_MTR_LOGGER):
        w = compute_mtr_oof_nnls_weights([c1, c2], X, y, kfold=3, random_state=0)
    assert w is None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_success_path_is_silent_and_returns_weights(caplog):
    """Honest-OOF succeeds -> ``(n_comp, K)`` non-negative weights, NO warning.
    Pins that the observability change did not perturb the success contract."""
    from sklearn.linear_model import LinearRegression
    from sklearn.multioutput import MultiOutputRegressor

    n, k, p = 80, 2, 4
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=(p, k))
    y = X @ coef + 0.01 * rng.normal(size=(n, k))
    comps = [MultiOutputRegressor(LinearRegression()) for _ in range(3)]

    with caplog.at_level(logging.WARNING, logger=_MTR_LOGGER):
        w = compute_mtr_oof_nnls_weights(comps, X, y, kfold=4, random_state=0)

    assert w is not None
    assert w.shape == (3, k)
    assert (w >= 0).all(), "NNLS weights must be non-negative"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], (
        "the success path must not emit any forfeiture WARNING"
    )
