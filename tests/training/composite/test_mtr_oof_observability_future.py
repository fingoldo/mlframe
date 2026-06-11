"""Regression + biz_value sensors for audit finding I7 on
``src/mlframe/training/core/_phase_composite_post_xt_ensemble/_phase_composite_post_xt_mtr_oof.py``.

I7 history
----------
* Cheap-partial landed 2026-06-11: ``compute_mtr_oof_nnls_weights`` returned ``None`` -- forfeiting the benched
  ~9% honest-OOF NNLS win (``bench_mtr_nnls_oof.py``) and silently degrading the MTR cross-target ensemble to
  equal-mean -- on (a) any single component fold-refit failure, (b) a non-finite OOF cell, or (c) any catch-all
  exception. All three were bumped to WARNING so an operator could at least SEE the forfeiture.

* Full per-component-exclusion fix landed 2026-06-11 (this contract): a SINGLE bad component no longer forfeits
  the WHOLE weighting. The bad component is EXCLUDED (its row in the returned ``(n_components, n_targets)`` weight
  matrix is left all-zero) and the NNLS solve runs over the SURVIVING components only, so the ensemble keeps the
  honest weighting whenever >=2 components survive. A zero weight-row contributes nothing to the caller's
  ``np.einsum("cnk,ck->nk", stack, weights)`` apply, so the ALL-components return shape (the shape the caller at
  ``_post_xt_ensemble_mtr.py`` injects) is preserved. ``None`` is now returned only when fewer than 2 components
  survive (no ensemble left to weight) or on the not-applicable small-data path.

These tests pin:
  1. A single bad component (raise OR non-finite) among >=2 good ones is EXCLUDED -> a full-width weight matrix
     with a zero row for the bad component + a WARNING naming the exclusion (NOT ``None``). They FAIL on the
     pre-fix all-or-nothing ``None`` logic.
  2. Fewer than 2 survivors -> ``None`` + a WARNING (no ensemble to weight).
  3. The not-applicable (too-few-components / too-few-rows) and all-clean success paths stay quiet.
  4. biz_value: the salvaged (one-component-excluded) honest weighting beats equal-mean on a synthetic where the
     honest NNLS surface is the right answer -- i.e. the exclusion does not throw away the win.
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


class _LinearMTRComponent:
    """sklearn-cloneable least-squares component for a non-degenerate NNLS
    surface (so the salvaged honest weighting is actually informative)."""

    def __init__(self, cols=None, jitter=0.0):
        self.cols = cols
        self.jitter = jitter
        self._coef = None

    def get_params(self, deep=True):
        return {"cols": self.cols, "jitter": self.jitter}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        cols = self.cols if self.cols is not None else list(range(X.shape[1]))
        Xc = X[:, cols]
        # Ridge-ish least squares (lstsq is enough for the test scales).
        self._coef, *_ = np.linalg.lstsq(Xc, y, rcond=None)
        self._cols = cols
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = X[:, self._cols] @ self._coef
        if self.jitter:
            out = out + self.jitter
        return out


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


def _make_linear_problem(n=120, k=2, p=4, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=(p, k))
    y = X @ coef + 0.05 * rng.normal(size=(n, k))
    return X, y


# ---------------------------------------------------------------------------
# Per-component exclusion: a SINGLE bad component is dropped, the rest weighted.
# These FAIL on the pre-fix all-or-nothing ``None`` logic.
# ---------------------------------------------------------------------------

def test_fold_refit_failure_excludes_component_not_whole_weighting(caplog):
    """One component whose fold-refit raises is EXCLUDED (zero weight-row); the
    surviving components keep the honest NNLS weighting. Pre-fix returned ``None``."""
    X, y = _make_linear_problem(k=2)
    good_a = _LinearMTRComponent(cols=[0, 1])
    good_b = _LinearMTRComponent(cols=[2, 3])
    bad = _RaisingComponent(np.array([0.0, 0.0]))

    with caplog.at_level(logging.WARNING, logger=_MTR_LOGGER):
        w = compute_mtr_oof_nnls_weights([good_a, good_b, bad], X, y, kfold=4, random_state=0)

    assert w is not None, "a single raising component must NOT forfeit the whole weighting"
    assert w.shape == (3, 2)
    # The bad component (index 2) carries a zero weight-row on every target.
    np.testing.assert_array_equal(w[2, :], np.zeros(2))
    # At least one surviving component carries weight (honest NNLS surface kept).
    assert w[:2, :].sum() > 0
    assert (w >= 0).all()
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("excluding component 2" in r.getMessage() for r in warnings), (
        "the excluded component must be named at WARNING"
    )
    assert all("forfeiting the benched" not in r.getMessage() for r in warnings), (
        "with survivors remaining the win is SALVAGED, not forfeited"
    )


def test_nonfinite_oof_excludes_component_not_whole_weighting(caplog):
    """A component that fits but predicts NaN -> non-finite OOF -> that component is
    EXCLUDED, the rest keep the honest weighting. Pre-fix returned ``None`` (and was
    originally silent)."""
    X, y = _make_linear_problem(k=2)
    good_a = _LinearMTRComponent(cols=[0, 1])
    good_b = _LinearMTRComponent(cols=[2, 3])
    nan_comp = _NonFiniteComponent(np.array([0.0, 0.0]))

    with caplog.at_level(logging.WARNING, logger=_MTR_LOGGER):
        w = compute_mtr_oof_nnls_weights([good_a, good_b, nan_comp], X, y, kfold=4, random_state=0)

    assert w is not None
    assert w.shape == (3, 2)
    np.testing.assert_array_equal(w[2, :], np.zeros(2))
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("non-finite" in r.getMessage() for r in warnings), (
        "the non-finite OOF exclusion must surface at WARNING"
    )
    assert any("excluding component 2" in r.getMessage() for r in warnings)


def test_too_few_survivors_returns_none_and_warns(caplog):
    """Only ONE good component left after exclusion -> no ensemble to weight ->
    ``None`` + a forfeiture WARNING (equal-mean is the correct answer with <2
    survivors)."""
    X, y = _make_linear_problem(k=2)
    good = _LinearMTRComponent(cols=[0, 1])
    bad1 = _RaisingComponent(np.array([0.0, 0.0]))
    bad2 = _NonFiniteComponent(np.array([0.0, 0.0]))

    with caplog.at_level(logging.WARNING, logger=_MTR_LOGGER):
        w = compute_mtr_oof_nnls_weights([good, bad1, bad2], X, y, kfold=4, random_state=0)

    assert w is None
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("only 1 usable component" in r.getMessage() for r in warnings), (
        "the <2-survivor forfeiture must be logged"
    )
    assert any("forfeiting" in r.getMessage() for r in warnings)


# ---------------------------------------------------------------------------
# The "not applicable" + all-clean success exits MUST stay quiet.
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
    """All components clean -> ``(n_comp, K)`` non-negative weights, NO warning.
    Pins that exclusion logic did not perturb the all-clean success contract."""
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
    # No component excluded -> every row may carry weight (none forced to 0).
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], (
        "the all-clean success path must not emit any exclusion/forfeiture WARNING"
    )


def test_all_clean_is_bit_identical_to_unfiltered_nnls():
    """When NO component fails, the per-component-exclusion path reduces EXACTLY to
    the plain full-matrix NNLS solve (the survivor set is all components). Pins the
    bench-equivalence claim: exclusion cannot regress the all-clean numerics."""
    from scipy.optimize import nnls
    from sklearn.base import clone
    from sklearn.model_selection import KFold

    X, y = _make_linear_problem(n=120, k=2, p=4, seed=3)
    comps = [
        _LinearMTRComponent(cols=[0, 1]),
        _LinearMTRComponent(cols=[2, 3]),
        _LinearMTRComponent(cols=[0, 2]),
    ]
    kfold, rs = 4, 7

    w = compute_mtr_oof_nnls_weights(comps, X, y, kfold=kfold, random_state=rs)
    assert w is not None and w.shape == (3, 2)

    # Reference: build the full OOF stack the same way and NNLS over ALL components.
    n, k = y.shape
    oof = np.full((len(comps), n, k), np.nan)
    kf = KFold(n_splits=kfold, shuffle=True, random_state=rs)
    for tr, ho in kf.split(np.arange(n)):
        for ci, comp in enumerate(comps):
            cl = clone(comp)
            cl.fit(X[tr], y[tr])
            oof[ci, ho, :] = cl.predict(X[ho])
    ref = np.zeros((len(comps), k))
    for kk in range(k):
        wk, _ = nnls(oof[:, :, kk].T, y[:, kk], maxiter=200)
        ref[:, kk] = wk if float(wk.sum()) > 0 else 1.0 / len(comps)
    np.testing.assert_allclose(w, ref, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# biz_value: the salvaged (one-excluded) honest weighting beats equal-mean.
# ---------------------------------------------------------------------------

def test_biz_val_excluded_component_salvage_beats_equal_mean():
    """A synthetic where the honest NNLS surface clearly beats equal-mean, with one
    component forced to fail. The salvaged weighting (bad component excluded, rest
    NNLS-weighted) must still beat equal-mean by a comfortable margin -- i.e. the
    per-component exclusion preserves the win instead of dropping to equal-mean.

    Setup: target y is driven by features [0,1]. ``good_a`` regresses on the right
    features and predicts y well; ``good_b`` regresses on noise features and predicts
    poorly; ``bad`` raises. Equal-mean dilutes the strong learner with the weak one;
    NNLS over the survivors up-weights ``good_a``. Measured honest-OOF test-RMSE
    improvement over equal-mean was ~30%+; floor at 8% (5-15% below measured)."""
    rng = np.random.default_rng(11)
    n, p, k = 300, 6, 2
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=(2, k))
    y = X[:, :2] @ coef + 0.05 * rng.normal(size=(n, k))

    # Train / test split (the salvaged weights are applied to a fresh test stack).
    i = int(n * 0.7)
    Xtr, ytr = X[:i], y[:i]
    Xte, yte = X[i:], y[i:]

    good_a = _LinearMTRComponent(cols=[0, 1])   # strong: right features
    good_b = _LinearMTRComponent(cols=[4, 5])   # weak: noise features
    bad = _RaisingComponent(np.array([0.0, 0.0]))
    comps = [good_a, good_b, bad]

    w = compute_mtr_oof_nnls_weights(comps, Xtr, ytr, kfold=5, random_state=0)
    assert w is not None and w.shape == (3, k)
    np.testing.assert_array_equal(w[2, :], np.zeros(k))  # bad excluded

    # Fit the survivors on full train, build a test-prediction stack, apply weights.
    fitted = []
    for ci, comp in enumerate(comps):
        if ci == 2:
            fitted.append(None)  # excluded; placeholder so indices align
            continue
        from sklearn.base import clone
        cl = clone(comp)
        cl.fit(Xtr, ytr)
        fitted.append(cl)
    # Stack only the non-excluded preds; excluded row uses zeros (weight is 0 anyway).
    stack = np.stack([
        (np.zeros((len(Xte), k)) if fitted[ci] is None else fitted[ci].predict(Xte))
        for ci in range(3)
    ], axis=0)

    pred_nnls = np.einsum("cnk,ck->nk", stack, w)
    rmse_nnls = float(np.sqrt(np.mean((pred_nnls - yte) ** 2)))

    # Equal-mean baseline over the SAME survivors (the degraded fallback the old
    # code would have produced on a single failure).
    surv_stack = stack[[0, 1]]
    pred_eq = surv_stack.mean(axis=0)
    rmse_eq = float(np.sqrt(np.mean((pred_eq - yte) ** 2)))

    improvement = (rmse_eq - rmse_nnls) / rmse_eq
    assert improvement >= 0.08, (
        f"salvaged honest-OOF NNLS should beat equal-mean by >=8%; got "
        f"{improvement:.1%} (nnls={rmse_nnls:.4f}, equal_mean={rmse_eq:.4f})"
    )
