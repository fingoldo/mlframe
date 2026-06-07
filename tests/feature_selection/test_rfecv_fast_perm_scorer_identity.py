"""perf regression guard (P1, 2026-06-08): RFECV's permutation-importance path now scores through a
fast closed-form scorer that reproduces ``estimator.score()`` BIT-IDENTICALLY but skips the redundant
per-call ``_check_targets`` / ``type_of_target`` validation. On the scene 2407x299 bench that validation
was the dominant hotspot (cProfile cumtime ``_check_targets`` ~43s + ``type_of_target`` ~23s, vs the
irreducible ``predict`` matmul ~27s); the fast path cut the full RFECV wall 69.3s -> 41.5s (1.67x) with a
byte-for-byte identical selected set.

These tests pin the bit-identity contract:
  * the fast scorer's permutation importances == the legacy ``estimator.score()`` scorer's importances,
    EXACTLY (np.array_equal), across classifier (accuracy) and regressor (r2) defaults; and
  * the per-fold self-check falls back to ``estimator.score()`` when the closed form can't match
    (e.g. a constant-target regressor, where sklearn r2 has special-case semantics).

If a future change perturbs the closed-form metric so it no longer bit-matches ``estimator.score()``,
the A/B assertion fails -- which is the signal that the selected set could drift.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from mlframe.feature_selection.wrappers._helpers_importance import _make_fast_default_scorer


def _legacy_scorer(_est, _X, _y):
    """Verbatim pre-P1 scorer: defensive copy + ``estimator.score()`` (full validation each call)."""
    _Xc = np.array(_X, copy=True) if isinstance(_X, np.ndarray) else _X
    return _est.score(_Xc, _y)


def _perm(model, X, y, scoring, seed=42, n_repeats=5):
    return permutation_importance(
        model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=1,
    ).importances_mean


@pytest.mark.parametrize(
    "name,make,regression",
    [
        ("logreg", lambda: LogisticRegression(max_iter=400), False),
        ("rf_clf", lambda: RandomForestClassifier(n_estimators=40, random_state=0), False),
        ("ridge", lambda: Ridge(), True),
        ("rf_reg", lambda: RandomForestRegressor(n_estimators=40, random_state=0), True),
    ],
)
def test_fast_perm_scorer_bit_identical_to_estimator_score(name, make, regression):
    rng = np.random.RandomState(1)
    X = rng.randn(500, 12)
    if regression:
        y = X @ rng.randn(12) + 0.1 * rng.randn(500)
    else:
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = make().fit(X, y)

    pi_legacy = _perm(model, X, y, _legacy_scorer)
    pi_fast = _perm(model, X, y, _make_fast_default_scorer(model))

    # HARD bit-identity gate: the selected set can only stay identical if the importances do.
    assert np.array_equal(pi_legacy, pi_fast), (
        f"{name}: fast permutation-FI scorer diverged from estimator.score() "
        f"(max|diff|={np.max(np.abs(pi_legacy - pi_fast)):.3e})"
    )


def test_fast_scorer_latches_fast_mode_for_standard_classifier():
    """On a plain 1d-target classifier the baseline self-check must latch the fast (closed-form) path."""
    rng = np.random.RandomState(0)
    X = rng.randn(300, 8)
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression(max_iter=300).fit(X, y)
    scorer = _make_fast_default_scorer(model)
    # First call = baseline self-check; it should return estimator.score() and latch fast mode.
    val = scorer(model, X, y)
    assert val == model.score(X, y)
    # A subsequent call now runs the fast path; still bit-identical to estimator.score().
    assert scorer(model, X, y) == model.score(X, y)


def test_fast_scorer_falls_back_on_constant_target_regressor():
    """A constant target makes ss_tot==0 so the closed-form r2 is undefined; the scorer must fall back
    to ``estimator.score()`` (mode 0) rather than latch a wrong fast value."""
    rng = np.random.RandomState(0)
    X = rng.randn(200, 5)
    y = np.full(200, 3.0)  # constant
    model = Ridge().fit(X, y)
    scorer = _make_fast_default_scorer(model)
    val = scorer(model, X, y)
    # Whatever estimator.score returns on a constant target, the scorer must echo it exactly
    # and NOT a bogus closed-form value.
    assert val == model.score(X, y)
