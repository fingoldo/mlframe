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


# ---------------------------------------------------------------------------
# P2 (2026-06-08): per-call copy elision via writeable-flag detection.
# ---------------------------------------------------------------------------

def test_p2_copy_elision_does_not_corrupt_inplace_shuffle():
    """After the writeable-flag self-check latches ``need_copy=False`` for a non-flag-flipping estimator,
    permutation_importance still produces identical importances. sklearn's permutation loop reuses one
    ``X_permuted`` buffer and shuffles a column in place each repeat; reading that buffer directly (no
    per-call copy) must NOT corrupt the shuffle or the score."""
    rng = np.random.RandomState(2)
    X = rng.randn(400, 10).copy()  # writeable, contiguous (the X_estimator-mirror shape)
    y = (X[:, 0] - X[:, 3] > 0).astype(int)
    model = LogisticRegression(max_iter=400).fit(X, y)

    pi_legacy = _perm(model, X, y, _legacy_scorer)
    pi_fast = _perm(model, X, y, _make_fast_default_scorer(model))
    assert np.array_equal(pi_legacy, pi_fast)


def test_p2_keeps_copy_for_writeable_flag_flipping_estimator():
    """An estimator whose ``predict`` flips the input ndarray's writeable flag to False (CatBoost) must
    keep the defensive copy: otherwise sklearn's next in-place column shuffle on its reused buffer raises
    'assignment destination is read-only'. The run must complete AND match the legacy scorer exactly."""
    cb = pytest.importorskip("catboost")
    rng = np.random.RandomState(3)
    X = rng.randn(400, 8).copy()
    y = (X[:, 1] > 0).astype(int)
    model = cb.CatBoostClassifier(iterations=40, depth=3, verbose=0, allow_writing_files=False).fit(X, y)

    pi_legacy = _perm(model, X, y, _legacy_scorer)
    pi_fast = _perm(model, X, y, _make_fast_default_scorer(model))
    assert np.array_equal(pi_legacy, pi_fast)


# ---------------------------------------------------------------------------
# P3 (2026-06-08): assume_finite=True around permutation_importance, gated on a
# one-time finite check of the fold so the per-call _assert_all_finite rescans
# are skipped bit-identically.
# ---------------------------------------------------------------------------

def test_p3_assume_finite_path_is_bit_identical():
    """``get_feature_importances('permutation')`` runs under ``assume_finite=True`` when the fold is all
    finite. The resulting importances must equal a default-context permutation_importance with the SAME
    fast scorer EXACTLY (assume_finite only skips a validation that would have passed)."""
    from mlframe.feature_selection.wrappers._helpers_importance import get_feature_importances

    rng = np.random.RandomState(1)
    X = rng.randn(600, 20).copy()
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = LogisticRegression(max_iter=400).fit(X, y)

    ref = _perm(model, X, y, _make_fast_default_scorer(model), seed=11)  # default context
    fi = get_feature_importances(model, list(range(20)), "permutation", data=X, target=y,
                                 n_repeats=5, random_state=11)
    got = np.array([fi[i] for i in range(20)])
    assert np.array_equal(ref, got)


def test_p3_finite_gate_off_on_nan_feature():
    """A NaN in X must turn the finite gate OFF so sklearn's per-call validation stays in force (the
    permutation path must not silently run under assume_finite when the data isn't finite)."""
    from mlframe.feature_selection.wrappers._helpers_importance import _fold_is_all_finite

    X = np.random.RandomState(2).randn(300, 10).copy()
    assert _fold_is_all_finite(X) is True
    X[5, 2] = np.nan
    assert _fold_is_all_finite(X) is False
    # integer arrays are always finite; object arrays conservatively False.
    assert _fold_is_all_finite(np.arange(20)) is True
    assert _fold_is_all_finite(np.array(["a", "b"], dtype=object)) is False
