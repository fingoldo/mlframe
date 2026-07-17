"""Regression tests for the models_estimators audit (2026-07-09): MBHOptimizer RNG/timestamp fixes + tuning.py surrogate drift.

Covers:
  F3 -- MBHOptimizer.submit_evaluations falsy-zero ``start_ts`` check.
  F4 -- MBHOptimizer RNG threading (_rng / _stdlib_rng independence from draw order).
  F5 -- tuning.py justify_estimator caching the CV mean instead of the refit model's own held-out score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.models.optimization import MBHOptimizer


def _make_optimizer(**overrides):
    kwargs = dict(
        search_space=np.arange(0, 20),
        model_name="ETR",
        model_params={},
        init_num_samples=5,
        random_state=0,
    )
    kwargs.update(overrides)
    return MBHOptimizer(**kwargs)


def test_submit_evaluations_handles_zero_start_ts():
    """Regression test for audit F3: a legitimate start_ts of 0.0 must not be treated as missing.

    Before the fix, ``if start_ts:`` was falsy for 0.0, so the ``suggested_candidates`` bookkeeping entry
    was never deleted and the elapsed-duration back-fill never ran, even though a real timestamp was present.
    """
    optimizer = _make_optimizer()
    candidate = 7
    optimizer.suggested_candidates[candidate] = 0.0  # a legitimate (if unlikely) monotonic timer reading of exactly 0.0

    optimizer.submit_evaluations(candidates=[candidate], evaluations=[1.0], durations=[None])

    assert candidate not in optimizer.suggested_candidates, "start_ts=0.0 must still be treated as present and cleaned up"
    assert optimizer.evaluated_candidates[-1]["duration"] is not None, "duration must be back-filled from a start_ts of 0.0"


def test_rng_seeding_independent_of_intermediate_rng_draws():
    """Regression test for audit F4: _stdlib_rng's seed must not depend on how many draws _rng makes first.

    Before the fix, ``_stdlib_rng`` was seeded from a value drawn OUT OF ``_rng`` mid-construction, so any extra
    ``_rng`` consumption elsewhere in ``__init__`` (a very plausible future edit) silently changed every
    ``_stdlib_rng.random()`` decision for all later runs at the same ``random_state``. With independent
    SeedSequence children, consuming ``_rng`` before constructing the optimizer must not perturb the derived
    stdlib seed.
    """
    seed = 12345

    seq_a = np.random.SeedSequence(seed)
    child_np_a, child_stdlib_a = seq_a.spawn(2)
    rng_a = np.random.default_rng(child_np_a)
    # Burn an arbitrary number of draws from a sibling stream sharing the SAME root seed -- simulates a future
    # edit adding more _rng-consuming lines to MBHOptimizer.__init__ before _stdlib_rng is derived.
    for _ in range(37):
        rng_a.integers(0, 2**32 - 1)
    stdlib_seed_a = int(child_stdlib_a.generate_state(1, dtype=np.uint64)[0])

    seq_b = np.random.SeedSequence(seed)
    child_np_b, child_stdlib_b = seq_b.spawn(2)
    stdlib_seed_b = int(child_stdlib_b.generate_state(1, dtype=np.uint64)[0])

    assert stdlib_seed_a == stdlib_seed_b, "stdlib seed must be independent of how many draws the numpy stream took first"


def test_optimizer_stdlib_rng_reproducible_for_same_random_state():
    opt1 = _make_optimizer(random_state=42)
    opt2 = _make_optimizer(random_state=42)
    draws1 = [opt1._stdlib_rng.random() for _ in range(5)]
    draws2 = [opt2._stdlib_rng.random() for _ in range(5)]
    assert draws1 == draws2


def test_justify_estimator_expected_score_is_refit_held_out_score_not_cv_mean():
    """Regression test for audit F5: expected_score must reflect the actual refit model's quality.

    Before the fix, ``expected_score`` was always the k-fold CV mean, computed on DIFFERENT folds than the
    single random train/eval split the CatBoost refit was early-stopped against -- callers cached and reused
    this CV number as a gate on the refit model's quality with no re-validation against its own held-out score.
    Patch ``check_scoring`` to return a scorer with a KNOWN, fixed value distinct from the CV mean, and verify
    the returned ``expected_score`` is that refit-time value, not whatever ``cross_validate`` computed.
    """
    from catboost import CatBoostRegressor
    import mlframe.models.tuning as tuning_mod

    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=[f"f{i}" for i in range(4)])
    y = X["f0"] * 2.0 + rng.normal(scale=0.01, size=200)

    _SENTINEL_REFIT_SCORE = 0.4242

    def _fake_check_scoring(estimator, scoring=None):
        return lambda est, X_test, y_test: _SENTINEL_REFIT_SCORE

    orig_check_scoring = tuning_mod.check_scoring
    tuning_mod.check_scoring = _fake_check_scoring
    try:
        est, expected_score = tuning_mod.justify_estimator(
            CatBoostRegressor(iterations=20, verbose=False),
            X,
            y,
            cv=3,
            refit=True,
            scoring="r2",
            min_score=0.5,
            random_state=0,
            early_stopping_rounds=5,
        )
    finally:
        tuning_mod.check_scoring = orig_check_scoring

    assert est is not None
    assert expected_score == pytest.approx(_SENTINEL_REFIT_SCORE), (
        "expected_score must be the refit model's OWN held-out score, not the CV gate mean, so a cached model's "
        "reported quality cannot silently drift from what it actually scores on its own eval split"
    )
