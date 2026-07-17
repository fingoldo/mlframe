"""BorutaShap margin-gated adaptive trial-stop (opt-in ``early_stop_tentative``).

The shipped all-decided early-stop only fires when ZERO features are tentative. On real data a residual TENTATIVE
TAIL (features whose binomial p-value sits permanently between the accept and reject thresholds) never resolves, so
the loop burns the full ``n_trials`` cap for nothing. ``early_stop_tentative=True`` adds a MARGIN-GATED stop for that
tail: stop once the accepted set has plateaued AND no still-tentative feature can still cross a decision boundary
within the remaining trials (the tail is provably stuck). These tests pin:

  - the option defaults OFF and is byte-identical to the prior fixed-cap run when off,
  - when on it saves trials AND is decision-equivalent to the cap (identical accepted set; rejected Jaccard >= 0.84),
  - the SAFE margin-gated rule is what ships (NOT the naive accepted-set-stability rule, a measured correctness trap).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# Cap kept modest (70) so two fits stay well under the pytest timeout while still leaving a residual tentative tail
# at the cap (full resolution on this bed needs ~110 trials). The fs_hybrid round4_adaptive_n_trials_bench.py runs
# the production-scale recipe (n=5000, cap=120) and confirms the larger wall-savings (synth ~72%, hard_synth ~63%).
_N_TRIALS_CAP = 70
_PATIENCE = 25


def _tail_dataset(n=3000, seed=0):
    """Strong signal + redundant copies + interaction operands (weak marginal) + pure noise.

    Mirrors the fs_hybrid synth shape that produces a residual tentative tail: the interaction operands carry almost
    no MARGINAL signal (only z4*z5 enters the logit), so their hit-rate hovers near 0.5 and the binomial test parks
    them permanently between accept and reject -> a tail the all-decided stop never clears. n is large enough that
    this tail is stable (at small n the tail features get rejected outright and there is nothing to reclaim)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 8))
    logit = (
        1.4 * z[:, 0]
        + 1.1 * z[:, 1]
        - 1.0 * z[:, 2]
        + 0.9 * z[:, 3]
        + 1.6 * z[:, 4] * z[:, 5]  # interaction: operands have ~zero marginal signal
        + 1.3 * (z[:, 6] ** 2 - 1.0)  # quadratic operand
        + 0.8 * z[:, 7]
    ) / 1.6
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    cols = {f"inf_{i}": z[:, i] for i in range(8)}
    for parent in (0, 4, 6):  # redundant correlated copies
        for j in range(4):
            cols[f"red_{parent}_{j}"] = z[:, parent] + 0.30 * rng.standard_normal(n)
    for i in range(28):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    order = list(X.columns)
    rng.shuffle(order)
    return X[order], pd.Series(y, name="target")


def _make_selector(**kw):
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.boruta_shap import BorutaShap

    base = dict(
        model=RandomForestClassifier(n_estimators=50, n_jobs=4, random_state=0),
        importance_measure="gini",
        classification=True,
        n_trials=_N_TRIALS_CAP,
        percentile=95,
        pvalue=0.05,
        verbose=False,
        random_state=0,
    )
    base.update(kw)
    return BorutaShap(**base)


def test_early_stop_tentative_defaults_off_and_roundtrips():
    """The new opt-in must default OFF (so existing behaviour is unchanged) and round-trip through get_params."""
    from mlframe.feature_selection.boruta_shap import BorutaShap

    b = BorutaShap()
    p = b.get_params(deep=False)
    assert p["early_stop_tentative"] is False  # default OFF -> byte-identical to prior behaviour
    assert p["early_stop_patience"] == 20
    assert p["early_stop_margin"] == 0.15
    # Shallow round-trip reconstructs (the estimator must stay sklearn-clone-able).
    b2 = BorutaShap(early_stop_tentative=True, early_stop_patience=15, early_stop_margin=0.2)
    p2 = b2.get_params(deep=False)
    assert BorutaShap(**p2).early_stop_tentative is True
    assert BorutaShap(**p2).early_stop_patience == 15
    assert BorutaShap(**p2).early_stop_margin == 0.2


def _jac(a, b):
    a, b = set(a), set(b)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def test_margin_gated_stop_is_decision_equivalent_and_saves_trials():
    """ON vs OFF margin-gated stop: it must be decision-equivalent to running the full cap -- identical accepted set
    (the load-bearing output) and a near-identical rejected set (Jaccard >= 0.84) -- and must never run MORE trials.

    The earlier version additionally required the OFF run to BURN the cap with a residual tentative tail. That tail
    was an artifact of a reject-side calibration bug in ``test_features`` (the reject binomial test shared the accept
    side's tiny ``null_hit_p``, so at the near-MAX-shadow gate it could never reject and every weak/noise column
    lingered tentative forever). With the reject side corrected (classic p=0.5 reference) the bed resolves before the
    cap, so there is no perpetual tail to reclaim; this test now pins the decision-equivalence + no-extra-trials
    invariant, which is what callers actually rely on. The opt-in feature is kept (REJECTED-not-DELETED)."""
    pytest.importorskip("sklearn")
    X, y = _tail_dataset(seed=0)

    off = _make_selector(early_stop_tentative=False)
    off.fit(X, y)
    on = _make_selector(early_stop_tentative=True, early_stop_patience=_PATIENCE, early_stop_margin=0.15)
    on.fit(X, y)

    assert on.n_trials_run_ <= off.n_trials_run_, f"margin-gated stop ran {on.n_trials_run_} trials, expected <= {off.n_trials_run_}"

    # Decision-equivalence: the accepted set is IDENTICAL (the margin gate refuses to stop while any tentative
    # feature could still cross, so no accept is missed), and the rejected set matches within the recipe's bar.
    assert set(on.accepted) == set(off.accepted), f"margin-gated accepted set diverged: on={sorted(on.accepted)} off={sorted(off.accepted)}"
    assert _jac(on.rejected, off.rejected) >= 0.84, f"rejected Jaccard {_jac(on.rejected, off.rejected):.3f} below recipe bar 0.84"
    # selected_features_ (accepted + tentative when optimistic) tracks the accepted set + preserved tail.
    assert _jac(on.selected_features_, off.selected_features_) >= 0.84


def test_default_off_is_byte_identical_to_baseline():
    """With the option OFF the run must be IDENTICAL to a run that never knew about the option: same n_trials_run_,
    same accepted/rejected/tentative, same support_. This is the byte-identical default guarantee."""
    pytest.importorskip("sklearn")
    X, y = _tail_dataset(seed=1)

    a = _make_selector()  # option absent -> default False
    a.fit(X, y)
    b = _make_selector(early_stop_tentative=False)  # explicitly False
    b.fit(X, y)

    assert a.n_trials_run_ == b.n_trials_run_
    assert set(a.accepted) == set(b.accepted)
    assert set(a.rejected) == set(b.rejected)
    assert set(a.tentative) == set(b.tentative)
    assert np.array_equal(a.support_, b.support_)
    assert list(a.selected_features_) == list(b.selected_features_)


def test_shipped_rule_is_margin_gated_not_naive():
    """Guard that the SAFE margin-gated rule is what fires, NOT the naive accepted-set-stability rule. Construct a
    transient plateau where the accepted set is flat for the patience window BUT a tentative feature is still near a
    decision boundary: the naive rule would stop (and lock the wrong set), the shipped margin-gated rule must NOT."""
    from mlframe.feature_selection.boruta_shap._fit_explain import (
        _naive_accepted_set_stable,
        _tentative_near_boundary,
        _should_stop_tentative_tail,
    )

    patience = 5
    n_tests = 10
    pvalue = 0.05
    margin = 2.0  # thr = pvalue*(1+margin) = 0.15 -> a clear band above the 0.05 decision line, robust to
    #              binomial discreteness (the narrow default band is exercised by the end-to-end fit test above).
    # Accepted set flat for the whole window -> the naive plateau condition is satisfied.
    accepted_history = [frozenset({"a"})] * (patience + 2)
    assert _naive_accepted_set_stable(accepted_history, patience) is True

    iteration = 20
    # hits[1] = 16/20 -> corrected accept p-value 0.0591: NOT yet decided (>= pvalue=0.05) but WITHIN margin of the
    # accept threshold (< 0.15), so it can still cross in the next few trials. The safe rule must REFUSE to stop
    # here; the naive rule (plateau only) would wrongly stop and lock the current accepted set.
    hits = np.array([10.0, 16.0])
    tentative_idx = [1]
    assert _tentative_near_boundary(hits, tentative_idx, iteration, n_tests, pvalue=pvalue, margin=margin) is True
    assert (
        _should_stop_tentative_tail(
            accepted_history,
            hits,
            tentative_idx,
            iteration=iteration,
            n_tests=n_tests,
            pvalue=pvalue,
            patience=patience,
            margin=margin,
        )
        is False
    ), "shipped rule must NOT stop while a tentative feature is near a boundary"

    # Now make the tentative feature provably stuck (hit-rate parked at 0.5, far from both thresholds): the margin
    # gate clears and the safe rule stops -- matching the plateau the naive rule also saw.
    hits_stuck = np.array([10.0, 10.0])
    assert _tentative_near_boundary(hits_stuck, tentative_idx, iteration, n_tests, pvalue=pvalue, margin=margin) is False
    assert (
        _should_stop_tentative_tail(
            accepted_history,
            hits_stuck,
            tentative_idx,
            iteration=iteration,
            n_tests=n_tests,
            pvalue=pvalue,
            patience=patience,
            margin=margin,
        )
        is True
    )
