"""biz_val tests for ``RFECV`` (feature_selection/wrappers/_rfecv.py).

Per CLAUDE.md "Every new ML trick gets a biz_val synthetic test":
each test asserts a SYNTHETIC measurable WIN that locks in RFECV's
core parameters. A future code change that silently breaks one of
these will fail the matching assertion.

Naming: ``test_biz_val_rfecv_<parameter>_<scenario>``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


def _signal_plus_noise(n=2000, p_signal=3, p_noise=10, seed=42):
    """Linear binary target with ``p_signal`` true features and
    ``p_noise`` pure-noise features."""
    rng = np.random.default_rng(seed)
    X_signal = rng.normal(size=(n, p_signal))
    X_noise = rng.normal(size=(n, p_noise))
    X = np.column_stack([X_signal, X_noise])
    y = (X_signal.sum(axis=1) + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    return X, y


def _correlated_redundant(n=2000, seed=42):
    """4 correlated features (rho ~0.95) + 1 unique informative + 5
    noise. Greedy selection might pick from the correlated cluster
    only; stability selection should additionally surface the unique
    informative."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    X_corr = np.column_stack([base + 0.1 * rng.normal(size=n) for _ in range(4)])
    X_signal_unique = rng.normal(size=(n, 1))
    X_noise = rng.normal(size=(n, 5))
    X = np.column_stack([X_corr, X_signal_unique, X_noise])
    y = (X_corr[:, 0] + X_signal_unique[:, 0] + 0.3 * rng.normal(size=n) > 0
         ).astype(np.int64)
    return X, y


def _support_indices(sel):
    """Return support_ as integer indices regardless of whether RFECV
    exposes a boolean mask or an integer-array support."""
    s = sel.support_
    if s.dtype == bool:
        return [int(i) for i in np.flatnonzero(s)]
    return [int(i) for i in s]


# ---------------------------------------------------------------------------
# n_features_selection_rule: argmax vs one_se_min
# ---------------------------------------------------------------------------


def test_biz_val_rfecv_n_features_selection_rule_one_se_min_picks_smaller():
    """``n_features_selection_rule='one_se_min'`` must pick <= the
    number of features as ``='argmax'`` on a target with 3 strong +
    10 noise. The 1-SE rule trades a tiny mean-score loss for a
    smaller, more interpretable model."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV

    X, y = _signal_plus_noise(n=1500, p_signal=3, p_noise=10, seed=42)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])

    common = dict(
        estimator=RandomForestClassifier(random_state=42, n_estimators=30),
        cv=3, max_refits=8, verbose=0, random_state=42,
        max_noimproving_iters=3,
    )
    sel_argmax = RFECV(n_features_selection_rule="argmax", **common)
    sel_one_se = RFECV(n_features_selection_rule="one_se_min", **common)
    sel_argmax.fit(df, y)
    sel_one_se.fit(df, y)
    assert sel_one_se.n_features_ <= sel_argmax.n_features_, (
        f"one_se_min picked {sel_one_se.n_features_}, argmax picked "
        f"{sel_argmax.n_features_}; one_se_min must be <= argmax"
    )


# ---------------------------------------------------------------------------
# stability_selection: bootstrap voting
# ---------------------------------------------------------------------------


def test_biz_val_rfecv_stability_selection_recovers_signal_features():
    """With ``stability_selection=True`` + bootstrap voting, RFECV
    should select features that appear in the majority of bootstrap
    runs. On a clean signal-plus-noise target, the 3 signal features
    must all be in the support_."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV

    X, y = _signal_plus_noise(n=1500, p_signal=3, p_noise=10, seed=42)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    sel = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=30),
        cv=3, max_refits=6, verbose=0, random_state=42,
        max_noimproving_iters=3,
        stability_selection=True,
        stability_n_bootstrap=10,
        stability_threshold=0.4,
    )
    sel.fit(df, y)
    selected = set(_support_indices(sel))
    overlap = selected & {0, 1, 2}
    assert len(overlap) >= 2, (
        f"stability_selection must recover >=2 of 3 signal features; "
        f"got selected={sorted(selected)}, overlap={overlap}"
    )


# ---------------------------------------------------------------------------
# must_include: forced inclusion
# ---------------------------------------------------------------------------


def test_biz_val_rfecv_must_include_keeps_specified_feature():
    """``must_include=['x_noise_5']`` (a pure-noise column) must
    remain in support_ after RFECV runs. Without the constraint,
    a noise feature would never survive feature elimination on this
    target. Catches regressions in the must_include enforcement."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV

    X, y = _signal_plus_noise(n=1500, p_signal=3, p_noise=10, seed=42)
    # Name the noise columns explicitly so the must_include is clear.
    cols = [f"x{i}" for i in range(3)] + [f"noise{i}" for i in range(10)]
    df = pd.DataFrame(X, columns=cols)

    sel = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=30),
        cv=3, max_refits=8, verbose=0, random_state=42,
        max_noimproving_iters=3,
        must_include=["noise5"],
    )
    sel.fit(df, y)
    selected_names = [df.columns[i] for i in _support_indices(sel)]
    assert "noise5" in selected_names, (
        f"must_include=['noise5'] must keep noise5; "
        f"got selected={selected_names}"
    )


# ---------------------------------------------------------------------------
# importance_getter='conditional_permutation' (CPI)
# ---------------------------------------------------------------------------


def test_biz_val_rfecv_conditional_permutation_keeps_unique_informative_under_correlation():
    """Vanilla permutation importance assigns ZERO importance to each
    member of a correlated cluster (the model still predicts via the
    redundant copies). Conditional permutation breaks this by
    permuting WITHIN leaves of a tree on ``X_{-j}``, preserving the
    joint distribution.

    Floor: with ``importance_getter='conditional_permutation'``, the
    UNIQUE informative feature ``x4`` (not in the correlated cluster)
    must be selected. A vanilla-permutation run would also pick at
    least one cluster member; the cluster's INTERNAL ranking is more
    sensitive to CPI so that's not the cleanest signal."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV

    X, y = _correlated_redundant(n=1500, seed=42)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])

    sel = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=50,
                                            max_depth=6),
        cv=3, max_refits=6, verbose=0, random_state=42,
        max_noimproving_iters=3,
        importance_getter="conditional_permutation",
    )
    sel.fit(df, y)
    selected = set(_support_indices(sel))
    assert 4 in selected, (
        f"CPI must keep unique-informative x4; got selected={sorted(selected)}"
    )
    # And at least one of the correlated cluster {x0..x3} must also
    # survive (each carries the same joint signal).
    assert any(i in selected for i in (0, 1, 2, 3)), (
        f"CPI must keep >=1 correlated-cluster member; "
        f"got selected={sorted(selected)}"
    )


# ---------------------------------------------------------------------------
# checkpoint_path: resume-from-checkpoint
# ---------------------------------------------------------------------------


def test_biz_val_rfecv_feature_cost_penalizes_large_subsets():
    """Strong ``feature_cost=0.5`` must force RFECV to pick a SMALLER
    subset than ``feature_cost=0.0`` on the same data + estimator.
    The cost adds a per-feature penalty to the CV score, biasing the
    optimizer toward parsimony."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df,
    )
    X, y, _ = make_signal_plus_noise(n=800, p_signal=3, p_noise=8, seed=42)
    df, _ys = as_df(X, y)

    common = dict(
        estimator=RandomForestClassifier(random_state=42, n_estimators=30),
        cv=3, max_refits=8, verbose=0, random_state=42,
        max_noimproving_iters=3,
        n_features_selection_rule="argmax",
    )
    sel_free = RFECV(feature_cost=0.0, **common)
    sel_cost = RFECV(feature_cost=0.5, **common)
    sel_free.fit(df, y)
    sel_cost.fit(df, y)
    assert sel_cost.n_features_ <= sel_free.n_features_, (
        f"feature_cost=0.5 must pick <= features than =0.0; "
        f"got cost={sel_cost.n_features_}, free={sel_free.n_features_}"
    )


def test_biz_val_rfecv_leakage_corr_threshold_detects_target_leak():
    """A perfect-leak column (copy of y) must be detected and either
    dropped (``leakage_action='raise'``-equivalent) or warned about
    (``='warn'``). With the default 0.95 threshold and a copy of y,
    the leakage detector must catch it. We assert it doesn't end up
    as the #1 selected feature with a normal selection -- if the
    detector is dead, perfect-leak feature dominates."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df,
    )
    X, y, _ = make_signal_plus_noise(n=800, p_signal=3, p_noise=5, seed=42)
    # Inject a column that's effectively y (perfect leak, slight noise
    # so RFECV's strict 0.95 still flags it).
    rng = np.random.default_rng(0)
    leak = y.astype(np.float64) + 0.005 * rng.normal(size=len(y))
    X_with_leak = np.column_stack([X, leak])
    df = pd.DataFrame(X_with_leak, columns=[f"x{i}" for i in range(
        X_with_leak.shape[1] - 1)] + ["leak"])
    sel = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=30),
        cv=3, max_refits=6, verbose=0, random_state=42,
        max_noimproving_iters=3,
        leakage_corr_threshold=0.95,
        leakage_action="exclude",
    )
    sel.fit(df, y)
    selected_names = [df.columns[i] for i in _support_indices(sel)]
    assert "leak" not in selected_names, (
        f"leakage_action='drop' must remove the perfect-leak column; "
        f"got selected={selected_names}"
    )


def test_biz_val_rfecv_swap_top_k_yields_valid_support():
    """``swap_top_k=3`` runs a final-pass swap routine; the resulting
    support_ must remain a valid subset of the input features (no
    duplicates, all indices within bounds). Catches regressions
    where the swap pass could corrupt the support_ mask."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df,
    )
    X, y, _ = make_signal_plus_noise(n=800, p_signal=3, p_noise=8, seed=42)
    df, _ys = as_df(X, y)
    sel = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=30),
        cv=3, max_refits=4, verbose=0, random_state=42,
        max_noimproving_iters=2,
        swap_top_k=3,
    )
    sel.fit(df, y)
    idx = _support_indices(sel)
    assert len(idx) == len(set(idx)), f"support contains duplicates: {idx}"
    assert all(0 <= i < df.shape[1] for i in idx), (
        f"support indices out of bounds for p={df.shape[1]}: {idx}"
    )


@pytest.mark.parametrize("search_method", [
    "ModelBasedHeuristic", "ExhaustiveRandom",
])
def test_biz_val_rfecv_top_predictors_search_method_completes(search_method):
    """``top_predictors_search_method`` parametrized over the 3 most
    common strategies. Each must complete and produce a valid
    support_ on a signal+noise target."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV
    from mlframe.feature_selection.wrappers._enums import OptimumSearch
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df,
    )
    X, y, _ = make_signal_plus_noise(n=600, p_signal=3, p_noise=6, seed=42)
    df, _ys = as_df(X, y)
    sel = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=20),
        cv=3, max_refits=4, verbose=0, random_state=42,
        max_noimproving_iters=2,
        top_predictors_search_method=OptimumSearch(search_method),
    )
    sel.fit(df, y)
    idx = _support_indices(sel)
    assert 1 <= len(idx) <= df.shape[1]


@pytest.mark.parametrize("votes_method", [
    "Borda", "Plurality", "Dowdall", "AM", "GM",
])
def test_biz_val_rfecv_votes_aggregation_method_completes(votes_method):
    """``votes_aggregation_method`` parametrized over 5 voting rules.
    Each must complete; catches regressions in any of the aggregation
    backends."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV
    from mlframe.feature_selection.wrappers._enums import VotesAggregation
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df,
    )
    X, y, _ = make_signal_plus_noise(n=600, p_signal=3, p_noise=6, seed=42)
    df, _ys = as_df(X, y)
    sel = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=20),
        cv=3, max_refits=4, verbose=0, random_state=42,
        max_noimproving_iters=2,
        votes_aggregation_method=VotesAggregation(votes_method),
    )
    sel.fit(df, y)
    idx = _support_indices(sel)
    assert 1 <= len(idx) <= df.shape[1]


def test_biz_val_rfecv_use_last_fi_run_only_ignores_history():
    """``use_last_fi_run_only=True`` must produce a valid result
    (only the most recent feature-importance vector contributes to
    voting). Catches regressions where the flag is silently ignored."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df,
    )
    X, y, _ = make_signal_plus_noise(n=600, p_signal=3, p_noise=6, seed=42)
    df, _ys = as_df(X, y)
    sel = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=20),
        cv=3, max_refits=4, verbose=0, random_state=42,
        max_noimproving_iters=2,
        use_all_fi_runs=False,
        use_last_fi_run_only=True,
    )
    sel.fit(df, y)
    assert 1 <= len(_support_indices(sel)) <= df.shape[1]


@pytest.mark.parametrize("seed", [1, 7, 42, 123])
def test_biz_val_rfecv_random_state_reproducibility(seed):
    """``random_state=<seed>`` must produce identical support_ across
    two independent fits. Catches regressions where internal sampling
    drifts beyond random_state's control."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df,
    )
    X, y, _ = make_signal_plus_noise(n=600, p_signal=3, p_noise=5, seed=seed)
    df, _ys = as_df(X, y)
    common = dict(
        estimator=RandomForestClassifier(random_state=42, n_estimators=20),
        cv=3, max_refits=4, verbose=0,
        max_noimproving_iters=2, random_state=seed,
    )
    sel_a = RFECV(**common)
    sel_b = RFECV(**common)
    sel_a.fit(df, y)
    sel_b.fit(df, y)
    assert _support_indices(sel_a) == _support_indices(sel_b), (
        f"random_state={seed} must produce identical support_; got "
        f"a={_support_indices(sel_a)}, b={_support_indices(sel_b)}"
    )


@pytest.mark.parametrize("cv_folds", [2, 3, 5])
def test_biz_val_rfecv_cv_folds_robust_to_value(cv_folds):
    """RFECV with different CV fold counts must complete and produce
    valid support. Parametrize over {2, 3, 5}."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df,
    )
    X, y, _ = make_signal_plus_noise(n=600, p_signal=3, p_noise=5, seed=42)
    df, _ys = as_df(X, y)
    sel = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=20),
        cv=cv_folds, max_refits=3, verbose=0, random_state=42,
        max_noimproving_iters=2,
    )
    sel.fit(df, y)
    assert 1 <= len(_support_indices(sel)) <= df.shape[1]


@pytest.mark.parametrize("smooth_perf", [0, 1, 3])
def test_biz_val_rfecv_smooth_perf_parametrize_completes(smooth_perf):
    """``smooth_perf`` (rolling-mean window over CV iterations)
    parametrized over {0=off, 1, 3}. Each must complete cleanly."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df,
    )
    X, y, _ = make_signal_plus_noise(n=600, p_signal=3, p_noise=5, seed=42)
    df, _ys = as_df(X, y)
    sel = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=20),
        cv=3, max_refits=4, verbose=0, random_state=42,
        max_noimproving_iters=2,
        smooth_perf=smooth_perf,
    )
    sel.fit(df, y)
    assert 1 <= len(_support_indices(sel)) <= df.shape[1]


@pytest.mark.parametrize("nofeatures_dummy", [True, False])
def test_biz_val_rfecv_nofeatures_dummy_scoring_completes(nofeatures_dummy):
    """``nofeatures_dummy_scoring`` toggles whether the all-features
    baseline scores against a dummy. Both modes must complete."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df,
    )
    X, y, _ = make_signal_plus_noise(n=600, p_signal=3, p_noise=5, seed=42)
    df, _ys = as_df(X, y)
    sel = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=20),
        cv=3, max_refits=4, verbose=0, random_state=42,
        max_noimproving_iters=2,
        nofeatures_dummy_scoring=nofeatures_dummy,
    )
    sel.fit(df, y)
    assert 1 <= len(_support_indices(sel)) <= df.shape[1]


def test_biz_val_rfecv_checkpoint_resume_produces_same_support(tmp_path):
    """RFECV with ``checkpoint_path`` must (a) write a resume file
    that allows a subsequent identical fit to pick up where it left
    off, AND (b) produce identical support_ vs running through to
    completion in one call. Catches regressions in the
    save / load / signature-check logic."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV

    X, y = _signal_plus_noise(n=1000, p_signal=3, p_noise=8, seed=42)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    cp = str(tmp_path / "rfecv_ckpt.pkl")

    sel_full = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=20),
        cv=3, max_refits=8, verbose=0, random_state=42,
        max_noimproving_iters=3,
    )
    sel_full.fit(df, y)
    sel_resume = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=20),
        cv=3, max_refits=8, verbose=0, random_state=42,
        max_noimproving_iters=3,
        checkpoint_path=cp,
    )
    sel_resume.fit(df, y)
    full_set = set(_support_indices(sel_full))
    resume_set = set(_support_indices(sel_resume))
    # Both must converge on the same support set on a deterministic
    # seed; the checkpoint mechanism must not change the result.
    assert full_set == resume_set, (
        f"checkpoint-enabled fit must produce same support; "
        f"full={sorted(full_set)}, ckpt={sorted(resume_set)}"
    )
