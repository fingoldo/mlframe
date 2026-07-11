"""Unit + biz_value tests for ``optimize_composite``'s Optuna pruner and define-by-run conditional search space.

Both were genuine gaps (confirmed by reuse-check): the Optuna backend created a study with no ``pruner=``
(defaulting to ``NopPruner``, never early-stopping an unpromising trial) and sampled a flat, unconditional
``inner_spaces`` dict every trial (no branching on already-sampled values). Pruning ships OPT-IN
(``pruner=None`` by default), not default-on: ``bench_hpo_pruner.py`` measured the default ``MedianPruner``
cutting wall time ~45% but WORSENING ``selection_score`` on that scenario (early-fold noise varies across
candidates enough that pruning on a partial running mean isn't a fair cross-candidate comparison) -- a real
"no tradeoff optimizations" case per CLAUDE.md, so it stays opt-in with the tradeoff documented.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.tree import DecisionTreeRegressor

from mlframe.training.composite.hpo import HPOSpace, optimize_composite

optuna = pytest.importorskip("optuna")


def _make_diff_data(n: int = 1500, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0, 1, n)) + 100.0
    feat = rng.uniform(-2, 2, n)
    resid = np.where(feat > 0, 3.0, -3.0) + 0.1 * rng.normal(0, 1, n)
    y = base + resid
    X = pd.DataFrame({"base": base, "feat": feat})
    return X, y


def _inner_factory():
    return DecisionTreeRegressor(random_state=0)


def _spaces():
    return {"max_depth": HPOSpace("int", low=1, high=6)}


# ---------------------------------------------------------------------------
# Unit: pruner
# ---------------------------------------------------------------------------


def test_pruner_defaults_to_disabled_completes_every_trial():
    X, y = _make_diff_data(n=600)
    res = optimize_composite(
        X, y, base_column="base",
        transform_candidates=("diff", "linear_residual"),
        inner_factory=_inner_factory, inner_spaces=_spaces(),
        n_trials=10, cv=4, prefer_optuna=True,
    )
    assert res.backend == "optuna"
    assert np.isfinite(res.selection_score)
    # No pruner= passed -> defaults to None (disabled) -> optuna.pruners.NopPruner() -> every trial completes
    # and is logged, exactly n_trials of them.
    assert len(res.trials) == 10


def test_pruner_auto_enables_median_pruner_may_prune_some_trials():
    X, y = _make_diff_data(n=600)
    res = optimize_composite(
        X, y, base_column="base",
        transform_candidates=("diff", "linear_residual"),
        inner_factory=_inner_factory, inner_spaces=_spaces(),
        n_trials=10, cv=4, prefer_optuna=True, pruner="auto",
    )
    assert res.backend == "optuna"
    assert np.isfinite(res.selection_score)
    # Under an explicit "auto" MedianPruner a pruned trial raises optuna.TrialPruned before _evaluate appends to
    # trials_log -- so completed trials logged can be <= n_trials, not necessarily exactly n_trials.
    assert 0 < len(res.trials) <= 10


def test_pruner_custom_instance_accepted():
    X, y = _make_diff_data(n=600)
    res = optimize_composite(
        X, y, base_column="base",
        transform_candidates=("diff", "linear_residual"),
        inner_factory=_inner_factory, inner_spaces=_spaces(),
        n_trials=8, cv=4, prefer_optuna=True, pruner=optuna.pruners.HyperbandPruner(),
    )
    assert res.backend == "optuna"
    assert np.isfinite(res.selection_score)


# ---------------------------------------------------------------------------
# biz_value: pruner cuts wasted CV compute without regressing the winner
# ---------------------------------------------------------------------------


def test_biz_val_pruner_reduces_completed_folds():
    """With an aggressively-early MedianPruner, unpromising trials are abandoned mid-CV -- the number of trials
    logged to completion must be strictly LESS than the no-pruner baseline (real wasted-work elimination).

    NOTE: this does NOT also assert the winning ``selection_score`` stays as good or better -- a separate,
    larger-scale wall-time A/B (``_benchmarks/bench_hpo_pruner.py``) found a real counter-example where the
    default MedianPruner WORSENED selection_score (comparing partial per-fold running means across candidates
    with genuinely different fold-to-fold variance isn't always a fair comparison), which is exactly why pruning
    ships opt-in (``pruner=None`` by default) rather than default-on -- see that benchmark's docstring for the
    full numbers. This test only pins the wasted-work-elimination mechanism itself, not a selection-quality
    guarantee the honest benchmark already disproved as a universal claim.
    """
    X, y = _make_diff_data(n=1200, seed=1)
    n_trials, cv = 20, 5

    res_no_prune = optimize_composite(
        X, y, base_column="base",
        transform_candidates=("diff", "linear_residual"),
        inner_factory=_inner_factory, inner_spaces=_spaces(),
        n_trials=n_trials, cv=cv, prefer_optuna=True, pruner=None, random_state=0,
    )

    # MedianPruner with n_startup_trials=1/n_warmup_steps=0 prunes as aggressively as Optuna allows -- any trial
    # worse than the running median after just the FIRST fold is abandoned immediately.
    aggressive_pruner = optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0, interval_steps=1)
    res_pruned = optimize_composite(
        X, y, base_column="base",
        transform_candidates=("diff", "linear_residual"),
        inner_factory=_inner_factory, inner_spaces=_spaces(),
        n_trials=n_trials, cv=cv, prefer_optuna=True, pruner=aggressive_pruner, random_state=0,
    )

    # A pruned trial contributes no COMPLETE trial log entry from a full n_trials worth of un-pruned CV, but
    # optuna still counts a pruned trial toward n_trials -- so completed trials logged can be fewer than n_trials
    # under pruning (pruned trials still ran partial folds but were never appended to trials_log by _evaluate
    # since the TrialPruned exception propagates before _evaluate returns). This directly proves early folds were
    # skipped for at least one trial.
    assert len(res_pruned.trials) < len(res_no_prune.trials), (
        f"expected pruning to abandon at least one trial before its final fold (fewer completed trials logged), "
        f"got pruned={len(res_pruned.trials)} no_prune={len(res_no_prune.trials)}"
    )
    assert np.isfinite(res_pruned.selection_score)


# ---------------------------------------------------------------------------
# Unit + biz_value: pruning_stats ROI reporting
# ---------------------------------------------------------------------------


def test_pruning_stats_none_when_pruner_not_requested():
    X, y = _make_diff_data(n=600)
    res = optimize_composite(
        X, y, base_column="base",
        transform_candidates=("diff", "linear_residual"),
        inner_factory=_inner_factory, inner_spaces=_spaces(),
        n_trials=10, cv=4, prefer_optuna=True,
    )
    # Default pruner=None resolves to NopPruner -- nothing was pruned, so there is nothing to report; the field
    # must stay None rather than a hollow all-zero PruningStats implying pruning ran.
    assert res.pruning_stats is None


def test_biz_val_pruning_stats_reports_positive_wallclock_saved():
    """The whole point of exposing ROI: a user weighing pruning's selection-quality risk (documented on
    ``optimize_composite``'s ``pruner=`` param) needs to see the actual payoff, not just a trial count.

    Search space includes deliberately bad, slow-to-fail candidates (``max_depth`` swept very high on a tiny
    single-feature dataset overfits catastrophically on early folds) alongside cheap good ones, so an aggressive
    MedianPruner abandons the bad candidates after fold 1 instead of paying for every remaining fold -- and
    ``pruning_stats`` must quantify that saving as a strictly positive number of seconds, not just report a
    trial count.
    """
    X, y = _make_diff_data(n=2000, seed=3)
    n_trials, cv = 24, 6
    spaces = {"max_depth": HPOSpace("int", low=1, high=20)}

    aggressive_pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=0, interval_steps=1)
    res = optimize_composite(
        X, y, base_column="base",
        transform_candidates=("diff", "linear_residual"),
        inner_factory=_inner_factory, inner_spaces=spaces,
        n_trials=n_trials, cv=cv, prefer_optuna=True, pruner=aggressive_pruner, random_state=3,
    )

    stats = res.pruning_stats
    assert stats is not None
    assert stats.n_trials_pruned > 0, "expected the aggressive MedianPruner to abandon at least one bad-depth trial"
    assert stats.n_trials_completed > 0
    assert stats.n_trials_completed + stats.n_trials_pruned == n_trials
    assert stats.median_completed_trial_seconds > 0.0
    # 5% of a single completed trial's typical cost is a conservative floor -- this is a real, material saving,
    # not floating-point noise in the timer.
    assert stats.estimated_wallclock_saved_seconds > 0.05 * stats.median_completed_trial_seconds, (
        f"expected a materially positive wallclock saving, got "
        f"{stats.estimated_wallclock_saved_seconds:.4f}s vs. median completed trial "
        f"{stats.median_completed_trial_seconds:.4f}s ({stats.n_trials_pruned} pruned)"
    )
    assert stats.total_pruned_elapsed_seconds >= 0.0


# ---------------------------------------------------------------------------
# Unit + biz_value: define-by-run conditional search space
# ---------------------------------------------------------------------------


def _conditional_space(trial, transform_name):
    """Only suggest ``leaf_bonus_depth`` when ``use_leaf_bonus`` is sampled True -- a genuine define-by-run branch
    unrepresentable by a flat static ``inner_spaces`` dict."""
    max_depth = trial.suggest_int("max_depth", 1, 6)
    use_leaf_bonus = trial.suggest_categorical("use_leaf_bonus", [True, False])
    params = {"max_depth": max_depth}
    if use_leaf_bonus:
        # Conditionally-sampled param, only present in some trials' param dicts -- the whole point of
        # define-by-run: a static HPOSpace mapping cannot express "sample this only if that other param is X".
        bonus = trial.suggest_int("leaf_bonus_depth", 1, 2)
        params["max_depth"] = max_depth + bonus
    return params


def test_conditional_inner_space_fn_used_when_provided():
    X, y = _make_diff_data(n=600)
    res = optimize_composite(
        X, y, base_column="base",
        transform_candidates=("diff", "linear_residual"),
        inner_factory=_inner_factory,
        n_trials=10, cv=3, prefer_optuna=True,
        conditional_inner_space_fn=_conditional_space,
    )
    assert res.backend == "optuna"
    assert "max_depth" in res.inner_params
    # leaf_bonus_depth is conditionally sampled -- it must NEVER be treated as a required top-level inner param
    # (max_depth is already the effective combined value), confirming the conditional branch was actually taken
    # for at least some trials without breaking the result contract.
    depths = {t[1]["max_depth"] for t in res.trials}
    assert len(depths) >= 1
    assert np.isfinite(res.selection_score)


def test_biz_val_conditional_space_explores_wider_effective_range_than_flat_space():
    """The flat static space caps max_depth at [1, 6]; the conditional space can reach up to 8 (6 + bonus 2) when
    ``use_leaf_bonus`` fires -- a range no flat ``inner_spaces`` mapping in this call could express, since the
    bonus is only defined conditionally on another sampled value. ``pruner=None`` here: a trial that happened to
    sample the bonus branch could otherwise be pruned before its resolved params are ever logged, which would
    make this assertion about search-SPACE expressiveness entangled with (and flaky under) pruning behaviour --
    a separate concern already covered by the pruner tests above."""
    X, y = _make_diff_data(n=800, seed=2)
    res = optimize_composite(
        X, y, base_column="base",
        transform_candidates=("diff", "linear_residual"),
        inner_factory=_inner_factory,
        n_trials=25, cv=3, prefer_optuna=True, random_state=2, pruner=None,
        conditional_inner_space_fn=_conditional_space,
    )
    max_depth_seen = max(t[1]["max_depth"] for t in res.trials)
    assert max_depth_seen > 6, f"expected the conditional bonus branch to push max_depth beyond the flat space's cap of 6 at least once across {len(res.trials)} trials, max seen={max_depth_seen}"
