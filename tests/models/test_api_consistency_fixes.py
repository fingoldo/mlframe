"""Regression tests for PUBLIC-API consistency / validation fixes in mlframe.models.** and evaluation/bootstrap.

Each test pins one fix and fails on the pre-fix code:
  API4  - MBHOptimizer wrapper default == class default (use_stds_for_exploitation, init_evaluate_descending)
  API5  - create_ctr_params GPU_ENABLED default == CatboostParamsOptimizer GPU_ENABLED default
  API6  - materialised and streaming ensemble default blend method agree
  API17 - dead input_dtype param removed from MBHOptimizer + optimize_finite_onedimensional_search_space wrapper
  API18 - ParamsOptimizer.create_study / report_trial_results are documented no-ops AND warn
  API23 - suggest_candidate disambiguates not-ready (NOT_READY) from exhaustion (None)
  API24 - get_model does not mutate the caller's DataFrame (keeps "target" column)
  API25 - get_model cache key does not collide for two different feature sets under one experiment name
  API26 - justify_estimator's CV gate is reproducible across calls with the same seed
  API27 - empty member set raises ValueError (both ensemble paths)
  API-P2 - ddof consistency (materialised == streaming), score min_samples_for_parallel default, selection ndarray yield
  API3  - auc_ci n_bootstrap default aligned to 1000
  API7  - auc_ci result carries a "point" alias key
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest


# ----------------------------------------------------------------------------------------------------------------------------
# API4 / API17 - MBHOptimizer wrapper vs class signature alignment
# ----------------------------------------------------------------------------------------------------------------------------


def _defaults(func):
    """Map each parameter name of func to its default value, skipping parameters without one."""
    return {p.name: p.default for p in inspect.signature(func).parameters.values() if p.default is not inspect.Parameter.empty}


def test_api4_wrapper_and_class_agree_on_defaults():
    """MBHOptimizer.__init__ and its optimize_finite_onedimensional_search_space wrapper share the same defaults."""
    from mlframe.models.optimization import MBHOptimizer, optimize_finite_onedimensional_search_space

    cls = _defaults(MBHOptimizer.__init__)
    wrap = _defaults(optimize_finite_onedimensional_search_space)
    for key in ("use_stds_for_exploitation", "init_evaluate_descending", "init_evaluate_ascending"):
        assert cls[key] == wrap[key], f"{key}: class={cls[key]} wrapper={wrap[key]} must agree"


def test_api17_input_dtype_removed():
    """The dead input_dtype param is removed from both MBHOptimizer and the search-space wrapper."""
    from mlframe.models.optimization import MBHOptimizer, optimize_finite_onedimensional_search_space

    assert "input_dtype" not in inspect.signature(MBHOptimizer.__init__).parameters
    assert "input_dtype" not in inspect.signature(optimize_finite_onedimensional_search_space).parameters


# ----------------------------------------------------------------------------------------------------------------------------
# API5 - GPU default consistency in tuning.py
# ----------------------------------------------------------------------------------------------------------------------------


def test_api5_gpu_enabled_default_consistent():
    """create_ctr_params and CatboostParamsOptimizer agree on the GPU_ENABLED default (False)."""
    from mlframe.models.tuning import create_ctr_params, CatboostParamsOptimizer

    ctr_default = inspect.signature(create_ctr_params).parameters["GPU_ENABLED"].default
    opt_default = inspect.signature(CatboostParamsOptimizer.__init__).parameters["GPU_ENABLED"].default
    assert ctr_default is False and opt_default is False


# ----------------------------------------------------------------------------------------------------------------------------
# API6 - default blend method agrees across materialised / streaming
# ----------------------------------------------------------------------------------------------------------------------------


def test_api6_default_blend_method_agrees():
    """Materialised and streaming ensemble predictors default to the same blend method."""
    from mlframe.models.ensembling.predict import (
        ensemble_probabilistic_predictions,
        ensemble_probabilistic_predictions_streaming,
    )

    mat = inspect.signature(ensemble_probabilistic_predictions).parameters["ensemble_method"].default
    stream = inspect.signature(ensemble_probabilistic_predictions_streaming).parameters["ensemble_method"].default
    assert mat == stream


# ----------------------------------------------------------------------------------------------------------------------------
# API18 - no-op stubs warn + return None
# ----------------------------------------------------------------------------------------------------------------------------


def test_api18_stubs_warn_and_return_none(caplog):
    """ParamsOptimizer.create_study/report_trial_results are documented no-op stubs that warn and return None."""
    from mlframe.models.tuning import ParamsOptimizer

    opt = ParamsOptimizer(random_state=0)
    with caplog.at_level("WARNING"):
        assert opt.create_study(task_id="t") is None
        assert opt.report_trial_results({"r2": 0.5}) is None
    msgs = " ".join(r.message for r in caplog.records)
    assert "no-op stub" in msgs
    # The docstrings must declare the stub status (contract matches behaviour).
    assert "STUB" in (ParamsOptimizer.create_study.__doc__ or "")
    assert "STUB" in (ParamsOptimizer.report_trial_results.__doc__ or "")


# ----------------------------------------------------------------------------------------------------------------------------
# API23 - not-ready vs exhaustion disambiguation
# ----------------------------------------------------------------------------------------------------------------------------


def test_api23_not_ready_distinct_from_none():
    """suggest_candidate returns the NOT_READY sentinel, distinct from None, while the surrogate is untrainable."""
    from mlframe.models.optimization import MBHOptimizer, NOT_READY

    opt = MBHOptimizer(search_space=np.arange(0, 20), model_name="ETR", model_params={}, init_num_samples=5, random_state=0, greedy_prob=0.0)
    opt.pre_seeded_candidates = []
    opt.known_candidates = np.array([1, 2, 3])
    opt.known_evaluations = np.array([])  # transient: surrogate not yet trainable
    opt.last_retrain_ninputs = 0
    opt.best_candidate = 1
    res = opt.suggest_candidate()
    assert res is NOT_READY
    assert res is not None  # the whole point: not-ready must NOT read as exhaustion


def test_api23_not_ready_does_not_terminate_search():
    """The search loop keeps evaluating through repeated NOT_READY suggestions instead of treating one as exhaustion."""
    # optimize_finite_onedimensional_search_space must keep evaluating even when early suggestions are NOT_READY
    # (all-identical targets), instead of breaking on the first NOT_READY as if the space were exhausted.
    from mlframe.models import optimization as opt_mod

    calls = {"n": 0}

    def evalfn(x):
        """Returns a constant target so the surrogate stays untrainable and keeps emitting NOT_READY."""
        calls["n"] += 1
        return 1.0  # constant -> surrogate "all targets same" -> NOT_READY repeatedly

    opt_mod.optimize_finite_onedimensional_search_space(
        eval_candidate_func=evalfn,
        search_space=np.arange(0, 12),
        direction=opt_mod.OptimizationDirection.Maximize,
        init_num_samples=3,
        max_fevals=8,
        model_name="ETR",
        model_params={},
        random_state=0,
        verbose=0,
    )
    # Pre-fix: the loop breaks on the first NOT_READY-as-None and evaluates only the pre-seeded samples.
    # Post-fix: the fallback keeps feeding unchecked candidates, so we reach more than the 3 seeds.
    assert calls["n"] > 3


def test_predict_runtimes_preempts_before_exceeding_budget():
    # predict_runtimes=True must stop the loop BEFORE running a candidate whose predicted duration
    # (mean of past per-eval durations) would push elapsed time past max_runtime_mins, instead of
    # only checking max_runtime_mins reactively after the eval already ran.
    """predict_runtimes=True stops the loop before a predicted-duration eval would exceed max_runtime_mins."""
    from mlframe.models import optimization as opt_mod
    import time

    calls = {"n": 0}

    def evalfn(x):
        """Records a call and sleeps briefly to simulate a per-eval duration for runtime prediction."""
        calls["n"] += 1
        time.sleep(0.05)
        return float(x)

    opt_mod.optimize_finite_onedimensional_search_space(
        eval_candidate_func=evalfn,
        search_space=np.arange(0, 200),
        direction=opt_mod.OptimizationDirection.Maximize,
        init_num_samples=3,
        max_runtime_mins=0.05 / 60.0 * 4,  # budget for ~4 evals at 0.05s each
        predict_runtimes=True,
        model_name="ETR",
        model_params={},
        random_state=0,
        verbose=0,
    )
    # Loop must have stopped near the predicted-exhaustion point, not run through the whole 200-item space.
    assert calls["n"] < 50


# ----------------------------------------------------------------------------------------------------------------------------
# API24 / API25 / API26 - get_model + justify_estimator
# ----------------------------------------------------------------------------------------------------------------------------


def _make_trials(n=60, seed=0, extra_col=False):
    """Builds a toy regression trials frame with an optional extra feature column."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    if extra_col:
        df["c"] = rng.normal(size=n)
    df["target"] = df["a"] * 2.0 + rng.normal(scale=0.01, size=n)
    return df


def test_api24_get_model_does_not_mutate_caller_df():
    """get_model leaves the caller's trials DataFrame untouched, including the "target" column."""
    from mlframe.models.tuning import get_model

    trials = _make_trials()
    cols_before = list(trials.columns)
    get_model(experiment_name="exp_api24", trials=trials, cat_features=[], cv=3, scoring="r2", min_score=-1e9)
    assert list(trials.columns) == cols_before
    assert "target" in trials.columns


def test_api25_cache_does_not_collide_on_distinct_feature_sets():
    """get_model's cache key does not collide across two different feature sets under the same experiment name."""
    from mlframe.models.tuning import get_model, trained_models

    trained_models.clear()
    name = "exp_api25_shared"
    # Two callers share the experiment name but optimise different feature spaces.
    df2 = _make_trials(extra_col=False)
    df3 = _make_trials(extra_col=True)
    _, cols2, _ = get_model(experiment_name=name, trials=df2, cat_features=[], cv=3, scoring="r2", min_score=-1e9)
    _, cols3, _ = get_model(experiment_name=name, trials=df3, cat_features=[], cv=3, scoring="r2", min_score=-1e9)
    # Pre-fix the second call would hit the first's cached model (2-feature model_columns) -> wrong columns.
    assert set(cols2) == {"a", "b"}
    assert set(cols3) == {"a", "b", "c"}


def test_api26_gate_reproducible_with_same_seed():
    """justify_estimator's CV gate score is reproducible across calls given the same random_state."""
    from mlframe.models.tuning import justify_estimator
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["x0", "x1", "x2"])
    y = X["x0"].values * 1.5 + rng.normal(scale=0.5, size=120)
    s1 = justify_estimator(LinearRegression(), X, y, cv=4, refit=False, scoring="r2", min_score=-1e9, random_state=123)[1]
    s2 = justify_estimator(LinearRegression(), X, y, cv=4, refit=False, scoring="r2", min_score=-1e9, random_state=123)[1]
    assert s1 == s2


# ----------------------------------------------------------------------------------------------------------------------------
# API27 - empty members raise (both paths)
# ----------------------------------------------------------------------------------------------------------------------------


def test_api27_empty_members_raise():
    """Both ensemble prediction paths raise ValueError when given no non-None member predictions."""
    from mlframe.models.ensembling.predict import (
        ensemble_probabilistic_predictions,
        ensemble_probabilistic_predictions_streaming,
    )

    with pytest.raises(ValueError, match="no non-None member predictions"):
        ensemble_probabilistic_predictions(None, None, verbose=False)
    with pytest.raises(ValueError, match="no non-None member predictions"):
        ensemble_probabilistic_predictions_streaming(None, verbose=False)


# ----------------------------------------------------------------------------------------------------------------------------
# API-P2 - ddof consistency, score default, selection ndarray
# ----------------------------------------------------------------------------------------------------------------------------


def test_apip2_uncertainty_ddof_consistent_across_paths():
    """Materialised and streaming ensemble paths compute uncertainty with the same ddof, matching bit-for-bit."""
    from mlframe.models.ensembling.predict import (
        ensemble_probabilistic_predictions,
        ensemble_probabilistic_predictions_streaming,
    )

    rng = np.random.default_rng(0)
    preds = [rng.uniform(0.05, 0.95, size=(50, 3)) for _ in range(5)]
    _, unc_mat, _ = ensemble_probabilistic_predictions(*preds, ensemble_method="arithm", uncertainty_quantile=0.5, verbose=False)
    _, unc_str, _ = ensemble_probabilistic_predictions_streaming(*preds, ensemble_method="arithm", verbose=False)
    np.testing.assert_allclose(unc_mat, unc_str, atol=1e-12)


def test_apip2_score_min_samples_default_matches_docstring():
    """score_ensemble's min_samples_for_parallel default is 1_000_000, matching its documented contract."""
    from mlframe.models.ensembling.score import score_ensemble

    assert inspect.signature(score_ensemble).parameters["min_samples_for_parallel"].default == 1_000_000


def test_apip2_selection_yields_ndarrays():
    """The selection module's CV splitter yields (train, test) indices as ndarrays, not other sequence types."""
    from mlframe.models import selection as sel_mod

    # Find the splitter class that yields (train, test) index arrays.
    splitter_cls = None
    for obj in vars(sel_mod).values():
        if inspect.isclass(obj) and hasattr(obj, "split") and obj.__module__ == sel_mod.__name__:
            splitter_cls = obj
            break
    assert splitter_cls is not None, "no splitter class found in selection.py"
    inst = splitter_cls(n_splits=3)
    X = np.zeros((30, 2))
    groups = np.repeat(np.arange(10), 3)
    train, test = next(iter(inst.split(X, groups=groups)))
    assert isinstance(train, np.ndarray) and isinstance(test, np.ndarray)


# ----------------------------------------------------------------------------------------------------------------------------
# API3 / API7 - auc_ci defaults + "point" alias
# ----------------------------------------------------------------------------------------------------------------------------


def test_api3_auc_ci_n_bootstrap_default_aligned():
    """auc_ci's n_bootstrap default is aligned to 1000."""
    from mlframe.evaluation.bootstrap import auc_ci

    assert inspect.signature(auc_ci).parameters["n_bootstrap"].default == 1000


def test_api7_auc_ci_has_point_alias():
    """auc_ci's result dict carries a "point" key aliasing the "auc" value, for both delong and bootstrap methods."""
    from mlframe.evaluation.bootstrap import auc_ci

    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 50)
    score = rng.uniform(size=100) + y * 0.5
    for method in ("delong", "bootstrap"):
        res = auc_ci(y, score, method=method, random_state=0)
        assert "point" in res, f"{method}: missing 'point' alias"
        assert res["point"] == res["auc"]
