"""Class for numerical optimization. In ML, serves to optimize hyperparameters, select best features."""

from __future__ import annotations

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from timeit import default_timer as timer
from typing import Callable, Optional, Sequence, Union

import numpy as np

# Shared with ._optimization_search (both siblings import these from the leaf module to avoid the
# circular partial-import a direct cross-import between the two would create); re-exported here for BC.
from ._optimization_shared import (
    BIG_VALUE,  # noqa: F401
    NOT_READY,
    SMALL_VALUE,  # noqa: F401
    CandidateSamplingMethod,
    OptimizationDirection,
    OptimizationProgressPlotting,
    _SearchNotReady,  # noqa: F401
    compute_candidates_exploration_scores,  # noqa: F401
    generate_fibonacci,  # noqa: F401
    plot_search_state,  # noqa: F401
)

# Wave 100 (2026-07-11): _ETRWithStd + MBHOptimizer (~600 lines: ctor param
# checks, RNG discipline, init sampling, suggest_candidate/submit_evaluations)
# moved to sibling file _optimization_search.py to drop this file below the
# 1k-line monolith threshold. Re-imported below so
# optimize_finite_onedimensional_search_space (further down this file) and
# existing external callers
# (`from mlframe.models.optimization import MBHOptimizer`) keep working. The
# import itself moves to the BOTTOM of this file (after plot_search_state,
# which the sibling imports back) to avoid a circular partial-import; the one
# call site below (optimize_finite_onedimensional_search_space) only needs the
# name resolved by the time it's CALLED, not when it's defined.


def optimize_finite_onedimensional_search_space(
    search_space: Sequence,  # search space, all possible input combinations to check
    eval_candidate_func: Callable,  # fitness function to be optimized over the search space
    ground_truth: Optional[np.ndarray] = None,  # known true fitness of the entire search space
    direction: OptimizationDirection = OptimizationDirection.Maximize,
    known_candidates: Optional[Union[list, np.ndarray]] = None,
    known_evaluations: Optional[Union[list, np.ndarray]] = None,
    # stopping conditions
    max_runtime_mins: Optional[float] = None,
    predict_runtimes: bool = False,  # when True and max_runtime_mins is set, skip a candidate whose predicted duration (mean of past evals) would blow the budget
    max_fevals: Optional[int] = None,
    best_desired_score: Optional[float] = None,
    max_noimproving_iters: Optional[int] = None,
    # inits
    seeded_inputs: Optional[Sequence] = None,  # seed items you want to be explored from start
    init_num_samples: Union[float, int] = 5,  # how many samples to generate & evaluate before fitting surrogate self.model
    init_evaluate_ascending: bool = False,
    init_evaluate_descending: bool = True,
    init_sampling_method: CandidateSamplingMethod = CandidateSamplingMethod.Equidistant,  # random, equidistant, fibo, rev_fibo?
    # EE dilemma
    exploitation_probability: float = 0.8,
    skip_best_candidate_prob: float = 0.0,  # pick the absolute best predicted candidate, or probabilistically the best
    use_distances_on_preds_collision: bool = True,
    use_stds_for_exploitation: bool = False,
    dist_scaling_coefficient: float = 0.5,
    # self.model
    acquisition_method: str = "EE",
    model_name: str = "CBQ",  # actual estimator instance here? + for lgbm also the linear mode flag
    model_params: Optional[dict] = None,
    quantile: float = 0.01,
    # Visualisation
    plotting: OptimizationProgressPlotting = OptimizationProgressPlotting.No,
    plotting_ndim: int = 1,  # number of dimensions to use for visualisation
    figsize: tuple = (8, 4),
    font_size: int = 10,
    x_label="nfeatures",
    y_label="score",
    expected_fitness_color: str = "green",
    legend_location: str = "best",
    # small settings
    verbose: int = 0,
    random_state: Union[int, np.random.Generator, None] = None,
) -> tuple:
    """Finds extremum of some (possibly multi-input) function F that is supposedly costly to estimate (like in HPT, FS tasks).
    Function F can also be a numerical sequence in form of some y scores array.

    To achieve high search efficiency, uses a surrogate self.model to mimic our true 'black box' function behaviour.
    Once F's values in a few points are computed, surrogate is fit and used to predict F in its entire definition range. Then, the most promising points (
    having the highest predicted values) can be checked directly.

    As surrogate models, can use quantile regressions (namely, one of modern gradient boosting libs that implement it), as they can have a notion of uncertainty.
    Do we REALLY need only models capable of uncertainty estimates?
    To internally estimate wellness of surrogate self.model, compares it to the best of the dummy regressors (on the test set).
    Also allows for early stopping (do we require it? is it useful at all here?).

    Uses exploration/exploitation modes (controlled by exploitation_probability parameter):
        at exploration step:
            points with highest self.model uncertainty and the most distant from already visited ones are suggested;
        at exploitation step:
            1) if a point with the current highest predicted value is unexplored, suggests it
            2) otherwise, picks points that can give highest result, ie have the best predicted value+uncertainty+are located far from known regions
            (in a log scale, to lower dist factor importance)

        Next candidate can be picked in a deterministic (strict argmax) or use_probabilistic_candidate_selection (by drawing a random number & comparing it with the candidate's choosing prob) manner.

    Can hold on to a pre-defined time/feval budget.

    Optionally plots the search path.

    Ask-tell interface:
    via iterator with yields?

    Working as a class:
        1) init the object
        2) feed initial population,retrain
        3) suggest next candidate(s), wait for the client to evaluate it(them)
        4) get next input-output batch, from the client, retrain

    Challenges:
    1) V allow non-quantile estimators
    2) allow multiple estimators
    3) when using multiple estimators, allow an option of estimating stds directly from their forecasts for the same points
    4) allow ES (train/test splts can vary across estimators)
    5) add control via dummy self.model: if dummy is beaten, allow exploitation step. else only exploration
    6) batch mode, when more than 1 suggestions are produced in a single call
    7) parallel
    8) async

    """

    optimizer = MBHOptimizer(
        search_space=np.asarray(search_space),
        ground_truth=ground_truth,
        direction=direction,
        known_candidates=known_candidates,
        known_evaluations=known_evaluations,
        seeded_inputs=seeded_inputs,
        init_num_samples=init_num_samples,
        init_evaluate_ascending=init_evaluate_ascending,
        init_evaluate_descending=init_evaluate_descending,
        init_sampling_method=init_sampling_method,
        exploitation_probability=exploitation_probability,
        skip_best_candidate_prob=skip_best_candidate_prob,
        use_distances_on_preds_collision=use_distances_on_preds_collision,
        use_stds_for_exploitation=use_stds_for_exploitation,
        dist_scaling_coefficient=dist_scaling_coefficient,
        acquisition_method=acquisition_method,
        model_name=model_name,
        model_params=model_params,
        quantile=quantile,
        plotting=plotting,
        plotting_ndim=plotting_ndim,
        figsize=figsize,
        font_size=font_size,
        x_label=x_label,
        y_label=y_label,
        expected_fitness_color=expected_fitness_color,
        legend_location=legend_location,
        verbose=verbose,
        random_state=random_state,
    )

    # ----------------------------------------------------------------------------------------------------------------------------
    # Inits
    # ----------------------------------------------------------------------------------------------------------------------------

    start_time = timer()
    ran_out_of_time = False

    # ----------------------------------------------------------------------------------------------------------------------------
    # Optimization Loop
    # ----------------------------------------------------------------------------------------------------------------------------

    while True:

        # get_best_dummy_score(estimator=estimator,X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        next_candidate = optimizer.suggest_candidate()
        if next_candidate is NOT_READY:
            # Surrogate not yet trainable (no evaluations submitted, or all known targets identical).
            # This is transient, NOT exhaustion -- fall back to any still-unchecked search-space point so
            # the search keeps producing evaluations instead of terminating early (silent truncation).
            checked = set(optimizer.known_candidates.tolist()) | set(optimizer.suggested_candidates.keys())
            fallback = next((c for c in np.asarray(optimizer.search_space).tolist() if c not in checked), None)
            if fallback is None:
                if verbose:
                    logger.info("Search space fully checked, quitting")
                break
            next_candidate = fallback
            optimizer.suggested_candidates[fallback] = timer()
        elif next_candidate is None:
            if verbose:
                logger.info("Search space fully checked, quitting")
            break

        if predict_runtimes and max_runtime_mins and optimizer.evaluated_candidates:
            mean_eval_duration = float(np.mean([c["duration"] for c in optimizer.evaluated_candidates if c["duration"] is not None]))
            if np.isfinite(mean_eval_duration) and (timer() - start_time) + mean_eval_duration > max_runtime_mins * 60:
                ran_out_of_time = True
                if verbose:
                    logger.info(
                        "max_runtime_mins=%s would be exceeded by the next candidate (predicted duration=%.2fs); quitting preemptively.",
                        f"{max_runtime_mins:_.1f}",
                        mean_eval_duration,
                    )
                break

        eval_start_time = timer()
        next_evaluation = eval_candidate_func(next_candidate)
        next_duration = timer() - eval_start_time
        optimizer.submit_evaluations(candidates=[next_candidate], evaluations=[next_evaluation], durations=[next_duration])

        # ----------------------------------------------------------------------------------------------------------------------------
        # Checking exit conditions
        # ----------------------------------------------------------------------------------------------------------------------------

        if best_desired_score:
            if direction == OptimizationDirection.Maximize:
                if optimizer.best_evaluation >= best_desired_score:
                    if verbose:
                        logger.info("best_desired_score=%s reached.", f"{optimizer.best_evaluation:_.6f}")
                    break
            elif direction == OptimizationDirection.Minimize:
                if optimizer.best_evaluation <= best_desired_score:
                    if verbose:
                        logger.info("best_desired_score=%s reached.", f"{optimizer.best_evaluation:_.6f}")
                    break

        if max_runtime_mins and not ran_out_of_time:
            ran_out_of_time = (timer() - start_time) > max_runtime_mins * 60
            if ran_out_of_time:
                if verbose:
                    logger.info("max_runtime_mins=%s reached.", f"{max_runtime_mins:_.1f}")
                break

        if max_fevals and optimizer.nsteps >= max_fevals:
            if verbose:
                logger.info("max_fevals=%s reached.", f"{max_fevals:_}")
            break

        if max_noimproving_iters and optimizer.n_noimproving_iters >= max_noimproving_iters:
            if verbose:
                logger.info("Max # of noimproved iters reached: %s", optimizer.n_noimproving_iters)
            break

    return (optimizer.best_candidate, optimizer.best_evaluation), optimizer.evaluated_candidates


# Wave 100 (2026-07-11): _ETRWithStd + MBHOptimizer (~600 lines) moved to sibling file
# _optimization_search.py to drop this file below the 1k-line monolith threshold; both this module and
# the sibling import their shared constants/enums/helpers (including plot_search_state) from the leaf
# ._optimization_shared module instead of from each other, which is what actually avoids the circular
# partial-import (see that module's docstring). Re-exported here so existing callers
# (`from mlframe.models.optimization import MBHOptimizer`) keep working.
from ._optimization_search import _ETRWithStd, MBHOptimizer  # noqa: F401
