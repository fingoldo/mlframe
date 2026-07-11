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

from typing import Any, Callable, Optional, Sequence, Union

import numpy as np


class _LazyModule:
    """Transparent lazy proxy: imports the wrapped module on first attribute
    access. Keeps matplotlib (~0.15s) off the eager import path -- this module
    is reachable from feature-selection imports, yet plt is only used by the
    optimizer's plotting callbacks.
    """

    def __init__(self, name: str):
        self._lm_name = name
        self._lm_mod: Optional[Any] = None

    def __getattr__(self, attr):
        if self._lm_mod is None:
            import importlib

            self._lm_mod = importlib.import_module(self._lm_name)
        return getattr(self._lm_mod, attr)


plt = _LazyModule("matplotlib.pyplot")

from enum import Enum, auto
from timeit import default_timer as timer

# ----------------------------------------------------------------------------------------------------------------------------
# Inits
# ----------------------------------------------------------------------------------------------------------------------------

SMALL_VALUE = 1e-4
BIG_VALUE = 1e6

# Sentinel distinguishing "surrogate not yet trainable" (transient: no evaluations submitted yet, or all
# known targets identical so the model cannot be fit) from a genuine None == "search space exhausted".
# Callers that drive the optimizer in a loop must NOT terminate on _NOT_READY -- it means "try again after
# submitting more evaluations", whereas None means "every candidate has been checked / suggested".
class _SearchNotReady:
    """Sentinel type for :data:`NOT_READY`; see the module comment above for the NOT_READY-vs-None contract."""

    __slots__ = ()
    def __repr__(self):
        return "NOT_READY"

NOT_READY = _SearchNotReady()

# ----------------------------------------------------------------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------------------------------------------------------------


class OptimizationDirection(Enum):
    """Whether the optimizer is hunting for the highest or lowest evaluation."""

    Minimize = auto()
    Maximize = auto()


class CandidateSamplingMethod(Enum):
    """Strategy for picking the initial (pre-surrogate) exploration candidates."""

    Random = auto()
    Equidistant = auto()
    Fibonacci = auto()
    ReversedFibonacci = auto()


class OptimizationProgressPlotting(Enum):
    """How often ``MBHOptimizer`` renders its diagnostic plot during the search loop."""

    No = auto()
    Final = auto()  # Plotting is done once, after the search finishes
    OnScoreImprovement = auto()  # Plotting is done on every improving candidate
    Regular = auto()  # Plotting is done on every candidate


# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def compute_candidates_exploration_scores(search_space: Sequence, known_candidates: Sequence):
    """Compute distances from all candidates to known points.
    Assuming search_space is sorted.
    """

    distances = np.zeros(len(search_space))  # distances to closest checked points
    if len(known_candidates) == 0:
        # No checked points yet -> every search-space point is maximally far; the loop below would leave r/lo unbound.
        return distances
    indices = {el: i for i, el in enumerate(search_space)}

    lo = None
    for i in sorted(known_candidates):
        r = indices[i]
        if lo is None:
            distances[:r] = np.abs(search_space[0:r] - search_space[r])
        else:
            m = (lo + r) // 2
            distances[lo:m] = np.abs(search_space[lo:m] - search_space[lo])
            distances[m:r] = np.abs(search_space[m:r] - search_space[r])
        lo = r
    distances[r:] = np.abs(search_space[r:] - search_space[r])

    return distances


def generate_fibonacci(n: int):
    """Creates Fibonacci sequence for a given n."""

    if n <= 0:
        return []

    fibonacci_sequence = [0, 1]
    while len(fibonacci_sequence) < n:
        next_number = fibonacci_sequence[-1] + fibonacci_sequence[-2]
        fibonacci_sequence.append(next_number)

    return np.array(fibonacci_sequence, dtype=np.int64)


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


def plot_search_state(
    search_space,
    next_cand: int,
    new_y: float,
    best_candidate: Optional[int],
    best_evaluation: float,
    nsteps: int,
    expected_fitness: Optional[np.ndarray],
    y_pred: Optional[np.ndarray],
    y_std: Optional[np.ndarray],
    ground_truth: Optional[np.ndarray],
    known_candidates: np.ndarray,
    known_evaluations: np.ndarray,
    skip_candidates: Sequence,
    acquisition_method: str,
    mode: str,
    additional_info: str,
    figsize: tuple = (8, 4),
    font_size: int = 10,
    x_label="nfeatures",
    y_label="score",
    expected_fitness_color: str = "green",
    legend_location: str = "lower right",
):
    """Render the current optimizer state: known evaluations, the surrogate's predicted fitness (+ std band when available), and the just-suggested candidate, on a dual-axis matplotlib figure.

    Purely diagnostic -- called from :meth:`MBHOptimizer.submit_evaluations` when ``plotting`` requests it (``OnScoreImprovement`` / ``Regular``); never affects the search itself.
    """

    # ---------------------------------------------------------------------------------------------------------------
    # Plot expected fitness of the points
    # ---------------------------------------------------------------------------------------------------------------

    plt.rcParams.update({"font.size": font_size})
    fig, axMain = plt.subplots(sharex=True, figsize=figsize, layout="tight")
    axExpectedFitness = axMain.twinx()

    if expected_fitness is not None:
        axExpectedFitness.plot(search_space, expected_fitness, color=expected_fitness_color, linestyle="dashed", label=acquisition_method, alpha=0.3)
        # axExpectedFitness.plot(search_space, y_std, color=expected_fitness_color,linestyle='dashed', label='y_std')
        # axExpectedFitness.plot(search_space, distances, color=expected_fitness_color,linestyle='dotted', label='distances')

    # ---------------------------------------------------------------------------------------------------------------
    # Plot the black box function, surrogate function, known points
    # ---------------------------------------------------------------------------------------------------------------

    if ground_truth is not None:
        axMain.plot(search_space, ground_truth, color="black", label="Ground truth")
    if y_pred is not None:
        axMain.plot(search_space, y_pred, color="red", linestyle="dashed", label="Surrogate Function")
        axMain.fill_between(search_space, y_pred - y_std, y_pred + y_std, color="blue", alpha=0.2)

    axMain.scatter(known_candidates, known_evaluations, color="blue", label="Known Points")

    if skip_candidates:
        idx = ~np.isin(known_candidates, skip_candidates)
        if idx.sum() > 0:
            axMain.set_ylim([known_evaluations[idx].min(), None])

    axExpectedFitness.set_yticklabels([])
    axExpectedFitness.set_yticks([])
    axExpectedFitness.set_ylabel(acquisition_method, color=expected_fitness_color)
    # axExpectedFitness.legend()
    axMain.set_xlabel(x_label)
    axMain.set_ylabel(y_label)

    # ---------------------------------------------------------------------------------------------------------------
    # Plot next candidate
    # ---------------------------------------------------------------------------------------------------------------

    axMain.scatter(next_cand, new_y, color="red", marker="D", label="Next candidate")

    # plt.xlabel(x_label)
    # plt.ylabel(y_label)
    # plt.title(f"Iteration #{nsteps}, mode={mode} {additional_info}")
    axMain.set_title(f"Iteration #{nsteps}, mode={mode} {additional_info}, best={best_evaluation:.6f}@{best_candidate:_}")
    axMain.legend(loc=legend_location)
    # Non-blocking show: ``plt.show()`` (default block=True) made the Qt
    # window MODAL, freezing the optimisation loop until the user closed
    # every figure manually. RFECV with optimizer_plotting='OnScoreImprovement'
    # spawns a window per score improvement -> dozens of stuck modals on
    # the desktop. block=False renders the window non-modally; the
    # plt.pause() flush gives the GUI event loop a tick to draw.
    try:
        plt.show(block=False)
        plt.pause(0.001)
    except Exception:  # nosec B110 - non-trivial body
        # Headless / Agg backend: show is a no-op, pause may not work
        # without a backend. Failure here must NEVER block training.
        pass
    plt.close(fig)


# Wave 100 (2026-07-11): _ETRWithStd + MBHOptimizer (~600 lines) moved to
# sibling file _optimization_search.py to drop this file below the 1k-line
# monolith threshold. Imported here (after plot_search_state, which the
# sibling imports back) to avoid a circular partial-import; re-exported so
# existing callers (`from mlframe.models.optimization import MBHOptimizer`)
# keep working.
from ._optimization_search import _ETRWithStd, MBHOptimizer  # noqa: F401
