"""Class for numerical optimization. In ML, serves to optimize hyperparameters, select best features."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)


while True:
    try:

        # ----------------------------------------------------------------------------------------------------------------------------
        # Normal Imports
        # ----------------------------------------------------------------------------------------------------------------------------

        from typing import *

        import numpy as np

        from pyutilz.system import tqdmu
        from pyutilz.pythonlib import store_params_in_object, get_parent_func_args

        import matplotlib.pyplot as plt

        from random import random
        from enum import Enum, auto
        from expiringdict import ExpiringDict
        from timeit import default_timer as timer

        from catboost import CatBoostRegressor

    except ModuleNotFoundError as e:

        logger.warning(e)

        if "cannot import name" in str(e):
            raise (e)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Packages auto-install
        # ----------------------------------------------------------------------------------------------------------------------------

        from pyutilz.pythonlib import ensure_installed

        ensure_installed("numpy expiringdict")

    else:
        break

# ----------------------------------------------------------------------------------------------------------------------------
# Inits
# ----------------------------------------------------------------------------------------------------------------------------

SMALL_VALUE = 1e-4
BIG_VALUE = 1e6

# ----------------------------------------------------------------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------------------------------------------------------------


class OptimizationDirection(Enum):
    Minimize = auto()
    Maximize = auto()


class CandidateSamplingMethod(Enum):
    Random = auto()
    Equidistant = auto()
    Fibonacci = auto()
    ReversedFibonacci = auto()


class OptimizationProgressPlotting(Enum):
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
    indices = {}
    for i, el in enumerate(search_space):
        indices[el] = i

    l = None
    for i in sorted(known_candidates):
        r = indices[i]
        if l is None:
            distances[:r] = np.abs(search_space[0:r] - search_space[r])
        else:
            m = (l + r) // 2
            distances[l:m] = np.abs(search_space[l:m] - search_space[l])
            distances[m:r] = np.abs(search_space[m:r] - search_space[r])
        l = r
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


class MBHOptimizer:
    """Optimizer aimed at suggesting prospective candidates to explore."""

    def __init__(
        self,
        search_space: np.ndarray,  # search space, all possible input combinations to check
        ground_truth: np.ndarray = None,  # known true fitness of the entire search space
        direction: OptimizationDirection = OptimizationDirection.Maximize,
        known_candidates: list = [],
        known_evaluations: list = [],
        # inits
        seeded_inputs: Sequence = [],  # seed items you want to be explored from start
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
        model_params: dict = {"iterations": 150},
        quantile: float = 0.01,
        input_dtype=np.int32,
        # Visualisation
        plotting: OptimizationProgressPlotting = OptimizationProgressPlotting.No,
        plotting_ndim: int = 1,  # number of dimensions to use for visualisation
        figsize: tuple = (8, 4),
        font_size: int = 10,
        x_label="nfeatures",
        y_label="score",
        expected_fitness_color: str = "green",
        legend_location: str = "lower right",
        # small settings
        verbose: int = 0,
        suggestions_cache_max_age_sec: int = 60 * 60,  # wait 1 hr before allowing repeated suggestions
        greedy_prob: float = 0.03,
    ):

        # ----------------------------------------------------------------------------------------------------------------------------
        # Params checks
        # ----------------------------------------------------------------------------------------------------------------------------

        assert quantile > 0.0 and quantile < 0.5
        assert len(search_space) > 0
        assert acquisition_method in ("EE")
        assert skip_best_candidate_prob >= 0.0 and skip_best_candidate_prob < 0.5
        assert dist_scaling_coefficient > 0.0 and dist_scaling_coefficient <= 1.0
        assert exploitation_probability >= 0.0 and exploitation_probability <= 1.0

        # ----------------------------------------------------------------------------------------------------------------------------
        # Save params
        # ----------------------------------------------------------------------------------------------------------------------------

        params = get_parent_func_args()
        store_params_in_object(obj=self, params=params)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Inits
        # ----------------------------------------------------------------------------------------------------------------------------

        if not isinstance(known_candidates, np.ndarray):
            known_candidates = np.array(known_candidates)
        if not isinstance(known_evaluations, np.ndarray):
            known_evaluations = np.array(known_evaluations)
        self.known_candidates = known_candidates
        self.known_evaluations = known_evaluations

        self.best_candidate, self.worst_candidate = None, None
        if self.direction == OptimizationDirection.Maximize:
            self.best_evaluation = -BIG_VALUE
            self.worst_evaluation = BIG_VALUE
        else:
            self.best_evaluation = BIG_VALUE
            self.worst_evaluation = -BIG_VALUE

        self.n_steps_since_greedy = 0
        self.n_noimproving_iters = 0
        self.nsteps = 0

        self.suggested_candidates = ExpiringDict(
            max_len=1e6,
            max_age_seconds=suggestions_cache_max_age_sec,
        )
        self.evaluated_candidates = []
        self.last_retrain_ninputs = 0
        self.additional_info = ""
        self.mode = ""

        self.expected_fitness = None
        self.y_pred = None
        self.y_std = None

        pre_seeded_candidates = []

        # ----------------------------------------------------------------------------------------------------------------------------
        # Let's establish initial dataset to fit our surrogate self.model
        # First use pre-seeded values. they must be evaluated strictly in given order.
        # ----------------------------------------------------------------------------------------------------------------------------

        if len(seeded_inputs) > 0:
            mode = "Seeded"
            for x in seeded_inputs:
                if x not in known_candidates and x not in pre_seeded_candidates:
                    pre_seeded_candidates.append(x)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Now sample additional points from across the definition range with some simple algo
        # ----------------------------------------------------------------------------------------------------------------------------

        if init_num_samples > 0:
            if isinstance(init_num_samples, float) and init_num_samples < 1.0:
                init_num_samples = len(search_space) * init_num_samples

            if init_sampling_method == CandidateSamplingMethod.Random:
                sampled_inputs = np.random.choice(search_space, size=min(init_num_samples, len(search_space)), replace=False)
            elif init_sampling_method == CandidateSamplingMethod.Equidistant:
                sampled_indices = np.linspace(0, len(search_space) - 1, init_num_samples).astype(int)
                sampled_inputs = np.array(search_space)[sampled_indices[:init_num_samples]]
            elif init_sampling_method in (CandidateSamplingMethod.Fibonacci, CandidateSamplingMethod.ReversedFibonacci):
                # Fibo!
                fibo_sequence = generate_fibonacci(2 + init_num_samples)[2:]
                fibo_sequence_indices = (fibo_sequence * (len(search_space) - 1) / fibo_sequence.max()).astype(np.int64)
                if init_sampling_method == CandidateSamplingMethod.ReversedFibonacci:
                    fibo_sequence_indices = len(search_space) - 1 - fibo_sequence_indices
                sampled_inputs = np.array(search_space)[fibo_sequence_indices[:init_num_samples]]
            else:
                raise ValueError(f"Sampling method {init_sampling_method} not supported.")

            # let's remove intersection with already checked seeded elements, if any
            sampled_inputs = set(sampled_inputs) - set(seeded_inputs)

            # sometimes it's required to process samples in certain order (like in FE/RFECV tasks, it's better to start with higher number of features, to have more accurate estimates)
            if init_evaluate_ascending:
                sampled_inputs = sorted(sampled_inputs)
            else:
                if init_evaluate_descending:
                    sampled_inputs = sorted(sampled_inputs)[::-1]

            # actual evaluation of initial samples
            if len(sampled_inputs) > 0:
                mode = "Sampled"
                for x in sampled_inputs:
                    if x not in known_candidates and x not in pre_seeded_candidates:
                        pre_seeded_candidates.append(x)

        assert len(pre_seeded_candidates) > 0
        self.pre_seeded_candidates = pre_seeded_candidates

        # ----------------------------------------------------------------------------------------------------------------------------
        # Init of the surrogate model
        # ----------------------------------------------------------------------------------------------------------------------------

        if self.model_name == "CBQ":
            quantiles: Sequence = [quantile, 0.5, 1 - quantile]
            loss_function = "MultiQuantile:alpha=" + ",".join(map(str, quantiles))
            self.model = CatBoostRegressor(
                **model_params,
                loss_function=loss_function,
                verbose=0,
            )
        elif self.model_name == "CB":
            self.model = CatBoostRegressor(
                **model_params,
                verbose=0,
            )

    def suggest_candidate(self):
        """Get next most promising candidate. Keeps a buffer of suggested, but not yet evaluated candidates, to avoid recommending them again.
        We only need a method to get one suggestion at time: in absense of new evaluations, no model retraining is needed, and response should be fast.
        """

        eval_start_time = timer()

        if self.pre_seeded_candidates:

            # ----------------------------------------------------------------------------------------------------------------------------
            # Checking pre-seeded first
            # ----------------------------------------------------------------------------------------------------------------------------

            print("pre_seeded_candidates=", self.pre_seeded_candidates)

            next_candidate = self.pre_seeded_candidates.pop(0)
            self.suggested_candidates[next_candidate] = eval_start_time

            return next_candidate

        else:

            # If no improvement found for a long time, become greedy and check points closest to known optimum, regardless
            # of underlying model's opinion.

            greedy_prob = min(self.greedy_prob * min(self.n_noimproving_iters, self.n_steps_since_greedy + 1), 1.0)
            if random() < greedy_prob:
                self.n_steps_since_greedy = 0

                expected_fitness = np.abs(np.array(self.search_space) - self.best_candidate)
                for _ in range(100):
                    # Just pick first unchecked and unsuggested candidate with the highest fitness
                    for best_idx in np.argsort(expected_fitness):
                        next_candidate = self.search_space[best_idx]
                        if next_candidate not in self.known_candidates and next_candidate not in self.suggested_candidates:
                            self.suggested_candidates[next_candidate] = eval_start_time
                            logger.info(
                                f"I became greedy! Recommending {next_candidate} that is closest unchecked to the best known so far {self.best_candidate} with eval={self.best_evaluation}"
                            )
                            return next_candidate
            else:

                self.n_steps_since_greedy += 1

                # ----------------------------------------------------------------------------------------------------------------------------
                # Fit surrogate model to known points & their evaluations
                # ----------------------------------------------------------------------------------------------------------------------------

                if len(self.known_candidates) > self.last_retrain_ninputs:

                    # First need to check that targets are not all the same:
                    if np.all(self.known_evaluations == self.known_evaluations[0]):
                        logger.warn(f"All targets are the same! Can't train the underlying process model.")
                        return None

                    if not hasattr(self.model, "partial_fit"):
                        self.model.fit(self.known_candidates.reshape(-1, 1), self.known_evaluations)
                    else:
                        n = self.nsteps - self.last_retrain_ninputs
                        self.model.partial_fit(self.known_candidates[:-n], self.known_evaluations[:-n])
                    self.last_retrain_ninputs = len(self.known_candidates)

                    # ----------------------------------------------------------------------------------------------------------------------------
                    # Make predictions for all points
                    # ----------------------------------------------------------------------------------------------------------------------------

                    if self.model_name == "CBQ":
                        res = self.model.predict(self.search_space.reshape(-1, 1))
                        y_pred = res[:, 1]
                        y_std = np.abs(res[:, 0] - res[:, 2]) + SMALL_VALUE
                    elif self.model_name == "CB":
                        res = self.model.predict(self.search_space.reshape(-1, 1))
                        y_pred = res
                        y_std = np.zeros_like(res)
                    else:
                        y_pred, y_std = self.model.predict(self.search_space.reshape(-1, 1), return_std=True)

                    self.y_pred = y_pred
                    self.y_std = y_std
                else:
                    y_pred = self.y_pred
                    y_std = self.y_std

                if random() < self.exploitation_probability:
                    self.mode = "exploitation"
                else:
                    self.mode = "exploration"

                # ----------------------------------------------------------------------------------------------------------------------------
                # Known points make it easy to compute distances from candidates
                # ----------------------------------------------------------------------------------------------------------------------------

                distances = compute_candidates_exploration_scores(search_space=self.search_space, known_candidates=self.known_candidates)

                # ----------------------------------------------------------------------------------------------------------------------------
                # Now, distances have to be normalized by known fitness range
                # ----------------------------------------------------------------------------------------------------------------------------

                max_dist = distances.max()
                distances = distances * np.abs(self.best_evaluation - self.worst_evaluation) * self.dist_scaling_coefficient / max_dist

                if self.mode == "exploration":

                    # pick the point with higest std and most distant from already known points
                    expected_fitness = y_std + distances
                    self.additional_info = ""

                elif self.mode == "exploitation":

                    expected_fitness = y_pred.copy()
                    if self.direction == OptimizationDirection.Minimize:
                        expected_fitness = -expected_fitness

                    if self.use_stds_for_exploitation:
                        expected_fitness += y_std

                    best_idx = np.argsort(expected_fitness)[-1]
                    if self.search_space[best_idx] in self.known_candidates:

                        # best supposed point already checked. let's take std and distance into account then.
                        expected_fitness += distances
                        self.additional_info = "plusdist"

                    else:

                        # best supposed point not checked yet
                        self.additional_info = "bestpredicted"

                        if self.use_distances_on_preds_collision:

                            # what if there are multiple non-checked points with the same predicted score?

                            best_value = expected_fitness[best_idx]
                            all_best_indices = np.where(expected_fitness == best_value)[0]
                            all_best_indices = [idx for idx in all_best_indices if self.search_space[idx] not in self.known_candidates]
                            if len(all_best_indices) > 1:
                                expected_fitness[all_best_indices] += distances[all_best_indices]

                    self.expected_fitness = expected_fitness

                # ----------------------------------------------------------------------------------------------------------------------------
                # Decide on the next candidate, based on predicted fitness
                # ----------------------------------------------------------------------------------------------------------------------------

                for _ in range(100):
                    # Just pick first unchecked and unsuggested candidate with the highest fitness
                    for best_idx in np.argsort(expected_fitness)[::-1]:
                        next_candidate = self.search_space[best_idx]
                        if next_candidate not in self.known_candidates and next_candidate not in self.suggested_candidates:
                            if self.skip_best_candidate_prob > 0.0:
                                # Randomly skip the best candidate, if required
                                if random() < self.skip_best_candidate_prob:
                                    continue
                            self.suggested_candidates[next_candidate] = eval_start_time
                            return next_candidate

    def submit_evaluations(self, candidates: Sequence, evaluations: Sequence, durations: Sequence):

        # if duration_seconds is None, it's computed automatically using timestamp of suggesting that particular candidate to that particular worker

        for next_candidate, next_evaluation, next_duration in zip(candidates, evaluations, durations):

            self.nsteps += 1

            if self.direction == OptimizationDirection.Maximize:
                if next_evaluation > self.best_evaluation:
                    self.best_evaluation = next_evaluation
                    self.best_candidate = next_candidate
                    self.n_noimproving_iters = 0
                else:
                    self.n_noimproving_iters += 1
                if next_evaluation < self.worst_evaluation:
                    self.worst_evaluation = next_evaluation
                    self.worst_candidate = next_candidate
            elif self.direction == OptimizationDirection.Minimize:
                if next_evaluation < self.best_evaluation:
                    self.best_evaluation = next_evaluation
                    self.best_candidate = next_candidate
                    self.n_noimproving_iters = 0
                else:
                    self.n_noimproving_iters += 1
                if next_evaluation > self.worst_evaluation:
                    self.worst_evaluation = next_evaluation
                    self.worst_candidate = next_candidate

            if self.verbose and self.n_noimproving_iters == 0:
                logger.info(f"Next optimum found at point ({self.best_candidate}): {self.best_evaluation:_.6f}")

            self.known_candidates = np.append(self.known_candidates, [next_candidate]).astype(int)
            self.known_evaluations = np.append(self.known_evaluations, next_evaluation)

            if next_candidate in self.pre_seeded_candidates:
                self.pre_seeded_candidates.remove(next_candidate)

            start_ts = self.suggested_candidates.get(next_candidate)
            if start_ts:
                try:
                    del self.suggested_candidates[next_candidate]
                except Exception as e:
                    pass

            if next_duration is None and start_ts:
                next_duration = timer() - start_ts
            self.evaluated_candidates.append(dict(candidate=next_candidate, evaluation=next_evaluation, duration=next_duration))

            if (
                self.n_noimproving_iters == 0 and self.plotting == OptimizationProgressPlotting.OnScoreImprovement
            ) or self.plotting == OptimizationProgressPlotting.Regular:
                plot_search_state(
                    search_space=self.search_space,
                    next_cand=next_candidate,
                    new_y=next_evaluation,
                    best_candidate=self.best_candidate,
                    best_evaluation=self.best_evaluation,
                    nsteps=self.nsteps,
                    expected_fitness=self.expected_fitness,
                    y_pred=self.y_pred,
                    y_std=self.y_std,
                    ground_truth=self.ground_truth,
                    known_candidates=self.known_candidates,
                    known_evaluations=self.known_evaluations,
                    acquisition_method=self.acquisition_method,
                    mode=self.mode,
                    additional_info=self.additional_info,
                    figsize=self.figsize,
                    font_size=self.font_size,
                    x_label=self.x_label,
                    y_label=self.y_label,
                    expected_fitness_color=self.expected_fitness_color,
                    legend_location=self.legend_location,
                    skip_candidates=[0],
                )


def optimize_finite_onedimensional_search_space(
    search_space: Sequence,  # search space, all possible input combinations to check
    eval_candidate_func: object,  # fitness function to be optimized over the search space
    ground_truth: np.ndarray = None,  # known true fitness of the entire search space
    direction: OptimizationDirection = OptimizationDirection.Maximize,
    known_candidates: list = [],
    known_evaluations: list = [],
    # stopping conditions
    max_runtime_mins: float = None,
    predict_runtimes: bool = False,  # intellectual setting that skips candidates whose evaluation won't finish within current runtime limit
    max_fevals: int = None,
    best_desired_score: float = None,
    max_noimproving_iters: int = None,
    # inits
    seeded_inputs: Sequence = [],  # seed items you want to be explored from start
    init_num_samples: Union[float, int] = 5,  # how many samples to generate & evaluate before fitting surrogate self.model
    init_evaluate_ascending: bool = False,
    init_evaluate_descending: bool = False,
    init_sampling_method: CandidateSamplingMethod = CandidateSamplingMethod.Equidistant,  # random, equidistant, fibo, rev_fibo?
    # EE dilemma
    exploitation_probability: float = 0.8,
    skip_best_candidate_prob: float = 0.0,  # pick the absolute best predicted candidate, or probabilistically the best
    use_distances_on_preds_collision: bool = True,
    use_stds_for_exploitation: bool = True,
    dist_scaling_coefficient: float = 0.5,
    # self.model
    acquisition_method: str = "EE",
    model_name: str = "CBQ",  # actual estimator instance here? + for lgbm also the linear mode flag
    model_params: dict = {"iterations": 150},
    quantile: float = 0.01,
    input_dtype=np.int32,
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
) -> None:
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
        search_space=search_space,
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
        input_dtype=input_dtype,
        plotting=plotting,
        plotting_ndim=plotting_ndim,
        figsize=figsize,
        font_size=font_size,
        x_label=x_label,
        y_label=y_label,
        expected_fitness_color=expected_fitness_color,
        legend_location=legend_location,
        verbose=verbose,
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
        if next_candidate is None:
            if verbose:
                logger.info("Search space fully checked, quitting")
            break

        next_evaluation = eval_candidate_func(next_candidate)
        optimizer.submit_evaluations(candidates=[next_candidate], evaluations=[next_evaluation], durations=[None])

        # ----------------------------------------------------------------------------------------------------------------------------
        # Checking exit conditions
        # ----------------------------------------------------------------------------------------------------------------------------

        if best_desired_score:
            if direction == OptimizationDirection.Maximize:
                if optimizer.best_evaluation >= best_desired_score:
                    if verbose:
                        logger.info(f"best_desired_score={optimizer.best_evaluation:_.6f} reached.")
                    break
            elif direction == OptimizationDirection.Maximize:
                if optimizer.best_evaluation <= best_desired_score:
                    if verbose:
                        logger.info(f"best_desired_score={optimizer.best_evaluation:_.6f} reached.")
                    break

        if max_runtime_mins and not ran_out_of_time:
            ran_out_of_time = (timer() - start_time) > max_runtime_mins * 60
            if ran_out_of_time:
                if verbose:
                    logger.info(f"max_runtime_mins={max_runtime_mins:_.1f} reached.")
                break

        if max_fevals and optimizer.nsteps >= max_fevals:
            if verbose:
                logger.info(f"max_fevals={max_fevals:_} reached.")
            break

        if max_noimproving_iters and optimizer.n_noimproving_iters >= max_noimproving_iters:
            if verbose:
                logger.info(f"Max # of noimproved iters reached: {optimizer.n_noimproving_iters}")
            break

    return (optimizer.best_candidate, optimizer.best_evaluation), optimizer.evaluated_candidates


def plot_search_state(
    search_space,
    next_cand: int,
    new_y: float,
    best_candidate: int,
    best_evaluation: float,
    nsteps: int,
    expected_fitness: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    ground_truth: np.ndarray,
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
    plt.show()
