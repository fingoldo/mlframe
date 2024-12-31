"""Feature selection within ML pipelines. Wrappers methods. Currently includes recursive feature elimination."""

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

        from os.path import exists
        import pandas as pd, numpy as np

        from pyutilz.system import tqdmu
        from pyutilz.numbalib import set_numba_random_seed
        from pyutilz.pythonlib import store_params_in_object, get_parent_func_args

        from mlframe.config import *
        from mlframe.optimization import *
        from mlframe.votenrank import Leaderboard
        from mlframe.utils import set_random_seed
        from mlframe.baselines import get_best_dummy_score
        from mlframe.helpers import has_early_stopping_support
        from mlframe.preprocessing import pack_val_set_into_fit_params
        from mlframe.metrics import compute_probabilistic_multiclass_error

        from sklearn.pipeline import Pipeline
        from sklearn.dummy import DummyClassifier, DummyRegressor
        from sklearn.metrics import make_scorer, mean_squared_error
        from sklearn.base import is_classifier, is_regressor, BaseEstimator, TransformerMixin
        from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit, GroupKFold, GroupShuffleSplit, KFold

        from enum import Enum, auto
        from timeit import default_timer as timer

        import matplotlib.pyplot as plt

        import random
        import copy

    except ModuleNotFoundError as e:

        logger.warning(e)

        if "cannot import name" in str(e):
            raise (e)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Packages auto-install
        # ----------------------------------------------------------------------------------------------------------------------------

        from pyutilz.pythonlib import ensure_installed

        ensure_installed("numpy pandas scikit-learn")  # cupy

    else:
        break

# ----------------------------------------------------------------------------------------------------------------------------
# Inits
# ----------------------------------------------------------------------------------------------------------------------------


class OptimumSearch(str, Enum):
    ScipyLocal = "ScipyLocal"  # Brent
    ScipyGlobal = "ScipyGlobal"  # direct, diff evol, shgo
    ModelBasedHeuristic = "ModelBasedHeuristic"  # GaussianProcess or Catboost with uncertainty, or quantile regression
    ExhaustiveRandom = "ExhaustiveRandom"
    ExhaustiveDichotomic = "ExhaustiveDichotomic"


class VotesAggregation(str, Enum):
    Minimax = "Minimax"
    OG = "OG"
    Borda = "Borda"
    Plurality = "Plurality"
    Dowdall = "Dowdall"
    Copeland = "Copeland"
    AM = "AM"
    GM = "GM"


class RFECV(BaseEstimator, TransformerMixin):
    """Finds subset of features having best CV score, by iterative narrowing down set of top_n candidates having highest importance, as per estimator's FI scores.

    Optimizes mean CV scores (possibly accounting for variation, possibly translated into ranks) divided by the features number.

    Uses several optimization methods:
        exhaustive search
        random search
        model-based heuristic search.

    Problems:
        Impactful, but correlated factors all get low importance and will be thrown away (probably only for forests, not boostings?).
        confirmed for boostings also! adding more predictors to original features worsens scores, whereas in theory it at least should not be worse!

        Due to noise some random features can become "important".

    Solution:
        use CV to calculate fold FI, then combine across folds (by voting).
        When estimating featureset quality at another TopN, use different splits & combine new FIs with all known before, to mitigate noise even more.

    Optionally plots (and saves) the optimization path - checked nfeatures and corresponding scores.
    If surrogate models are used, also shows predicted scores along with confidence bounds.

    Challenges:
        CV performance itself can be a multi-component value! Say, both ROC AUC and CALIB metrics can be considered. Voting can be a solution.
        Estimator might itself be a HPT search instance. Or a pipeline.
        It could be good to have several estimators. Their importance evaluations must be accounted for simultaneously (voting).
        Estimator might need eval_set or similar (eval_frac).
        Different folds invocations could benefit from generating all possible hyper parameters. Even if FS does not care, collected info could be used further at the HPT step.

    Parameters
    ----------
        cv : int, cross-validation generator or an iterable, default=None

    Attributes
    ----------

    estimator_ : ``Estimator`` instance
        The fitted estimator used to select features.

    cv_results_ : dict of ndarrays
        A dict with keys:

        split(k)_test_score : ndarray of shape (n_subsets_of_features,)
            The cross-validation scores across (k)th fold.

        mean_test_score : ndarray of shape (n_subsets_of_features,)
            Mean of scores over the folds.

        std_test_score : ndarray of shape (n_subsets_of_features,)
            Standard deviation of scores over the folds.


    n_features_ : int
        The number of selected features with cross-validation.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    ranking_ ?: narray of shape (n_features,)
        The feature ranking, such that `ranking_[i]`
        corresponds to the ranking
        position of the i-th feature.
        Selected (i.e., estimated best)
        features are assigned rank 1.

    support_ : ndarray of shape (n_features,)
        The mask of selected features.

    """

    def __init__(
        self,
        estimator: BaseEstimator,
        fit_params: dict = {},
        max_nfeatures: int = None,
        mean_perf_weight: float = 1.0,
        std_perf_weight: float = 1.0,
        feature_cost: float = 0.00 / 100,
        smooth_perf: int = 0,
        # stopping conditions
        max_runtime_mins: float = None,
        max_refits: int = None,
        best_desired_score: float = None,
        max_noimproving_iters: int = 30,
        # CV
        cv: Union[object, int, None] = 3,
        cv_shuffle: bool = False,
        # Other
        early_stopping_val_nsplits: Union[int, None] = 4,
        early_stopping_rounds: Union[int, None] = None,
        scoring: Union[object, None] = None,
        nofeatures_dummy_scoring: bool = False,
        top_predictors_search_method: OptimumSearch = OptimumSearch.ModelBasedHeuristic,
        votes_aggregation_method: VotesAggregation = VotesAggregation.Borda,
        use_all_fi_runs: bool = True,
        use_last_fi_run_only: bool = False,
        use_one_freshest_fi_run: bool = False,
        use_fi_ranking: bool = False,
        importance_getter: Union[str, Callable, None] = None,
        random_state: int = None,
        leave_progressbars: bool = True,
        verbose: Union[bool, int] = 0,
        show_plot: bool = False,
        cat_features: Union[Sequence, None] = None,
        keep_estimators: bool = False,
        estimators_save_path: str = None,  # fitted estimators get saved into join(estimators_save_path,estimator_type_name,nestimator_nfeatures_nfold.dump)
        # Required features and achieved ml metrics get saved in a dict join(estimators_save_path,required_features.dump).
        frac: float = None,
        skip_retraining_on_same_shape: bool = False,
        stop_file: str = "stop",
    ):

        # checks
        if frac is not None:
            assert frac > 0.0 and frac < 1.0

        # assert isinstance(estimator, (BaseEstimator,))

        # save params

        params = get_parent_func_args()
        store_params_in_object(obj=self, params=params)
        self.signature = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, pd.Series, np.ndarray], groups: Union[pd.Series, np.ndarray] = None, **fit_params):

        # ----------------------------------------------------------------------------------------------------------------------------
        # Compute inputs/outputs signature
        # ----------------------------------------------------------------------------------------------------------------------------

        signature = (X.shape, y.shape)
        if self.skip_retraining_on_same_shape:
            if signature == self.signature:
                if self.verbose:
                    logger.info(f"Skipping retraining on the same inputs signature {signature}")
                return self

        # ---------------------------------------------------------------------------------------------------------------
        # Inits
        # ---------------------------------------------------------------------------------------------------------------

        estimator = self.estimator
        fit_params = copy.copy(self.fit_params)
        max_runtime_mins = self.max_runtime_mins
        max_refits = self.max_refits
        cv = self.cv
        cv_shuffle = self.cv_shuffle
        early_stopping_val_nsplits = self.early_stopping_val_nsplits
        early_stopping_rounds = self.early_stopping_rounds
        scoring = self.scoring
        top_predictors_search_method = self.top_predictors_search_method
        votes_aggregation_method = self.votes_aggregation_method
        use_all_fi_runs = self.use_all_fi_runs
        use_last_fi_run_only = self.use_last_fi_run_only
        use_one_freshest_fi_run = self.use_one_freshest_fi_run
        use_fi_ranking = self.use_fi_ranking
        importance_getter = self.importance_getter
        random_state = self.random_state
        leave_progressbars = self.leave_progressbars
        verbose = self.verbose
        show_plot = self.show_plot
        cat_features = self.cat_features
        keep_estimators = self.keep_estimators
        feature_cost = self.feature_cost
        smooth_perf = self.smooth_perf
        frac = self.frac
        best_desired_score = self.best_desired_score
        max_noimproving_iters = self.max_noimproving_iters

        start_time = timer()
        ran_out_of_time = False

        if random_state is not None:
            set_random_seed(random_state)

        feature_importances = {}
        evaluated_scores_std = {}
        evaluated_scores_mean = {}

        if isinstance(X, pd.DataFrame):
            original_features = X.columns.tolist()
        else:
            original_features = np.arange(X.shape[1])

        # ----------------------------------------------------------------------------------------------------------------------------
        # Init cv
        # ----------------------------------------------------------------------------------------------------------------------------

        if cv is None or str(cv).isnumeric():
            if cv is None:
                cv = 3
            if is_classifier(estimator):
                if groups is not None:
                    cv = StratifiedGroupKFold(n_splits=cv, shuffle=cv_shuffle, random_state=random_state)
                else:
                    cv = StratifiedKFold(n_splits=cv, shuffle=cv_shuffle, random_state=random_state)
            else:
                if groups is not None:
                    cv = GroupKFold(n_splits=cv, shuffle=cv_shuffle, random_state=random_state)
                else:
                    cv = KFold(n_splits=cv, shuffle=cv_shuffle, random_state=random_state)
            if verbose:
                logger.info(f"Using cv={cv}")

        if early_stopping_val_nsplits:
            val_cv = copy.copy(cv)
            val_cv.n_splits = early_stopping_val_nsplits
            if not early_stopping_rounds:
                early_stopping_rounds = 20  # TODO: derive as 1/5 of nestimators'
        else:
            val_cv = None

        if verbose:
            iters_pbar = tqdmu(
                desc="RFECV iterations",
                leave=leave_progressbars,
                total=min(len(original_features) + 1, max_refits) if max_refits else len(original_features) + 1,
            )

        # ----------------------------------------------------------------------------------------------------------------------------
        # Init scoring
        # ----------------------------------------------------------------------------------------------------------------------------

        if scoring is None:
            if is_classifier(estimator):
                logger.info(f"Scoring omited, using probabilistic_multiclass_error by default.")
                scoring = make_scorer(score_func=compute_probabilistic_multiclass_error, needs_proba=True, needs_threshold=False, greater_is_better=False)
            elif is_regressor(estimator):
                logger.info(f"Scoring omited, using mean_squared_error by default.")
                scoring = make_scorer(score_func=mean_squared_error, needs_proba=False, needs_threshold=False, greater_is_better=False)
            else:
                raise ValueError(f"Appropriate scoring not known for estimator type: {estimator}")

        # ----------------------------------------------------------------------------------------------------------------------------
        # Init importance_getter
        # ----------------------------------------------------------------------------------------------------------------------------

        if isinstance(estimator, Pipeline):
            estimator_type = type(estimator.steps[-1][1]).__name__
        else:
            estimator_type = type(estimator).__name__

        if importance_getter is None or importance_getter == "auto":
            if estimator_type in ("LogisticRegression",):
                importance_getter = "coef_"
            else:
                importance_getter = "feature_importances_"

        # ----------------------------------------------------------------------------------------------------------------------------
        # Start evaluating different nfeatures, being guided by the selected search method
        # ----------------------------------------------------------------------------------------------------------------------------

        nsteps = 0
        dummy_scores = []
        fitted_estimators = {}
        selected_features_per_nfeatures = {}

        if top_predictors_search_method == OptimumSearch.ModelBasedHeuristic:
            Optimizer = MBHOptimizer(
                search_space=(
                    np.array(np.arange(min(self.max_nfeatures, len(original_features)) + 1).tolist() + [len(original_features)])
                    if self.max_nfeatures
                    else np.arange(len(original_features) + 1)
                ),
                direction=OptimizationDirection.Maximize,
                init_sampling_method=CandidateSamplingMethod.Equidistant,
                init_evaluate_ascending=False,
                init_evaluate_descending=True,
                plotting=OptimizationProgressPlotting.OnScoreImprovement,
                seeded_inputs=[min(2, len(original_features))],
            )
        else:
            Optimizer = None

        n_noimproving_iters = 0
        best_score = -1e6

        while nsteps < len(original_features):

            if verbose:
                iters_pbar.update(1)

            # ----------------------------------------------------------------------------------------------------------------------------
            # Select current set of features to work on, based on ranking received so far, and the optimum search method
            # ----------------------------------------------------------------------------------------------------------------------------

            current_features = get_next_features_subset(
                nsteps=nsteps,
                original_features=original_features,
                feature_importances=feature_importances,
                evaluated_scores_mean=evaluated_scores_mean,
                evaluated_scores_std=evaluated_scores_std,
                use_all_fi_runs=use_all_fi_runs,
                use_last_fi_run_only=use_last_fi_run_only,
                use_one_freshest_fi_run=use_one_freshest_fi_run,
                use_fi_ranking=use_fi_ranking,
                top_predictors_search_method=top_predictors_search_method,
                votes_aggregation_method=votes_aggregation_method,
                Optimizer=Optimizer,
            )

            if current_features is None or len(current_features) == 0:
                break  # nothing more to try
            if self.stop_file and exists(self.stop_file):
                logger.warning(f"Stop file {self.stop_file} detected, quitting.")
                break

            selected_features_per_nfeatures[len(current_features)] = current_features

            # ----------------------------------------------------------------------------------------------------------------------------
            # Each split better be different. so, even if random_state is provided, random_state to the cv is generated separately
            # (and deterministically) each time based on the original random_state.
            # ----------------------------------------------------------------------------------------------------------------------------

            scores = []

            splitter = cv.split(X=X, y=y, groups=groups)
            if verbose:
                splitter = tqdmu(splitter, desc="CV folds", leave=False, total=cv.n_splits)

            # ----------------------------------------------------------------------------------------------------------------------------
            # Evaluate currently selected set of features on CV
            # ----------------------------------------------------------------------------------------------------------------------------

            for nfold, (train_index, test_index) in enumerate(splitter):

                if frac:
                    size = int(len(train_index) * frac)
                    if size > 10:
                        train_index = np.random.choice(train_index, size=size)

                X_train, y_train, X_test, y_test = split_into_train_test(
                    X=X, y=y, train_index=train_index, test_index=test_index, features_indices=current_features
                )  # this splits both dataframes & ndarrays in the same fashion

                if val_cv and has_early_stopping_support(estimator_type):

                    # ----------------------------------------------------------------------------------------------------------------------------
                    # Make additional early stopping split from X_train
                    # ----------------------------------------------------------------------------------------------------------------------------

                    if groups is not None:
                        if isinstance(groups, pd.Series):
                            train_groups = groups.iloc[train_index]
                        else:
                            train_groups = groups[train_index]
                    else:
                        train_groups = None

                    for true_train_index, val_index in val_cv.split(X=X_train, y=y_train, groups=train_groups):
                        break  # need only 1 iteration of 2nd split

                    X_train, y_train, X_val, y_val = split_into_train_test(X=X_train, y=y_train, train_index=true_train_index, test_index=val_index)

                    # ----------------------------------------------------------------------------------------------------------------------------
                    # If estimator is known, apply early stopping to its fit params
                    # ----------------------------------------------------------------------------------------------------------------------------

                    temp_cat_features = [current_features.index(var) for var in cat_features if var in current_features] if cat_features else None

                    temp_fit_params = pack_val_set_into_fit_params(
                        model=estimator,
                        X_val=X_val,
                        y_val=y_val,
                        early_stopping_rounds=early_stopping_rounds,
                        cat_features=temp_cat_features,
                    )  # crafts fit params with early stopping tailored to particular model type.
                    temp_fit_params.update(fit_params)

                else:
                    temp_fit_params = {}
                    X_val = None

                # ----------------------------------------------------------------------------------------------------------------------------
                # Fit our estimator on current train fold. Score on test & and get its feature importances.
                # ----------------------------------------------------------------------------------------------------------------------------

                # TODO! invoke different hyper parameters generation here

                if keep_estimators:
                    fitted_estimator = copy.copy(estimator)
                else:
                    fitted_estimator = estimator

                fitted_estimator.fit(X=X_train, y=y_train, **temp_fit_params)

                score = scoring(fitted_estimator, X_test, y_test)
                scores.append(score)
                fi = get_feature_importances(
                    model=fitted_estimator, current_features=current_features, data=X_test, reference_data=X_val, importance_getter=importance_getter
                )
                # feature_indices,imp_values=list(fi.keys()),list(fi.values())
                # print(np.array(feature_indices)[np.argsort(imp_values)[-10:]])

                key = f"{len(current_features)}_{nfold}"
                feature_importances[key] = fi

                if keep_estimators:
                    fitted_estimators[key] = fitted_estimator

                # print(f"feature_importances[step{len(current_features)}_fold{nfold}]=" + str({key: value for key, value in fi.items() if value > 0}))

                if 0 not in evaluated_scores_mean:

                    # ----------------------------------------------------------------------------------------------------------------------------
                    # Dummy baselines must serve as fitness @ 0 features.
                    # ----------------------------------------------------------------------------------------------------------------------------

                    if not self.nofeatures_dummy_scoring:
                        if scoring._sign == 1:
                            dummy_scores.append(score / 10 if score > 0 else score * 10)
                        else:
                            dummy_scores.append(score * 10 if score > 0 else score / 10)
                    else:
                        dummy_scores.append(
                            get_best_dummy_score(estimator=estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, scoring=scoring)
                        )
                        # print(f"Best dummy score (at 0 features, fold {nfold}): {best_dummy_score}")

            if 0 not in evaluated_scores_mean:
                scores_mean, scores_std = store_averaged_cv_scores(
                    pos=0, scores=dummy_scores, evaluated_scores_mean=evaluated_scores_mean, evaluated_scores_std=evaluated_scores_std
                )
                if verbose:
                    logger.info(f"baseline with nfeatures=0, scores={scores_mean:.6f} ± {scores_std:.6f}")
                if top_predictors_search_method == OptimumSearch.ModelBasedHeuristic:
                    Optimizer.submit_evaluations(candidates=[0], evaluations=[scores_mean - scores_std], durations=[None])

            scores_mean, scores_std = store_averaged_cv_scores(
                pos=len(current_features), scores=scores, evaluated_scores_mean=evaluated_scores_mean, evaluated_scores_std=evaluated_scores_std
            )
            if top_predictors_search_method == OptimumSearch.ModelBasedHeuristic:
                Optimizer.submit_evaluations(candidates=[len(current_features)], evaluations=[scores_mean - scores_std], durations=[None])

                if verbose:
                    logger.info(f"trying nfeatures={len(current_features)}, score={scores_mean:.6f} ± {scores_std:.6f}")

            if len(evaluated_scores_mean) == 2:
                # only 2 cases covered currently: 0 features & all features
                if evaluated_scores_mean[0] - evaluated_scores_std[0] > scores_mean - scores_std:
                    logger.info(
                        f"Stopping RFECV early: performance with no features {evaluated_scores_mean[0] - evaluated_scores_std[0]:.6f} is not worse than with all features {scores_mean - scores_std:.6f}."
                    )
                    break

            # ----------------------------------------------------------------------------------------------------------------------------
            # Checking exit conditions
            # ----------------------------------------------------------------------------------------------------------------------------

            nsteps += 1

            if max_runtime_mins and not ran_out_of_time:
                delta = timer() - start_time
                ran_out_of_time = delta > max_runtime_mins * 60
                if ran_out_of_time:
                    if verbose:
                        logger.info(f"max_runtime_mins={max_runtime_mins:_.1f} reached.")
                    break

            if max_refits and nsteps >= max_refits:
                if verbose:
                    logger.info(f"max_refits={max_refits:_} reached.")
                break

            if scores_mean >= best_score:
                best_score = scores_mean
                n_noimproving_iters = 0
            else:
                n_noimproving_iters += 1

            if best_desired_score is not None and scores_mean >= best_desired_score:
                if verbose:
                    logger.info(f"best_desired_score {best_desired_score:_.6f} reached.")
                break

                if best_desired_score is not None and scores_mean <= best_desired_score:
                    if verbose:
                        logger.info(f"best_desired_score {best_desired_score:_.6f} reached.")
                    break

            if max_noimproving_iters and n_noimproving_iters >= max_noimproving_iters:
                if verbose:
                    logger.info(f"Max # of noimproved iters reached: {n_noimproving_iters}")
                break

        # ----------------------------------------------------------------------------------------------------------------------------
        # Saving best result found so far as final
        # ----------------------------------------------------------------------------------------------------------------------------

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else list(map(str, np.arange(self.n_features_in_)))

        self.estimators_ = fitted_estimators  # a dict with key=nfeatures_nfold
        self.feature_importances_ = feature_importances  # a dict with key=nfeatures_nfold
        self.selected_features_ = selected_features_per_nfeatures  # a dict with key=nfeatures

        checked_nfeatures = sorted(evaluated_scores_mean.keys())
        cv_std_perf = [evaluated_scores_std[n] for n in checked_nfeatures]
        cv_mean_perf = [evaluated_scores_mean[n] for n in checked_nfeatures]
        self.cv_results_ = {"nfeatures": checked_nfeatures, "cv_mean_perf": cv_mean_perf, "cv_std_perf": cv_std_perf}

        self.select_optimal_nfeatures_(
            checked_nfeatures=checked_nfeatures,
            cv_mean_perf=cv_mean_perf,
            cv_std_perf=cv_std_perf,
            mean_perf_weight=self.mean_perf_weight,
            std_perf_weight=self.std_perf_weight,
            feature_cost=feature_cost,
            smooth_perf=smooth_perf,
            use_all_fi_runs=use_all_fi_runs,
            use_last_fi_run_only=use_last_fi_run_only,
            use_one_freshest_fi_run=use_one_freshest_fi_run,
            use_fi_ranking=use_fi_ranking,
            votes_aggregation_method=votes_aggregation_method,
            verbose=verbose,
            show_plot=show_plot,
        )

        self.signature = signature
        return self

    def select_optimal_nfeatures_(
        self,
        checked_nfeatures: np.ndarray,
        cv_mean_perf: np.ndarray,
        cv_std_perf: np.ndarray,
        mean_perf_weight: float = 1.0,
        std_perf_weight: float = 1.0,
        feature_cost: float = 0.00 / 100,
        smooth_perf: int = 0,
        use_all_fi_runs: bool = True,
        use_last_fi_run_only: bool = False,
        use_one_freshest_fi_run: bool = False,
        use_fi_ranking: bool = False,
        votes_aggregation_method: VotesAggregation = VotesAggregation.Borda,
        verbose: bool = False,
        show_plot: bool = False,
        plot_file=None,
        font_size: int = 12,
        figsize: tuple = (10, 7),
    ):

        base_perf = np.array(cv_mean_perf) * mean_perf_weight - np.array(cv_std_perf) * std_perf_weight
        if smooth_perf:
            smoothed_perf = pd.Series(base_perf).rolling(smooth_perf, center=True).mean().values
            idx = np.isnan(smoothed_perf)
            smoothed_perf[idx] = base_perf[idx]
            base_perf = smoothed_perf

        # ultimate_perf = base_perf / (np.log1p(np.arange(len(base_perf))) + comparison_base)
        ultimate_perf = base_perf - np.arange(len(base_perf)) * feature_cost

        best_idx = np.argmax(ultimate_perf)
        best_top_n = checked_nfeatures[best_idx]

        if show_plot or plot_file:
            plt.rcParams.update({"font.size": font_size})
            fig, ax1 = plt.subplots(figsize=figsize)
            ax2 = ax1.twinx()

            ax1.set_xlabel("Number of features selected")
            ax1.set_ylabel("Mean CV score", c="b")

            ax1.errorbar(checked_nfeatures, cv_mean_perf, yerr=cv_std_perf, c="b", alpha=0.4)

            ax2.plot(checked_nfeatures, ultimate_perf, c="g")
            ax1.plot(checked_nfeatures[best_idx], base_perf[best_idx], "ro")
            ax2.set_ylabel("Adj CV score", c="g")

            plt.title("Performance by nfeatures")
            plt.tight_layout()

            if plot_file:
                plt.savefig(plot_file)
            if show_plot:
                plt.show()

        # after making a cutoff decision:

        self.n_features_ = best_top_n
        if best_top_n == 0:
            self.support_ = np.array([])
        else:

            if True:

                # An obvious solution is to return exact features that we used when measuring scores.
                self.support_ = np.array([self.feature_names_in_.index(feature_name) for feature_name in self.selected_features_[best_top_n]])

            else:

                # ----------------------------------------------------------------------------------------------------------------------------
                # A more advanced alternative would be to last time vote for feature_importances using all info up to date
                # ----------------------------------------------------------------------------------------------------------------------------

                fi_to_consider = select_appropriate_feature_importances(
                    feature_importances=self.feature_importances_,
                    nfeatures=best_top_n,
                    n_original_features=self.n_features_in_,
                    use_all_fi_runs=use_all_fi_runs,
                    use_last_fi_run_only=use_last_fi_run_only,
                    use_one_freshest_fi_run=use_one_freshest_fi_run,
                    use_fi_ranking=use_fi_ranking,
                )

                self.ranking_ = get_actual_features_ranking(
                    feature_importances=fi_to_consider,
                    votes_aggregation_method=votes_aggregation_method,
                )

                self.support_ = np.array([(i in self.ranking_[:best_top_n]) for i in self.feature_names_in_])

        if verbose:
            dummy_gain = base_perf[0] / base_perf[best_idx] - 1
            allfeat_gain = base_perf[-1] / base_perf[best_idx] - 1
            logger.info(
                f"{self.n_features_:_} predictive factors selected out of {self.n_features_in_:_} during {len(self.selected_features_):_} rounds. Gain vs dummy={dummy_gain*100:.1f}%, gain vs all features={allfeat_gain*100:.1f}%"
            )

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.support_]
        else:
            return X[:, self.support_]


def split_into_train_test(
    X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray], train_index: np.ndarray, test_index: np.ndarray, features_indices: np.ndarray = None
) -> tuple:
    """Split X & y according to indices & dtypes. Basically this accounts for diffeent dtypes (pd.DataFrame, np.ndarray) to perform the same."""

    if isinstance(X, pd.DataFrame):
        X_train, y_train = (X.iloc[train_index, :] if features_indices is None else X.iloc[train_index, :][features_indices]), (
            y.iloc[train_index, :] if isinstance(y, pd.DataFrame) else (y.iloc[train_index] if isinstance(y, pd.Series) else y[train_index])
        )
        X_test, y_test = (X.iloc[test_index, :] if features_indices is None else X.iloc[test_index, :][features_indices]), (
            y.iloc[test_index, :] if isinstance(y, pd.DataFrame) else (y.iloc[test_index] if isinstance(y, pd.Series) else y[test_index])
        )
    else:
        X_train, y_train = (X[train_index, :] if features_indices is None else X[train_index, :][:, features_indices]), (
            y[train_index, :] if len(y.shape) > 1 else y[train_index]
        )
        X_test, y_test = (X[test_index, :] if features_indices is None else X[test_index, :][:, features_indices]), (
            y[test_index, :] if len(y.shape) > 1 else y[test_index]
        )

    return X_train, y_train, X_test, y_test


def store_averaged_cv_scores(pos: int, scores: list, evaluated_scores_mean: dict, evaluated_scores_std: dict) -> None:

    scores = np.array(scores)
    scores_mean, scores_std = np.median(scores), np.std(scores)

    evaluated_scores_mean[pos] = scores_mean
    evaluated_scores_std[pos] = scores_std

    return scores_mean, scores_std


def get_feature_importances(
    model: object,
    current_features: list,
    importance_getter: Union[str, Callable],
    data: Union[pd.DataFrame, np.ndarray, None] = None,
    reference_data: Union[pd.DataFrame, np.ndarray, None] = None,
) -> dict:

    if isinstance(importance_getter, str):
        res = getattr(model, importance_getter)
        if importance_getter == "coef_":
            res = np.abs(res)
        if res.ndim > 1:
            res = res.sum(axis=0)
    else:
        res = importance_getter(model=model, data=data, reference_data=reference_data)

    assert len(res) == len(current_features)
    return {feature_index: feature_importance for feature_index, feature_importance in zip(current_features, res)}


def get_next_features_subset(
    nsteps: int,
    original_features: list,
    feature_importances: pd.DataFrame,
    evaluated_scores_mean: dict,
    evaluated_scores_std: dict,
    use_all_fi_runs: bool,
    use_last_fi_run_only: bool,
    use_one_freshest_fi_run: bool,
    use_fi_ranking: bool,
    top_predictors_search_method: OptimumSearch = OptimumSearch.ScipyLocal,
    votes_aggregation_method: VotesAggregation = VotesAggregation.Borda,
    Optimizer: object = None,
) -> list:
    """Generates a "next_nfeatures_to_check" candidate to evaluate.
    Decides on a subset of FIs to use (all, freshest preceeding, all preceeding).
    Combines FIs from different runs into ranks using voting.
    Selects next_nfeatures_to_check best ranked features as candidates for the upcoming FI evaluation.
    The whole idea of this approach is that we don't need to go all the way from len(original_features) up to 0 and evaluate
    EVERY nfeatures. for 10k features and 1TB datast it's a waste.
    """

    # ----------------------------------------------------------------------------------------------------------------------------
    # First step is to try all features.
    # ----------------------------------------------------------------------------------------------------------------------------

    if nsteps == 0:
        return original_features
    else:
        remaining = list(set(np.arange(1, len(original_features))) - set(evaluated_scores_mean.keys()))
        if len(remaining) == 0:
            return []
        else:

            if top_predictors_search_method == OptimumSearch.ExhaustiveRandom:
                next_nfeatures_to_check = random.choice(remaining)
            elif top_predictors_search_method == OptimumSearch.ModelBasedHeuristic:
                next_nfeatures_to_check = Optimizer.suggest_candidate()

            if next_nfeatures_to_check is not None:

                # ----------------------------- -----------------------------------------------------------------------------------------------
                # At each step, feature importances must be recalculated in light of recent training on a smaller subset.
                # The features already thrown away all receive constant importance update of the same size, to keep up with number of trains (?)
                # ----------------------------------------------------------------------------------------------------------------------------

                fi_to_consider = select_appropriate_feature_importances(
                    feature_importances=feature_importances,
                    nfeatures=next_nfeatures_to_check,
                    n_original_features=len(original_features),
                    use_all_fi_runs=use_all_fi_runs,
                    use_last_fi_run_only=use_last_fi_run_only,
                    use_one_freshest_fi_run=use_one_freshest_fi_run,
                    use_fi_ranking=use_fi_ranking,
                )
                ranks = get_actual_features_ranking(feature_importances=fi_to_consider, votes_aggregation_method=votes_aggregation_method)
                # print(f"fi_to_consider={fi_to_consider}")
                # print(f"ranks={ranks}")
                # print(f"next_nfeatures_to_check={next_nfeatures_to_check}, features chosen={ranks[:next_nfeatures_to_check]}")

                return ranks[:next_nfeatures_to_check]
            else:
                return []


def select_appropriate_feature_importances(
    feature_importances: dict,
    nfeatures: int,
    n_original_features: int,
    use_all_fi_runs: bool = True,
    use_last_fi_run_only: bool = False,
    use_one_freshest_fi_run: bool = False,
    use_fi_ranking: bool = False,
) -> dict:

    if use_last_fi_run_only:
        # use train folds with specific length. key is nfeatures_nfold
        fi_to_consider = {key: value for key, value in feature_importances.items() if len(value) == n_original_features}
    else:
        if use_all_fi_runs:
            # use all fi data collected so far
            fi_to_consider = {key: value for key, value in feature_importances.items() if len(value) > 1} if n_original_features > 1 else feature_importances
        else:
            # can only use runs preceeding nfeatures here.
            if use_one_freshest_fi_run:
                # freshest preceeding
                fi_to_consider = {}
                for possible_nfeatures in range(nfeatures + 1, n_original_features):
                    for key, value in feature_importances.items():
                        if len(value) == possible_nfeatures:

                            fi_to_consider[key] = value
                    if fi_to_consider:
                        print(f"using freshest FI of {possible_nfeatures} features for nfeatures={nfeatures}")
                        break
            else:
                # all preceeding
                fi_to_consider = {key: value for key, value in feature_importances.items() if (len(value) > nfeatures and len(value) != 1)}
    if use_fi_ranking:
        fi_to_consider = {key: pd.Series(value).rank(ascending=True, pct=True).to_dict() for key, value in fi_to_consider.items()}
    return fi_to_consider


def get_actual_features_ranking(feature_importances: dict, votes_aggregation_method: VotesAggregation) -> list:
    """Absolute FIs from estimators trained on CV for each nfeatures are stored separatly.
    They can be used to recompute final voted importances using any desired voting algo.
    But of course the exploration path was already lead by specific voting algo active at the fitting time.

    GM, and esp Minimax & Plurality are suboptimal for FS.
    """

    lb = Leaderboard(table=pd.DataFrame(feature_importances))
    if votes_aggregation_method == VotesAggregation.Borda:
        ranks = lb.borda_ranking()
    elif votes_aggregation_method == VotesAggregation.AM:
        ranks = lb.mean_ranking(mean_type="arithmetic")
    elif votes_aggregation_method == VotesAggregation.GM:
        ranks = lb.mean_ranking(mean_type="geometric")
    elif votes_aggregation_method == VotesAggregation.Copeland:
        ranks = lb.copeland_ranking()
    elif votes_aggregation_method == VotesAggregation.Dowdall:
        ranks = lb.dowdall_ranking()
    elif votes_aggregation_method == VotesAggregation.Minimax:
        ranks = lb.minimax_ranking()
    elif votes_aggregation_method == VotesAggregation.OG:
        ranks = lb.optimality_gap_ranking(gamma=1)
    elif votes_aggregation_method == VotesAggregation.Plurality:
        ranks = lb.plurality_ranking()

    # print("Current features ranks:")
    # print(ranks)
    return ranks.index.values.tolist()
