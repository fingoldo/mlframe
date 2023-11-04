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

        import pandas as pd, numpy as np
        import cupy as cp

        from pyutilz.system import tqdmu
        from pyutilz.numbalib import set_random_seed
        from pyutilz.pythonlib import store_params_in_object, get_parent_func_args

        from mlframe.config import *
        from mlframe.metrics import calib_error
        from mlframe.preprocessing import pack_val_set_into_fit_params

        from sklearn.dummy import DummyClassifier, DummyRegressor
        from sklearn.metrics import make_scorer, mean_squared_error
        from sklearn.base import is_classifier, is_regressor, BaseEstimator, TransformerMixin
        from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit, GroupKFold, GroupShuffleSplit, KFold

        from enum import Enum, auto
        from timeit import default_timer as timer

        import matplotlib.pyplot as plt

        import random
        import copy

        from votenrank import Leaderboard

    except Exception as e:

        logger.warning(e)

        if "cannot import name" in str(e):
            raise (e)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Packages auto-install
        # ----------------------------------------------------------------------------------------------------------------------------

        from pyutilz.pythonlib import ensure_installed

        ensure_installed("numpy pandas cupy scikit-learn")

    else:
        break

# ----------------------------------------------------------------------------------------------------------------------------
# Inits
# ----------------------------------------------------------------------------------------------------------------------------

LARGE_CONST: float = 1e30


class OptimumSearch(str, Enum):
    ScipyLocal = "ScipyLocal"  # Brent
    ScipyGlobal = "ScipyGlobal"  # direct, diff evol, shgo
    SurrogateModel = "SurrogateModel"  # GaussianProcess or Catboost with uncertainty, or quantile regression
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
        bayesian search.


    A problem:
        impactful, but correlated factors all get low importance and will be thrown away (probably only for forests, not boostings).
    Solution:
        mb try more than one estimator?

    Other problem:
        due to noise some random features can become "important".
    Solution:
        use CV to calculate fold FI, then combine across folds (by voting).
        When estimating featureset quality at another TopN, use different splits & combine new FIs with all known before, to mitigate noise even more.

    Optionally plots (and saves) the optimization path - checked top_n and corresponding scores.
    If surrogate models are used, also shows predicted scores along with confidence bounds.

    Challenges:
        CV performance itself can be a multi-component value! Say, both ROC AUC and CALIB metrics can be considered. Voting can be a solution.
        Estimator might itself be a HPT search instance. Or a pipeline.
        It could be good to have several estimators. Their importance evaluations must be accounted for simultaneously (voting).
        Estimator might need eval_set or similar (eval_frac).


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
        feature_cost: float = 0.01 / 100,
        smooth_perf: int = 3,
        max_runtime_mins: float = None,
        max_refits: int = None,
        cv: Union[object, int, None] = 3,
        cv_shuffle: bool = True,
        early_stopping_val_nsplits: Union[int, None] = 4,
        early_stopping_rounds: Union[int, None] = None,
        scoring: Union[object, None] = None,
        top_predictors_search_method: OptimumSearch = OptimumSearch.ScipyLocal,
        votes_aggregation_method: VotesAggregation = VotesAggregation.Borda,
        use_all_fi_runs: bool = True,
        use_last_fi_run_only: bool = False,
        use_one_freshest_fi_run: bool = False,
        importance_getter: Union[str, Callable, None] = None,
        random_state: int = None,
        leave_progressbars: bool = True,
        verbose: Union[bool, int] = 0,
        show_plot: bool = False,
        cat_features: Union[Sequence, None] = None,
        keep_estimators: bool = False,
    ):

        # checks

        # assert isinstance(estimator, (BaseEstimator,))

        # save params

        params = get_parent_func_args()
        store_params_in_object(obj=self, params=params)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, pd.Series, np.ndarray], groups: Union[pd.Series, np.ndarray] = None):

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
        importance_getter = self.importance_getter
        random_state = self.random_state
        leave_progressbars = self.leave_progressbars
        verbose = self.verbose
        show_plot = self.show_plot
        cat_features = self.cat_features
        keep_estimators = self.keep_estimators
        feature_cost = self.feature_cost
        smooth_perf = self.smooth_perf

        start_time = timer()
        ran_out_of_time = False

        if random_state is not None:
            np.random.seed(random_state)
            cp.random.seed(random_state)
            set_random_seed(random_state)

        feature_importances = {}
        evaluated_scores_std = {}
        evaluated_scores_mean = {}

        if isinstance(X, pd.DataFrame):
            original_features = X.columns.tolist()
        else:
            original_features = np.arange(X.shape[1])

        # ensure cat_features contains indices
        if False and cat_features:
            cat_features = [(original_features.index(var) if isinstance(var, str) else var) for var in cat_features]
            print(cat_features)
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
                desc="RFECV iterations", leave=leave_progressbars, total=min(len(original_features), max_refits) if max_refits else len(original_features)
            )

        # ----------------------------------------------------------------------------------------------------------------------------
        # Init scoring
        # ----------------------------------------------------------------------------------------------------------------------------

        if scoring is None:
            if is_classifier(estimator):
                scoring = make_scorer(score_func=calib_error, needs_proba=True, needs_threshold=False, greater_is_better=False)
            elif is_regressor(estimator):
                scoring = make_scorer(score_func=mean_squared_error, needs_proba=False, needs_threshold=False, greater_is_better=False)
            else:
                raise ValueError(f"Scoring not known for estimator type: {estimator}")

        # ----------------------------------------------------------------------------------------------------------------------------
        # Init importance_getter
        # ----------------------------------------------------------------------------------------------------------------------------

        if importance_getter is None or importance_getter == "auto":
            importance_getter = "feature_importances_"

        # ----------------------------------------------------------------------------------------------------------------------------
        # Start evaluating different nfeatures, being guided by the selected search method
        # ----------------------------------------------------------------------------------------------------------------------------

        nsteps = 0
        dummy_scores = []
        fitted_estimators = {}
        selected_features_per_nfeatures = {}

        while nsteps < len(original_features):

            if verbose:
                iters_pbar.update(1)

            # ----------------------------------------------------------------------------------------------------------------------------
            # Select current set of features to work on, based on ranking received so far, and the search method
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
                top_predictors_search_method=top_predictors_search_method,
                votes_aggregation_method=votes_aggregation_method,
            )

            if current_features is None or len(current_features) == 0:
                break  # nothing more to try

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

                X_train, y_train, X_test, y_test = split_into_train_test(
                    X=X, y=y, train_index=train_index, test_index=test_index, features_indices=current_features
                )  # this splits both dataframes & ndarrays in the same fashion
                if val_cv:

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
                    # print("current_features=", current_features, "temp_cat_features=", temp_cat_features)
                    temp_fit_params = pack_val_set_into_fit_params(
                        model=estimator,
                        X_val=X_val,
                        y_val=y_val,
                        early_stopping_rounds=early_stopping_rounds,
                        cat_features=temp_cat_features,
                    )  # crafts fit params with early stopping tailored to particular model type.
                    temp_fit_params.update(fit_params)

                else:
                    X_val = None

                # ----------------------------------------------------------------------------------------------------------------------------
                # Fit our estimator on current train fold. Score on test & and get its feature importances.
                # ----------------------------------------------------------------------------------------------------------------------------

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

                key = f"{len(current_features)}_{nfold}"
                feature_importances[key] = fi

                if keep_estimators:
                    fitted_estimators[key] = fitted_estimator

                # print(f"feature_importances[step{len(current_features)}_fold{nfold}]=" + str({key: value for key, value in fi.items() if value > 0}))

                if 0 not in evaluated_scores_mean:

                    # ----------------------------------------------------------------------------------------------------------------------------
                    # Dummy baselines must serve as fitness @ 0 features.
                    # ----------------------------------------------------------------------------------------------------------------------------

                    best_dummy_score = -LARGE_CONST

                    if is_classifier(estimator):
                        dummy_model_type = DummyClassifier
                        strategies = "most_frequent prior stratified uniform"
                    elif is_regressor(estimator):
                        dummy_model_type = DummyRegressor
                        strategies = "mean median"
                    else:
                        strategies = None
                        if verbose:
                            logger.info(f"Unexpected estimator type: {estimator}")

                    if strategies:
                        for strategy in strategies.split():
                            model = dummy_model_type(strategy=strategy)
                            model.fit(X=X_train, y=y_train)
                            dummy_score = scoring(model, X_test, y_test)
                            if score > best_dummy_score:
                                best_dummy_score = dummy_score

                    dummy_scores.append(best_dummy_score)
                    # print(f"Best dummy score (at 0 features, fold {nfold}): {best_dummy_score}")

            if 0 not in evaluated_scores_mean:
                store_averaged_cv_scores(pos=0, scores=dummy_scores, evaluated_scores_mean=evaluated_scores_mean, evaluated_scores_std=evaluated_scores_std)

            store_averaged_cv_scores(
                pos=len(current_features), scores=scores, evaluated_scores_mean=evaluated_scores_mean, evaluated_scores_std=evaluated_scores_std
            )

            # ----------------------------------------------------------------------------------------------------------------------------
            # Checking exit conditions
            # ----------------------------------------------------------------------------------------------------------------------------

            nsteps += 1

            if max_runtime_mins and not ran_out_of_time:
                ran_out_of_time = (timer() - start_time) > max_runtime_mins * 60
                if ran_out_of_time:
                    if verbose:
                        logger.info(f"max_runtime_mins={max_runtime_mins:_.1f} reached.")
                    break

            if max_refits and nsteps >= max_refits:
                if verbose:
                    logger.info(f"max_refits={max_refits:_} reached.")
                break

        # ----------------------------------------------------------------------------------------------------------------------------
        # Saving best result found so far as final
        # ----------------------------------------------------------------------------------------------------------------------------

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns if isinstance(X, pd.DataFrame) else map(str, np.arange(self.n_features_in_))

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
            feature_cost=feature_cost,
            smooth_perf=smooth_perf,
            use_all_fi_runs=use_all_fi_runs,
            use_last_fi_run_only=use_last_fi_run_only,
            use_one_freshest_fi_run=use_one_freshest_fi_run,
            votes_aggregation_method=votes_aggregation_method,
            verbose=verbose,
            show_plot=show_plot,
        )

        return self

    def select_optimal_nfeatures_(
        self,
        checked_nfeatures: np.ndarray,
        cv_mean_perf: np.ndarray,
        cv_std_perf: np.ndarray,
        feature_cost: float = 0.01 / 100,
        smooth_perf: int = 3,
        use_all_fi_runs: bool = True,
        use_last_fi_run_only: bool = False,
        use_one_freshest_fi_run: bool = False,
        votes_aggregation_method: VotesAggregation = VotesAggregation.Borda,
        verbose: bool = False,
        comparison_base: float = 10,
        show_plot: bool = False,
        plot_file=None,
        font_size: int = 12,
        figsize: tuple = (10, 7),
    ):

        base_perf = np.array(cv_mean_perf) - np.array(cv_std_perf) / 2
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

        # last time vote for feature_importances using all info up to date

        fi_to_consider = select_appropriate_feature_importances(
            feature_importances=self.feature_importances_,
            nfeatures=best_top_n,
            n_original_features=self.n_features_in_,
            use_all_fi_runs=use_all_fi_runs,
            use_last_fi_run_only=use_last_fi_run_only,
            use_one_freshest_fi_run=use_one_freshest_fi_run,
        )

        self.ranking_ = get_actual_features_ranking(
            feature_importances=fi_to_consider,
            votes_aggregation_method=votes_aggregation_method,
        )

        self.support_ = np.array([(i in self.ranking_[:best_top_n]) for i in self.feature_names_in_])

        if verbose:
            logger.info(f"{self.n_features_:_} predictive factors selected out of {self.n_features_in_:_} during {len(self.selected_features_):_} rounds.")

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            return X[self.support_names_]
        else:
            return X[self.support_]


def split_into_train_test(
    X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray], train_index: np.ndarray, test_index: np.ndarray, features_indices: np.ndarray = None
) -> tuple:
    """Split X & y according to indices & dtypes. Basically this accounts for diffeent dtypes (pd.DataFrame, np.ndarray) to perform the same."""

    if isinstance(X, pd.DataFrame):
        X_train, y_train = (X.iloc[train_index, :] if features_indices is None else X.iloc[train_index, :][features_indices]), (
            y.iloc[train_index, :] if isinstance(y, pd.DataFrame) else y.iloc[train_index]
        )
        X_test, y_test = (X.iloc[test_index, :] if features_indices is None else X.iloc[test_index, :][features_indices]), (
            y.iloc[test_index, :] if isinstance(y, pd.DataFrame) else y.iloc[test_index]
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
    evaluated_scores_mean[pos] = np.mean(scores)
    evaluated_scores_std[pos] = np.std(scores)


def get_feature_importances(
    model: object,
    current_features: list,
    importance_getter: Union[str, Callable],
    data: Union[pd.DataFrame, np.ndarray, None] = None,
    reference_data: Union[pd.DataFrame, np.ndarray, None] = None,
) -> dict:

    if isinstance(importance_getter, str):
        res = getattr(model, importance_getter)
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
    top_predictors_search_method: OptimumSearch = OptimumSearch.ScipyLocal,
    votes_aggregation_method: VotesAggregation = VotesAggregation.Borda,
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

            if next_nfeatures_to_check:

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
                )
                ranks = get_actual_features_ranking(feature_importances=fi_to_consider, votes_aggregation_method=votes_aggregation_method)

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
    return fi_to_consider


def get_actual_features_ranking(feature_importances: dict, votes_aggregation_method: VotesAggregation) -> list:

    """Absolute FIs from estimators trained on CV for each nfeatures are stored separatly.
    They can be used to recompute final voted importances using any desired voting algo.
    But of course the exploration path was already lead by specific voting algo active at the fitting time.
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
        ranks = lb.optimality_gap_ranking()
    elif votes_aggregation_method == VotesAggregation.Plurality:
        ranks = lb.plurality_ranking()

    # print("Current features ranks:")
    # print(ranks)
    return ranks.index.values.tolist()
