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


    Main problem: impactful, but correlated factors all get low importance and will be thrown away (probably only for forests, not boostings).
    Other problem: due to noise some random features can become "important". Solution: use CV to calculate fold FI, then combine across the folds (by voting).
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
        classes_ : ndarray of shape (n_classes,)
            The classes labels. Only available when `estimator` is a classifier.

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

        .. versionadded:: 1.0

    n_features_ : int
        The number of selected features with cross-validation.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    ranking_ : narray of shape (n_features,)
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
        cat_features: Union[Sequence, None] = None,
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
        cat_features = self.cat_features

        start_time = timer()
        run_out_of_time = False

        if random_state is not None:
            np.random.seed(random_state)
            cp.random.seed(random_state)
            set_random_seed(random_state)

        feature_importances = {}
        evaluated_scores_mean = {}
        evaluated_scores_std = {}

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
                early_stopping_rounds = 20  # TODO: derive as 1/5 of nestimators
        else:
            val_cv = None

        if verbose:
            iters_pbar = tqdmu(desc="RFECV iterations", leave=leave_progressbars, total=len(original_features))

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

        nsteps = 0
        dummy_scores = []
        selected_features = {}
        while nsteps < len(original_features):

            if verbose:
                iters_pbar.update(1)

            # ----------------------------------------------------------------------------------------------------------------------------
            # Each split better be different. so, even if random_state is provided, random_state to the cv is generated separately
            # (and deterministically) each time based on the original random_state.
            # ----------------------------------------------------------------------------------------------------------------------------

            scores = []

            splitter = cv.split(X=X, y=y, groups=groups)
            if verbose:
                splitter = tqdmu(splitter, desc="CV folds", leave=False, total=cv.n_splits)

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
                break

            selected_features[len(current_features)] = current_features

            for nfold, (train_index, test_index) in enumerate(splitter):

                X_train, y_train, X_test, y_test = split_into_train_test(
                    X=X, y=y, train_index=train_index, test_index=test_index, features_indices=current_features
                )

                if val_cv:

                    # ----------------------------------------------------------------------------------------------------------------------------
                    # Make additional early stoppping split
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
                    # If estimator is known, apply early stopping
                    # ----------------------------------------------------------------------------------------------------------------------------

                    fit_params = pack_val_set_into_fit_params(
                        model=estimator,
                        X_val=X_val,
                        y_val=y_val,
                        early_stopping_rounds=early_stopping_rounds,
                        fit_params=fit_params,
                        cat_features=[var for var in cat_features if var in current_features],
                    )

                else:
                    X_val = None

                estimator.fit(X=X_train, y=y_train, **fit_params)
                score = scoring(estimator, X_test, y_test)
                scores.append(score)
                fi = get_feature_importances(
                    model=estimator, current_features=current_features, data=X_test, reference_data=X_val, importance_getter=importance_getter
                )
                feature_importances[f"{nsteps}_{nfold}"] = fi

                print(f"feature_importances[{nsteps}_{nfold}]=" + str({key: value for key, value in fi.items() if value > 0}))

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

            if 0 not in evaluated_scores_mean:
                add_scores(pos=0, scores=dummy_scores, evaluated_scores_mean=evaluated_scores_mean, evaluated_scores_std=evaluated_scores_std)

            add_scores(pos=len(current_features), scores=scores, evaluated_scores_mean=evaluated_scores_mean, evaluated_scores_std=evaluated_scores_std)

            # ----------------------------------------------------------------------------------------------------------------------------
            # Checking exit conditions
            # ----------------------------------------------------------------------------------------------------------------------------

            nsteps += 1

            if max_runtime_mins and not run_out_of_time:
                run_out_of_time = (timer() - start_time) > max_runtime_mins * 60
                if run_out_of_time:
                    logger.info(f"max_runtime_mins={max_runtime_mins:_.1f} reached.")
                    break

            if max_refits and nsteps >= max_refits:
                if verbose:
                    logger.info(f"max_refits={max_refits:_} reached.")
                break

        # ----------------------------------------------------------------------------------------------------------------------------
        # Saving best result found so far as final
        # ----------------------------------------------------------------------------------------------------------------------------

        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Mean test score")

        checked_top_n = sorted(evaluated_scores_mean.keys())
        cv_mean_perf = [evaluated_scores_mean[n] for n in checked_top_n]
        cv_std_perf = [evaluated_scores_std[n] for n in checked_top_n]

        plt.errorbar(
            checked_top_n,
            cv_mean_perf,
            yerr=cv_std_perf,
        )
        plt.title("Recursive Feature Elimination \nwith correlated features")
        plt.show()

        ultimate_perf = np.array(cv_mean_perf) - np.array(cv_std_perf) / 2
        best_top_n = checked_top_n[np.argmax(ultimate_perf)]

        # self.support_ = summary["selected_features"]
        self.selected_features = selected_features[best_top_n]
        self.n_features_ = best_top_n

        # last time vote for feature_importances using all info up to date
        self.feature_importances_ = get_actual_features_ranking(feature_importances=feature_importances, votes_aggregation_method=votes_aggregation_method)

        if verbose:
            logger.info(f"{self.n_features_:_} predictive factors selected out of {len(original_features):_} during {nsteps:_} rounds.")

        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            return X[self.support_names_]
        else:
            return X[self.support_]


def split_into_train_test(
    X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray], train_index: np.ndarray, test_index: np.ndarray, features_indices: np.ndarray = None
) -> tuple:
    """Split X & y according to indices & dtypes."""

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


def add_scores(pos: int, scores: list, evaluated_scores_mean: dict, evaluated_scores_std: dict) -> None:
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
                next_top_n = random.choice(remaining)

            if next_top_n:

                # ----------------------------------------------------------------------------------------------------------------------------
                # At each step, feature importances must be recalculated in light of recent training on a smaller subset.
                # The features already thrown away all receive constant importance update of the sanme size, to keep up with number of trains.
                # ----------------------------------------------------------------------------------------------------------------------------
                if use_last_fi_run_only:
                    fi_to_consider = {key: value for key, value in feature_importances.items() if len(value) == len(original_features)}
                else:
                    if use_all_fi_runs:
                        fi_to_consider = feature_importances
                    else:
                        if use_one_freshest_fi_run:
                            for key, value in feature_importances.items():
                                if len(value) >= next_top_n:
                                    print(f"using FI of {len(value)} features for next_top_n={next_top_n}")
                                    fi_to_consider = {key: value}
                                    break
                        else:
                            fi_to_consider = {key: value for key, value in feature_importances.items() if len(value) > next_top_n}

                ranks = get_actual_features_ranking(feature_importances=fi_to_consider, votes_aggregation_method=votes_aggregation_method)

                print(f"next_top_n={next_top_n}, features chosen={ranks[:next_top_n]}")

                return ranks[:next_top_n]


def get_actual_features_ranking(feature_importances: dict, votes_aggregation_method: VotesAggregation) -> list:

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

    print(ranks)
    return ranks.index.values.tolist()
