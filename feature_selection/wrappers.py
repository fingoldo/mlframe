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
        from pyutilz.pythonlib import store_func_params_in_object
        from sklearn.base import is_classifier, is_regressor, BaseEstimator, TransformerMixin
        from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit, GroupKFold, GroupShuffleSplit
        from sklearn.dummy import DummyClassifier, DummyRegressor
        from sklearn.metrics import make_scorer, mean_squared_error

        from mlframe.metrics import calib_error

        from enum import Enum, auto
        from timeit import default_timer as timer
        from pyutilz.numbalib import set_random_seed

        from votenrank import Leaderboard

    except Exception as e:

        logger.warning(e)

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


class OptimumSearch(Enum):
    ScipyLocal = "ScipyLocal"  # Brent
    ScipyGlobal = "ScipyGlobal"  # direct, diff evol, shgo
    SurrogateModel = "SurrogateModel"  # GaussianProcess or Catboost with uncertainty, or quantile regression
    ExhaustiveRandom = "ExhaustiveRandom"
    ExhaustiveDichotomic = "ExhaustiveDichotomic"


class VotesAggregation(Enum):
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
        estimator: object,
        fit_params: dict = {},
        max_runtime_mins: float = None,
        max_refits: int = None,
        cv: Union[object, int, None] = 3,
        cv_shuffle: Union[bool, None] = None,
        early_stopping_val_nsplits: Union[int, None] = 4,
        scoring_func: Union[Callable, None] = None,
        scoring_greater_is_better: Union[bool, None] = None,
        scoring_needs_proba: Union[bool, None] = None,
        scoring_needs_threshold: Union[bool, None] = None,
        top_predictors_search_method: OptimumSearch = OptimumSearch.ScipyLocal,
        votes_aggregation_method: VotesAggregation = VotesAggregation.Borda,
        importance_getter: Union[str, Callable] = "auto",
        random_state: int = None,
        leave_progressbars: bool = True,
        verbose: Union[bool, int] = 0,
    ):

        # checks

        assert isinstance(estimator, (BaseEstimator,))

        # save params

        store_func_params_in_object(obj=self)

    def fit(self, X, y, groups=None):

        # ---------------------------------------------------------------------------------------------------------------
        # Inits
        # ---------------------------------------------------------------------------------------------------------------

        load_object_params_into_func(obj=self, locals=locals())

        start_time = timer()
        run_out_of_time = False

        if random_seed is not None:
            np.random.seed(random_seed)
            cp.random.seed(random_seed)
            set_random_seed(random_seed)

        current_FIs = np.zeros(X.shape[1], dtype=np.float32)
        evaluated_FIs = {}

        model_type_name = type(estimator).__name__

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
            iters_pbar = tqdmu(desc="RFECV iterations", leave=leave_progressbars)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Init scoring
        # ----------------------------------------------------------------------------------------------------------------------------

        if scoring_func is None:
            if is_classifier(estimator):
                scoring_func = calib_error
                scoring_needs_proba = True
                scoring_needs_threshold = False
                scoring_greater_is_better = False
            elif is_regressor(estimator):
                scoring_func = mean_squared_error
                scoring_needs_proba = False
                scoring_needs_threshold = False
                scoring_greater_is_better = False
            else:
                raise ValueError(f"Unexpected estimator type: {estimator}")

        # ----------------------------------------------------------------------------------------------------------------------------
        # Dummy baselines must serve as fitness @ 0 features.
        # ----------------------------------------------------------------------------------------------------------------------------

        for i, (train_index, test_index) in enumerate(cv.split(X=X, y=y, groups=groups)):

            if is_classifier(estimator):
                for strategy in "most_frequent prior stratified uniform".split():
                    model = DummyClassifier(strategy=strategy)
                    model.fit(X=None, y=Y_train[target_name])
                    score = scoring_func(y_true=,y_pred=)
        
        # ----------------------------------------------------------------------------------------------------------------------------
        # First step is to try all features.
        # ----------------------------------------------------------------------------------------------------------------------------

        nsteps = 0
        while True:
            if verbose:
                iters_pbar.n += 1

            # ----------------------------------------------------------------------------------------------------------------------------
            # At each step, feature importances must be recalculated in light of recent training on a smaller subset.
            # The features already thrown away all receive constant importance update of the sanme size, to keep up with number of trains.
            # ----------------------------------------------------------------------------------------------------------------------------

            # ----------------------------------------------------------------------------------------------------------------------------
            # Each split better be different. so, even if random_seed is provided, random_seed to the cv is generated separately (and deterministically) each time based on the original random_seed.
            # ----------------------------------------------------------------------------------------------------------------------------

            def estimate_fis_and_cv_perf(X, y, groups, cv, verbose, leave_progressbars):

                splitter = cv.split(X=X, y=y, groups=groups)
                if verbose:
                    splitter = tqdmu(splitter, desc="CV folds", leave=leave_progressbars)

                for i, (train_index, test_index) in enumerate(splitter):
                    
                    if isinstance(X, pd.DataFrame):
                        X_train, y_train = X.iloc[train_index, :], y = y.iloc[train_index, :]
                        X_test, y_test = X.iloc[test_index, :], y = y.iloc[test_index, :]
                    else:
                        X_train, y_train = X[train_index, :], y = y[train_index, :]
                        X_test, y_test = X[test_index, :], y = y[test_index, :]

                    if early_stopping_val_nsplits:

                        # if estimator is known, apply early stopping

                        if model_type_name in XGBOOST_MODEL_TYPES:
                            fit_kwargs = dict(verbose=verbose)

                            if ES == EarlyStopping.NativeOrNo:            
                                model.set_params(early_stopping_rounds=early_stopping_rounds)

                                if pipe:
                                    X_val_processed= pre_pipeline.transform(X_val)
                                else:
                                    X_val_processed=X_val
                            
                                fit_kwargs["eval_set"] = ((X_val_processed, Y_val),)                        

                    estimator.fit(X=X_train, y=y_train, **fit_params)

            # ----------------------------------------------------------------------------------------------------------------------------
            # Checking exit conditions
            # ----------------------------------------------------------------------------------------------------------------------------

            if max_runtime_mins and not run_out_of_time:
                run_out_of_time = (timer() - start_time) > max_runtime_mins * 60
                if run_out_of_time:
                    logger.info("max_runtime_mins={max_runtime_mins:_.1f} reached.")
                    break

            if max_refits and nsteps >= max_refits:
                if verbose:
                    logger.info("max_refits={max_refits:_} reached.")
                break

        # ----------------------------------------------------------------------------------------------------------------------------
        # Saving best result found so far as final
        # ----------------------------------------------------------------------------------------------------------------------------

        self.support_ = summary["selected_features"]
        self.support_names_ = summary["selected_features_names"]
        self.n_features_ = len(self.support_)

        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            return X[self.support_names_]
        else:
            return X[self.support_]
