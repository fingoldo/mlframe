"""Feature selection within ML pipelines. Wrappers method. Currently includes recursive feature elimination."""

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
        from sklearn.base import is_classifier, is_regressor, BaseEstimator, TransformerMixin
        from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit, GroupKFold, GroupShuffleSplit

        from enum import Enum, auto
        from timeit import default_timer as timer
        from pyutilz.numbalib import set_random_seed

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
    Brent = "Brent"
    SurrogateModel = "SurrogateModel"
    GaussianProcess = "GaussianProcess"
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
    """Finds subset of features having best CV score, by iterative narrowing down set of candidates having highest importance, as per estimator.

    Optimizes mean CV scores (possibly translated into ranks) divided by the features number.

    Uses several optimization methods:
        exhaustive search
        random search
        bayesian search.


    Main problem: impactful, but correlated factors all get low importance and will be thrown away (probably only for forests, not boostings).
    Other problem: due to noise some random features can become "important". Solution: use CV to calculate fold FI, then combine across the folds (by voting).


    Optionally creates (and saves) a chart of optimization path.

    Challenges:
        CV performance itself can be a multi-component value! Say, both ROC AUC and CALIB metrics can be considered. Voting can be a solution.
        estimator might itself be a HPT search.
        it could be good to have several estimators. their importance evaluations must be accounted for simultaneously (voting).
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
        shuffle: bool = True,
        scoring: Union[str, Callable, None] = None,
        top_predictors_search_method: OptimumSearch = OptimumSearch.Brent,
        votes_aggregation_method: VotesAggregation = VotesAggregation.Borda,
        importance_getter: Union[str, Callable] = "auto",
        random_state: int = None,
        leave_progressbars: bool = True,
        verbose: Union[bool, int] = 0,
    ):

        # checks

        assert isinstance(estimator, (BaseEstimator,))

        # save params

        self.cv_ = cv
        self.verbose_ = verbose
        self.estimator_ = estimator
        self.max_refits_ = max_refits
        self.max_runtime_mins_ = max_runtime_mins
        self.votes_aggregation_method_ = votes_aggregation_method
        self.top_predictors_search_method_ = top_predictors_search_method

    def fit(self, X, y, groups=None):

        # ---------------------------------------------------------------------------------------------------------------
        # Inits
        # ---------------------------------------------------------------------------------------------------------------

        start_time = timer()
        run_out_of_time = False

        if random_seed is not None:
            np.random.seed(random_seed)
            cp.random.seed(random_seed)
            set_random_seed(random_seed)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Init cv
        # ----------------------------------------------------------------------------------------------------------------------------

        if cv is None or str(cv).isnumeric():
            if cv is None:
                cv = 3
            if is_classifier(estimator):
                if groups is not None:
                    cv = StratifiedGroupKFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
                else:
                    cv = StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
            else:
                if groups is not None:
                    cv = GroupKFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
                else:
                    cv = KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)

        if verbose:
            iters_pbar = tqdmu(desc="RFECV iterations", leave=leave_progressbars)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Dummy baselines must serve as fitness @ 0 features.
        # ----------------------------------------------------------------------------------------------------------------------------

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
            # Each split better be different. so, even if random_seed is provided, random_seed to the cv is generated separtely (and deterministically) each time based on the original random_seed.
            # ----------------------------------------------------------------------------------------------------------------------------

            splitter = group_kfold.split(X=X, y=y, groups=groups)
            if verbose:
                cv_pbar = tqdmu(desc="CV folds", leave=leave_progressbars)

            for i, (train_index, test_index) in enumerate(splitter):
                if isinstance(X, pd.DataFrame):
                    estimator.fit(X=X.iloc[train_index, :], y=y.iloc[train_index, :], **fit_params)
                else:
                    estimator.fit(X=X[train_index, :], y=y[train_index, :], **fit_params)

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
